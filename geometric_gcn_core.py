import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN(nn.Module):
    def __init__(self, mlp_list: List[int], k: int, aggr: str = 'max'):
        super(DGCNN, self).__init__()
        self.k = k
        self.aggr = aggr 

        # First edge conv layers using graph features
        self.conv_layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        for i in range(len(mlp_list) - 1):
            in_channels = mlp_list[i] * 2 # for edge feature [x_i || x_j - x_i]
            out_channels = mlp_list[i + 1]

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )

        #embedding the last value in mlp_list
        emb_dims = mlp_list[-1]
        self.conv5 = nn.Sequential(
            nn.Conv1d(sum(mlp_list[1:]), emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 3) 

    def forward(self, x, batch=None):
        batch_size = x.size(0)
        features = []
        #print(f"DGCNN.forward - input x          = {tuple(x.shape)}")

        for idx, conv in enumerate(self.conv_layers):
            x_graph = get_graph_feature(x, k=self.k)
            #print(f"  - after get_graph_feature [{idx}] = {tuple(x_graph.shape)}")
            x = conv(x_graph)
            #print(f"  - after conv{idx}            = {tuple(x.shape)}")
            x = x.max(dim=-1)[0]
            #print(f"  - after max(dim=-1)          = {tuple(x.shape)}")
            features.append(x)

        x = torch.cat(features, dim=1)
        #print(f"  - after torch.cat(features)  = {tuple(x.shape)}")
        x = self.conv5(x)
        #print(f"  - after conv5 (1d)           = {tuple(x.shape)}")

        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        #print(f"  - after adaptive_max_pool1d  = {tuple(x1.shape)}")
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        #print(f"  - after adaptive_avg_pool1d  = {tuple(x2.shape)}")
        x = torch.cat((x1, x2), dim=1)
        #print(f"  - after final torch.cat      = {tuple(x.shape)}")

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        #print(f"  - final output               = {tuple(x.shape)}")
        return x
    


#test code for DGCNN class; returns torch.Size([4,3])
# mlp_list = [3, 64, 128]  
# k = 20
# aggr = 'max'  
# model = DGCNN(mlp_list=mlp_list, k=k, aggr=aggr)
# x = torch.rand(4, 3, 468)
# output = model(x)
# print("Output shape:", output.shape)


def global_max_pool(x, batch):
    B = batch.max().item() + 1
    out = torch.full((B, x.size(1)), float('-inf'), device=x.device)
    out = out.scatter_reduce(0, batch[:, None].expand(-1, x.size(1)), x, reduce='amax', include_self=True)
    return out


class EdgeCNNMT(nn.Module):
    def __init__(self, mlp_list: List[int], k: int, aggr: str, num_aus: int = 24, num_classes: int = 3):
        super().__init__()
        self.k = k
        self.aggr = aggr

        block1_mlp = mlp_list[:-2]
        self.block1 = DGCNN( mlp_list=block1_mlp, k=k, aggr=aggr )
        dim1 = sum(block1_mlp[1:]) 
        block2_mlp = [ dim1, mlp_list[-1] ] 
        self.block2 = DGCNN(mlp_list=block2_mlp, k=k, aggr=aggr)
        dim2 = mlp_list[-1] 
        self.lin1 = nn.Linear(dim1 + dim2, 512)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            ) for _ in range(num_aus)
        ])


    def forward(self, x, batch):
        B,C,N = x.shape
        #print(f"MT.forward - input x={x.shape}, batch={batch.shape}")

        out = x
        feats1 = []
        for idx, conv in enumerate(self.block1.conv_layers):
            g = get_graph_feature(out, k=self.k)    #[B,2*C, N, k]
            #print(f"  block1 get_graph_feature[{idx}] = {g.shape}")
            out = conv(g).max(dim=-1)[0]               #[B, out_c, N]
            #print(f"  block1 conv{idx} output          = {out.shape}")
            feats1.append(out)
        f1 = torch.cat(feats1, dim=1)                  #[B, dim1, N]
        #print(f"  block1 concatenated f1           = {f1.shape}")

        out = f1
        feats2 = []
        for idx, conv in enumerate(self.block2.conv_layers):
            g = get_graph_feature(out, k=self.k)    #[B,2 * dim1, N, k]
            #print(f"  block2 get_graph_feature[{idx}] = {g.shape}")
            out = conv(g).max(dim=-1)[0]               #[B, out_c, N]
            #print(f"  block2 conv{idx} output          = {out.shape}")
            feats2.append(out)
        f2 = torch.cat(feats2, dim=1)                  #[B, dim2, N]
        #print(f"  block2 concatenated f2           = {f2.shape}")
        
        cat = torch.cat([f1, f2], dim=1)           
        #print(f"  combined cat                     = {cat.shape}")
        pooled = cat.max(dim=2)[0]                    
        #print(f"  pooled over N                    = {pooled.shape}")
        proj = self.lin1(pooled)                    
        #print(f"  after lin1(proj)                 = {proj.shape}")

        out_preds = []
        for i, head in enumerate(self.heads):
            h = head(proj).unsqueeze(2)                
            #print(f"  head[{i}] output                 = {h.shape}")
            out_preds.append(h)
        out = torch.cat(out_preds, dim=2)              
        #print(f"  final out                        = {out.shape}")

        return out
    
    @property
    def name(self):
        return "EdgeCNNMT"




#test code for EdgeCNNMT class; returns torch.Size([4,3,16]) 
# B, C, N = 4, 3, 128        # batch, channel, num points
# mlp_list = [3, 64, 128, 256, 512]
# k = 20
# aggr = 'max'
# num_aus = 16
# num_classes = 3

# x = torch.rand(B, C, N)
# batch = torch.arange(B).repeat_interleave(N)
# model = EdgeCNNMT(mlp_list = mlp_list, k = k, aggr = aggr, num_aus = num_aus, num_classes = num_classes)
# out = model(x, batch)
# print("Output shape:", out.shape)
#print("Output tensor:\n", out)









