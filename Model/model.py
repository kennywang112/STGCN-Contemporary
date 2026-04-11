# This file integrates layers.py and base_model.py from the original repository

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.gcgru import GConvGRU

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func

        # Corresponds to bottleneck down-sampling in TF
        if c_in > c_out:
            self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        
        # GLU requires doubling the output channels
        if act_func == 'GLU':
            self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt))
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, Kt))

    def forward(self, x):
        # x shape: [Batch, Time, Node, Channel]
        
        # Residual connection alignment
        if self.c_in > self.c_out:
            x_input = self.align_conv(x)

        elif self.c_in < self.c_out:
            # Padding zeros to match channel dimensions
            batch_size, _, n_nodes, time_steps = x.shape
            padding = torch.zeros(batch_size, self.c_out - self.c_in, n_nodes, time_steps).to(x.device)
            x_input = torch.cat([x, padding], dim=1)

        else:
            x_input = x

        # Truncate time dimension to match the convolution output size
        x_input = x_input[:, :, :, self.Kt - 1:]

        # Execute temporal convolution
        x_conv = self.conv(x)

        if self.act_func == 'GLU':
            # Exact replication of TF GLU gate mechanism
            # (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
            p = x_conv[:, :self.c_out, :, :]
            q = x_conv[:, self.c_out:, :, :]
            return (p + x_input) * torch.sigmoid(q)
        
        elif self.act_func == 'relu':
            return F.relu(x_conv + x_input)
        
        elif self.act_func == 'sigmoid':
            return torch.sigmoid(x_conv)
        
        elif self.act_func == 'linear':
            return x_conv

class SpatioConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        
        if c_in > c_out:
            self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
            
        self.theta = nn.Parameter(torch.FloatTensor(Ks * c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x, graph_kernel):
        # Align channels
        if self.c_in > self.c_out:
            x_input = self.align_conv(x)

        elif self.c_in < self.c_out:
            batch_size, _, n_nodes, time_steps = x.shape
            padding = torch.zeros(batch_size, self.c_out - self.c_in, n_nodes, time_steps).to(x.device)
            x_input = torch.cat([x, padding], dim=1)

        else:
            x_input = x

        # Graph onvolution matrix multiplication
        batch_size, _, n_nodes, time_steps = x.shape
        
        # Reshape for matrix multiplication
        x_tmp = x.permute(0, 3, 1, 2).reshape(-1, n_nodes)
        x_mul = torch.matmul(x_tmp, graph_kernel).reshape(-1, self.c_in, self.Ks, n_nodes)
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(-1, self.c_in * self.Ks)
        x_gconv = torch.matmul(x_ker, self.theta).reshape(-1, n_nodes, self.c_out)
        
        x_gc = x_gconv.reshape(batch_size, time_steps, n_nodes, self.c_out).permute(0, 3, 2, 1)
        return F.relu(x_gc + x_input)

class STConvBlock(nn.Module):
    def __init__(self, Ks, Kt, channels, n_nodes, drop_rate=0.5):
        super(STConvBlock, self).__init__()
        c_si, c_t, c_oo = channels
        
        self.t_conv1 = TemporalConvLayer(Kt, c_si, c_t, act_func='GLU')
        self.s_conv = SpatioConvLayer(Ks, c_t, c_t)
        self.t_conv2 = TemporalConvLayer(Kt, c_t, c_oo, act_func='linear')
        self.layer_norm = nn.LayerNorm([n_nodes, c_oo])
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, graph_kernel):
        x_s = self.t_conv1(x)
        x_t = self.s_conv(x_s, graph_kernel)
        x_o = self.t_conv2(x_t)
        
        # Layer Norm requires the channel dimension to be last, then permute back
        x_o = x_o.permute(0, 3, 2, 1)
        x_ln = self.layer_norm(x_o).permute(0, 3, 2, 1)
        return self.dropout(x_ln)

class STGCN(nn.Module):
    def __init__(self, Ks, Kt, blocks, n_his, n_nodes):
        super(STGCN, self).__init__()
        self.blocks = nn.ModuleList()
        for channels in blocks:
            self.blocks.append(STConvBlock(Ks, Kt, channels, n_nodes))
            
        Ko = n_his - 2 * len(blocks) * (Kt - 1)
        
        c_out = blocks[-1][-1]
        self.t_conv_out1 = TemporalConvLayer(Ko, c_out, c_out, act_func='GLU')
        self.layer_norm_out = nn.LayerNorm([n_nodes, c_out])
        self.t_conv_out2 = TemporalConvLayer(1, c_out, c_out, act_func='relu')
        
        # Corresponds to TF output_layer and fully_con_layer
        self.fc = nn.Conv2d(c_out, 1, kernel_size=(1, 1))

    def forward(self, x, graph_kernel):
        for block in self.blocks:
            x = block(x, graph_kernel)
            
        x = self.t_conv_out1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm_out(x).permute(0, 3, 2, 1)
        x = self.t_conv_out2(x)
        out = self.fc(x)
        return out

class GCGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K):
        super(GCGRU, self).__init__()
        self.rnn = GConvGRU(in_channels=in_channels, out_channels=hidden_channels, K=K)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, X, edge_index, edge_weight):
        # [Batch, Channel, Node, Time] -> [30, 1, 43, 12]
        batch_size = X.size(0)
        num_nodes = X.size(2)
        num_timesteps = X.size(3)

        batched_edge_indices = []
        for b in range(batch_size):
            batched_edge_indices.append(edge_index + b * num_nodes)

        batched_edge_index = torch.cat(batched_edge_indices, dim=1)
        batched_edge_weight = edge_weight.repeat(batch_size)

        # [Batch, Channel, Node, Time] -> [Batch, Node, Channel, Time]
        X = X.permute(0, 2, 1, 3)
        # [Batch * Node, Channel, Time]
        X = X.reshape(batch_size * num_nodes, X.size(2), num_timesteps)

        H = torch.zeros(batch_size * num_nodes, self.rnn.out_channels, device=X.device)
        
        for t in range(num_timesteps):
            # [Batch*Node, Channel]
            x_t = X[:, :, t] 
            H = self.rnn(x_t, batched_edge_index, batched_edge_weight, H)
            
        out = self.linear(H) # [Batch*Node, out_channels]

        # [Batch, Node, out_channels]
        out = out.view(batch_size, num_nodes, -1)
        out = out.permute(0, 2, 1)
        # [Batch, out_channels, Node, 1]
        out = out.unsqueeze(-1)
        
        return out