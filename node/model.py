import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        self.to_q = nn.Linear(out_ft, out_ft, bias=False)
        self.to_k = nn.Linear(out_ft, out_ft, bias=False)
        self.to_v = nn.Sequential(nn.Linear(out_ft, out_ft, bias=False),
                                 nn.PReLU(),
                                 nn.Linear(out_ft, out_ft, bias=False)
                                 )
        self.attn = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False, attn=False, training=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            if attn:
                out = torch.bmm(adj, seq_fts)
                out_q = F.normalize(self.to_q(out), dim=-1)
                out_k = F.normalize(self.to_k(out), dim=-1)
                attn_mx = torch.einsum('bmd, bnd->bmn', (out_q, out_k))
#                 attn_mx = torch.randn(adj.size()).cuda()
                div_loss = ((out_q - out_k)**2).sum(dim=-1).mean()
            
                zero_vec = -9e15*torch.ones_like(attn_mx)
                attention = torch.where(adj > 0, attn_mx, zero_vec)
                attention = F.softmax(attention, dim=-1)
                attention = F.dropout(attention, 0.1, training=training)
                out = self.to_v(torch.bmm(attention, out))
                return self.act(out), div_loss
#                 out = torch.bmm(adj, seq_fts)
#                 return self.act(out), 0
            else:
                out = torch.bmm(adj, seq_fts)
                return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


class Model(nn.Module):
    def __init__(self, n_in, n_h, projection=False):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()
        self.new_dis = DIS(n_h, 256, projection)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1, div_loss = self.gcn1(seq1, adj, sparse, attn=True,training=True)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)
        
        h_2 = self.gcn2(seq1, diff, sparse, attn=False)
        
        ret = self.new_dis(h_1, h_2)

        return ret, h_1, h_2, div_loss

    def embed(self, seq, adj, diff, sparse, msk):
        h_1, div_loss = self.gcn1(seq, adj, sparse, attn=True, training=False)
        h_2 = self.gcn2(seq, diff, sparse, attn=False)
        
        c = self.read(h_1, msk)
        return (h_1 + h_2).detach(), c.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret

class GEN(nn.Module):
    def __init__(self, out_dim, z_dim, contra_dim):
        super(GEN, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(out_dim+z_dim, out_dim, bias=False),
            nn.PReLU(),
            nn.Linear(out_dim, contra_dim, bias=False)
        )
    def forward(self, x, noise):
        """
        :param x: batch * out_dim
        :param noise: batch * z_dim
        :return: batch * out_dim
        """
        feature = torch.cat([x, noise], dim=-1)
        return self.gen(feature)

class DIS(nn.Module):
    def __init__(self, n_out, n_c, projection=False):
        super(DIS, self).__init__()
#         self.fc = nn.Linear(out_ft * 2, 1, bias=False)
        self.head = projection
        self.projection = nn.Sequential(
                            nn.Linear(n_out, n_c),
                            nn.PReLU(),
                            nn.Linear(n_c, n_c)
                            
        )
    def forward(self, graph1, graph2):
        if self.head:
            graph1 = self.projection(graph1)
            graph2 = self.projection(graph2)
        
        # B * N * N
        graph1 = F.normalize(graph1, dim=-1)
        graph2 = F.normalize(graph2, dim=-1)
        logits = torch.einsum('bnd, bmd->bnm', [graph1, graph2])
        return logits

