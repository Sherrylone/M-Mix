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
    def forward(self, seq, adj, attn=False, training=True):
        seq_fts = self.fc(seq)
        if attn:
            out = torch.bmm(adj, seq_fts)
            out_q = F.normalize(self.to_q(out), dim=-1)
            out_k = F.normalize(self.to_k(out), dim=-1)
            attn_mx = torch.einsum('bmd, bnd->bmn', (out_q, out_k)).detach()
            div_loss = ((out_q - out_k)**2).sum(dim=-1).mean()
            
            zero_vec = -9e15*torch.ones_like(attn_mx)
            attention = torch.where(adj > 0, attn_mx, zero_vec)
            attention = F.softmax(attention, dim=-1)
            attention = F.dropout(attention, 0.1, training=training)
            out = self.to_v(torch.bmm(attention, out)) + self.bias
            return self.act(out), div_loss
        else:
            out = torch.bmm(adj, seq_fts) + self.bias
            return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()
        self.new_dis = DIS(n_h)

    def forward(self, seq, adj, diff):
        h_1, div_loss = self.gcn1(seq, adj, attn=True,training=True)
        h_2 = self.gcn2(seq, diff, attn=False)
        c_1 = self.read(h_1)
        c_2 = self.read(h_2)
        ret, ret2 = self.new_dis(h_1, h_2,c_1, c_2)
        return ret, ret2, div_loss

    def embed(self, seq, adj, diff, msk=None):
        h_1, div_loss = self.gcn1(seq, adj, attn=True, training=False)
        h_2 = self.gcn2(seq, diff, attn=False)
        
        c_1 = self.read(h_1, msk)
        c_2 = self.read(h_2, msk)
        return (c_1 + c_2).detach()
#         return c_1.detach()

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
    def __init__(self, out_ft):
        super(DIS, self).__init__()
#         self.fc = nn.Linear(out_ft * 2, 1, bias=False)
    def forward(self, node1, node2, graph1, graph2):
        """
        n1 || n1
        n2 || n1
        n3 || n1
        ...
        n1 || n3
        n2 || n3
        n3 || n3
        """
#         N = graph1.size(1)
#         graph1 = graph1.repeat_interleave(N, dim=1)
#         graph2 = graph2.repeat(1, N, 1)
#         logits = self.fc(torch.cat([graph1, graph2], dim=-1))
        
        # B * N * N
        B, N, D = node1.size()
        node1 = F.normalize(node1, dim=-1).view(B * N, D)
        node2 = F.normalize(node2, dim=-1).view(B * N, D)
        node_logits = torch.einsum('nd, md->nm', [node1, node2])
        
        graph1 = F.normalize(graph1, dim=-1)
        graph2 = F.normalize(graph2, dim=-1)
        graph_logits = torch.einsum('nd, md->nm', [graph1, graph2])

        return node_logits, graph_logits

