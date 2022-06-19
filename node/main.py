import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor, compute_adjacency, generate_pos, compute_norm
from dataset import load
from model import *
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power, inv

hid_units = 256
nb_epochs = 2000
patience = 20
lr = 3e-4
l2_coef = 0.0
sparse = False
# sample_size = 19717
sample_size = 10000
batch_size = 4
projection = False

def eval(model):
    xent = nn.CrossEntropyLoss()
    
    nb_classes = np.unique(labels).shape[0]
    features_eval = torch.FloatTensor(features[np.newaxis])
    adj_eval = torch.FloatTensor(adj[np.newaxis])
    diff_eval = torch.FloatTensor(diff[np.newaxis])
    features_eval = features_eval.cuda()
    adj_eval = adj_eval.cuda()
    diff_eval = diff_eval.cuda()
    labels_eval = torch.LongTensor(labels).cuda()
    idx_train_eval = torch.LongTensor(idx_train)
    idx_test_eval = torch.LongTensor(idx_test)

    embeds, _ = model.embed(features_eval, adj_eval, diff_eval, False, None)
    train_embs = embeds[0, idx_train_eval]
    test_embs = embeds[0, idx_test_eval]
    
    train_lbls = labels_eval[idx_train_eval]
    test_lbls = labels_eval[idx_test_eval]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        acc = 0
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            if _ % 10 == 0:
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
            
            loss.backward()
            opt.step()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print("Test: ", accs.min().item(), accs.mean().item(), accs.max().item())
    return accs.max().item(), accs.mean().item(), accs.min().item()

def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def disc_loss(logits):
    b_xent = nn.CrossEntropyLoss()
    width = logits.size(1)
    logits1 = logits.view(batch_size * width, width)
    logits2 = logits.transpose(2, 1).contiguous().view(batch_size * width, width)
    new_label = torch.LongTensor([i for i in range(width)] * batch_size).cuda() 
    loss = b_xent(logits1, new_label) / 2 + b_xent(logits2, new_label) / 2
    return loss

def attn_loss(embed1, embed2, adj):
    zero_vec = -9e15*torch.ones_like(adj)
    attention = torch.einsum('bmd, bnd->bmn', (embed1, embed2))
    attention = torch.where(adj > 0, attention, zero_vec)
    attention = F.softmax(attention, dim=-1)
    embed1_view = torch.einsum('bmn, bnd->bmd', (attention, embed2))
    embed2_view = torch.einsum('bnm, bmd->bnd', (attention.transpose(1,2), embed1))
#     embed1 = F.normalize(embed1, dim=-1)
    embed1_view = F.normalize(embed1_view, dim=-1)
#     embed2 = F.normalize(embed2, dim=-1)
    embed2_view = F.normalize(embed2_view, dim=-1)
    
#     loss1 = disc_loss(torch.einsum('bmd, bnd->bmn', (embed1_view, embed1)))
#     loss2 = disc_loss(torch.einsum('bmd, bnd->bmn', (embed2_view, embed2)))
    loss3 = disc_loss(torch.einsum('bmd, bnd->bmn', (embed1, embed2)))
    return loss3
#     return (((embed1-embed1_view.detach())**2).sum(dim=-1).mean()+((embed2-embed2_view.detach())**2).sum(dim=-1).mean())/2
#     return ((embed1-embed2)**2).sum(dim=-1).mean()

def train(verbose=True):
    
    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels_train = torch.LongTensor(labels)
    idx_train_train = torch.LongTensor(idx_train)
    idx_test_train = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units, projection)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels_train = labels_train.cuda()
        lbl = lbl.cuda()
        idx_train_train = idx_train_train.cuda()
        idx_test_train = idx_test_train.cuda()

    b_xent = nn.CrossEntropyLoss()
    
    cnt_wait = 0
    best = 1e9
    best_t = 0
    max_scores = []
    mean_scores = []
    min_scores = []
    for epoch in range(nb_epochs):
        if epoch % 10 == 0:
            max_score, mean_score, min_score = eval(model)
            max_scores.append(max_score)
            mean_scores.append(mean_score)
            min_scores.append(min_score)
#             with open("./curve-%d-%d.txt" % (sample_size, hid_units), 'a+') as f:
#                 f.write("%.5f, %.5f, %.5f\n" % (min_score, mean_score, max_score))
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]
        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, h1, h2, div_loss = model(bf, shuf_fts, ba, bd, sparse, None, None, None)
#         loss = attn_loss(h1, h2, ba)
        
        width = logits.size(1)
        logits1 = logits.view(batch_size * width, width)
        logits2 = logits.transpose(2, 1).contiguous().view(batch_size * width, width)
        new_label = torch.LongTensor([i for i in range(width)] * batch_size).cuda()
        
        loss = b_xent(logits1, new_label) / 2 + b_xent(logits2, new_label) / 2 - div_loss
#         loss = b_xent(logits1, new_label) / 2 + b_xent(logits2, new_label) / 2

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}, Div loss: {2:0.4f}'.format(epoch, loss.item(), div_loss))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './Models/model-PubMed-Ours.pkl')
        else:
            cnt_wait += 1

#         if cnt_wait == patience:
#             if verbose:
#                 print('Early stopping!')
#             break
        
    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('./Models/model-PubMed-Ours.pkl'))
    eval(model)
    return max_scores, mean_scores, min_scores
    
# 'cora', 'citeseer', 'pubmed'
dataset = 'pubmed'
adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)
# diff = compute_ppr(np.array(np.sign(adj)))

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    max_score = 0

    max_scores, mean_scores, min_scores = train()
    plt.plot([i for i in range(1, len(max_scores)+1)], max_scores)
    plt.plot([i for i in range(1, len(mean_scores)+1)], mean_scores)
    plt.savefig("./Images/acc_track.jpg")
    