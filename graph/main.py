import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset import load
from model import Model
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power, inv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def train(epoch=40, batch=64):
    nb_epochs = epoch
    batch_size = batch
    patience = 20
    lr = 3e-4
    l2_coef = 0.0
    hid_units = 512

    feat_train = torch.FloatTensor(feat).cuda()
    diff_train = torch.FloatTensor(diff).cuda()
    adj_train = torch.FloatTensor(adj).cuda()

    ft_size = feat[0].shape[1]

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    criter = nn.CrossEntropyLoss()

    model.cuda()

    cnt_wait = 0
    best = 1e9

    itr = (adj.shape[0] // batch_size) + 1
    mean_accs = []
    max_accs =[]
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        train_idx = np.arange(adj.shape[0])
        np.random.shuffle(train_idx)

        for idx in range(0, len(train_idx), batch_size):
            model.train()
            optimiser.zero_grad()

            batch = train_idx[idx: idx + batch_size]
            mask = num_nodes[idx: idx + batch_size]

            node_logits, graph_logits, div_loss = model(feat_train[batch], adj_train[batch], diff_train[batch])
            
            labels = torch.LongTensor([i for i in range(node_logits.size(1))]).cuda()
            graph_labels = torch.LongTensor([i for i in range(graph_logits.size(1))]).cuda()
    
            loss = criter(node_logits, labels) + criter(graph_logits, graph_labels) + div_loss
#             loss = criter(graph_logits, graph_labels) + div_loss
#             loss = criter(graph_logits, graph_labels)
            loss.backward()
            optimiser.step()

        epoch_loss /= itr
        if epoch % 10 == 0:
            mean_acc, max_acc = eval(model)
            mean_accs.append(mean_acc)
            max_accs.append(max_acc)
        if epoch_loss < best:
            best = epoch_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './Models/Ours-%s.pkl' % dataset)
#         else:
#             cnt_wait += 1

#         if cnt_wait == patience:
#             break

#     model.load_state_dict(torch.load(f'./Models/-{gpu}-d.pkl'))
    return mean_accs, max_accs
    

def eval(model):
    feat_eval = torch.FloatTensor(feat).cuda()
    diff_eval = torch.FloatTensor(diff).cuda()
    adj_eval = torch.FloatTensor(adj).cuda()
    labels_eval = torch.LongTensor(labels).cuda()

    embeds = model.embed(feat_eval, adj_eval, diff_eval)

    x = embeds.cpu().numpy()
    y = labels_eval.cpu().numpy()

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print(np.mean(accuracies), np.max(accuracies), np.std(accuracies))
    return np.mean(accuracies), np.max(accuracies)
   
# ds = ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COIL-DEL']
dataset = 'IMDB-BINARY'
adj, diff, feat, labels, num_nodes = load(dataset)
sub = False
if sub == True:
    size = 1000
    adj = adj[:size]
    feat = feat[:size]
    labels = labels[:size]
    num_nodes = num_nodes[:size]
# diff = []
# for i in tqdm(range(adj.shape[0])):
#     dif = compute_ppr(np.array(np.sign(adj[i])))
#     diff.append(dif)
# diff = np.array(diff)

if __name__ == '__main__':
    gpu = 0
    torch.cuda.set_device(gpu)
    batch = 256
    epoch = 200
    seed = np.random.randint(0, 1000)
#     seed = 823

    print(f'Dataset: {dataset}, Epoch: {epoch}, Seed: {seed}')
    mean_accs, max_accs = train(epoch, batch)
    plt.plot([i for i in range(len(mean_accs))], mean_accs)
    plt.plot([i for i in range(len(max_accs))], max_accs)
    plt.savefig("./Images/curve.jpg")