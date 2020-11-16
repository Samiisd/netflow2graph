
"""
I am not proud of what is going on within that file, so please read it only if
that's mandatory for your understanding of the notebook.

There is a lot of code duplicated, which could have been refactored, but I've
been lazy.
"""






















































































"""
NOOOOOO, you really don't want to read that code, don't continue further...
"""





























































"""
You're going to regret this :/
"""



























































"""
If you are here, then nothing can stop you anymore...
May the Force be with you!
"""

import torch
import numpy as np
from functools import partial
from tqdm.notebook import tqdm
tqdm = partial(tqdm, dynamic_ncols=True)


def evaluateSimple(model, forward, dt, criterion):
    losses = []
    corr, tot = 0, 0
    with torch.no_grad():
        model.eval()
        for g in tqdm(dt, desc='eval', leave=False):
            Y_true = g.ndata['y']

            Y_pred = forward(model, g) 

            loss = criterion(Y_pred, Y_true)
            losses.append(loss.item())
            Y_pred = Y_pred.max(dim=-1)[1]

            tot += len(Y_pred)
            corr += (Y_true == Y_pred).sum().item()

    return np.mean(losses), corr/tot

def trainSimple(model, optimizer, criterion, forward, dt_train, dt_val, dt_test, nb_epochs=100, freq_show_loss=10):
    loss_val, acc_val = evaluateSimple(model, forward, dt_val, criterion)
    loss_test, acc_test = evaluateSimple(model, forward, dt_test, criterion)
    print('Before:')
    print(f'\tloss_val: {loss_val:.3f} / acc_val: {acc_val:.3f}')
    print(f'\tloss_test: {loss_test:.3f} / acc_tes: {acc_test:.3f}')

    for epoch in tqdm(range(nb_epochs), desc='epoch'):
        losses = []
        corr, tot = 0, 0
        
        model.train()
        pb_training = tqdm(dt_train, desc='train', leave=False)
        for idx, g in enumerate(pb_training):
            Y_true = g.ndata['y']
            Y_pred = forward(model, g)
            
            loss = criterion(Y_pred, Y_true)
            losses.append(loss.item())
            
            Y_pred = Y_pred.max(dim=-1)[1]
            tot += len(Y_pred)
            corr += (Y_pred == Y_true).sum().item()
            
            if idx % freq_show_loss == 0:
                pb_training.set_description(f"l:{np.mean(losses):.3f}, acc:{corr/tot:.3f}")
            
            # weights optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_val, acc_val = evaluateSimple(model, forward, dt_val, criterion)
        print(f"epoch {epoch:03d}: loss_val:{loss_val:.3f}, acc: {acc_val:.3f}")
              
    loss_val, acc_val = evaluateSimple(model, forward, dt_val, criterion)
    loss_test, acc_test = evaluateSimple(model, forward, dt_test, criterion)
    print('After:')
    print(f'\tloss_val: {loss_val:.3f} / acc_val: {acc_val:.3f}')
    print(f'\tloss_test: {loss_test:.3f} / acc_tes: {acc_test:.3f}')




from pathlib import Path
from dgl.data import DGLDataset
import flow2graph
from dgl.data.utils import save_graphs, load_graphs

class CTU13Dataset(DGLDataset):
    from functools import cached_property
    
    label_normal = flow2graph.Label.normal.value
    label_background = flow2graph.Label.background.value
    label_malicious = flow2graph.Label.malicious.value
    
    def __init__(self, raw_dir=None, save_dir=None, force_reload=False, verbose=False):
        self._save_dir = Path(save_dir) # (1) temporary fix waiting for: https://github.com/dmlc/dgl/pull/2262
        self._raw_dir = Path(raw_dir)
        
        self._p_raws = sorted(list(self._raw_dir.glob('[0-9]*.pkl')))
        self._p_unraws = list(map(lambda p: self._save_dir/f"{p.name.rstrip('pkl')}nx", self._p_raws))
        
        super().__init__(name='CTU13',
                         raw_dir=self._raw_dir, 
                         save_dir=123, # (1)
                         force_reload=force_reload, 
                         verbose=verbose)
    def process(self):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(self._p_raws)), desc='processing', disable=self.verbose):
            rp, wp = self._p_raws[i], self._p_unraws[i]
            
            g = nx.read_gpickle(rp)
            g = dgl.from_networkx(g, node_attrs=['label'], edge_attrs=['features', 'label'])
            
            label_edge, label_node = g.edata.pop('label'), g.ndata.pop('label')
            save_graphs(str(wp), g, labels={
                'edge': label_edge,
                'node': label_node,
            })
            
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        g, labels = load_graphs(str(self._p_unraws[idx]))
        return g[0], labels

    def __len__(self):
        return len(self._p_unraws)

    def has_cache(self):
        return len(self) == len(list(self._save_dir.glob('[0-9]*.nx')))

def evaluateCTU13(model, dt, criterion):
    losses = []
    acc_e, acc_n = 0, 0
    nb_e, nb_n = 0, 0
    with torch.no_grad():
        model.eval()
        for (g, labels) in tqdm(dt, desc='eval', leave=False):
            label_node = (labels['node'] == CTU13Dataset.label_malicious).long()
            label_edge = (labels['edge'] == CTU13Dataset.label_malicious).long()

            ft_edges = g.edata['features']

            pred_node, pred_edge = tuple(map(lambda y: y.squeeze(), model(g, ft_edges)))
            loss = criterion(pred_node, pred_edge, label_node, label_edge)
            
            # that's not really the accuracy, but True Positives (malicious)
            acc_e += (pred_edge.max(dim=-1)[1])[label_edge.bool()].sum().item()
            acc_n += (pred_node.max(dim=-1)[1])[label_node.bool()].sum().item()
            nb_e += label_edge.sum().item()
            nb_n += label_node.sum().item()
            
            losses.append(loss.item()) 
    return np.mean(losses), acc_n/nb_e, acc_e/nb_e
    
def trainCTU13(model, optimizer, criterion, dt_train, dt_val, dt_test, nb_epochs=10, freq_show_loss=10):
    loss_val, acc_val_n, acc_val_e = evaluateCTU13(model, dt_val, criterion)
    loss_test, acc_test_n, acc_test_e = evaluateCTU13(model, dt_test, criterion)
    print("Before train:")
    print(f"\t Validation: loss:{loss_val:.3f} | acc_n:{acc_val_n:.3f} | acc_e: {acc_val_e:.3f}")
    print(f"\t Test      : loss:{loss_test:.3f} | acc_n:{acc_test_n:.3f} | acc_e: {acc_test_e:.3f}")       
    
    for epoch in tqdm(range(nb_epochs), desc='epoch'):
        losses = []
        
        acc_train_e, acc_train_n = 0, 0
        nb_train_e, nb_train_n = 0, 0
        
        model.train()
        
        # training loop
        pb_training = tqdm(dt_train, desc='train', leave=False)
        for idx, (g, labels) in enumerate(pb_training):
            label_n = (labels['node'] == CTU13Dataset.label_malicious).long()
            label_e = (labels['edge'] == CTU13Dataset.label_malicious).long()
            
            ft_e = g.edata['features']

            # criterion
            pred_n, pred_e = model(g, ft_e)
            pred_n, pred_e = pred_n.squeeze(), pred_e.squeeze()
            loss = criterion(y_pred_n=pred_n, y_pred_e=pred_e, y_true_n=label_n, y_true_e=label_e)
            
            # loss monitoring
            acc_train_e += (pred_e.max(dim=-1)[-1])[label_e.bool()].sum().item()
            acc_train_n += (pred_n.max(dim=-1)[-1])[label_n.bool()].sum().item()
            nb_train_e += label_e.sum().item()
            nb_train_n += label_n.sum().item()
            losses.append(loss.item())
            if idx % freq_show_loss == 0:
                pb_training.set_description(f"loss: {np.mean(losses):.2f}, \
                acc_e:{acc_train_e/nb_train_e:.2f}, \
                acc_n:{acc_train_n/nb_train_n:.2f}")
            
            # weights optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train, acc_train_n, acc_train_e = np.mean(losses), acc_train_n/nb_train_n, acc_train_e/nb_train_e
        loss_val, acc_val_n, acc_val_e = evaluateCTU13(model, dt_val, criterion)
        print(f"epoch {epoch}: loss_train={loss_train:.3f} loss_val={loss_val:.3f} | "
              f"acc_train_n={acc_train_n:.3f} acc_val_n={acc_val_n:.3f} | "
              f"acc_train_e={acc_train_e:.3f} acc_val_e={acc_val_e:.3f}") 

    loss_val, acc_val_n, acc_val_e = evaluateCTU13(model, dt_val, criterion)
    loss_test, acc_test_n, acc_test_e = evaluateCTU13(model, dt_test, criterion)
    print("After train:")
    print(f"\t Validation: loss:{loss_val:.3f} | acc_n:{acc_val_n:.3f} | acc_e: {acc_val_e:.3f}")
    print(f"\t Test      : loss:{loss_test:.3f} | acc_n:{acc_test_n:.3f} | acc_e: {acc_test_e:.3f}")

from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plot_heatmap(y_true, y_pred):
    data = {
        'y_true': y_true,
        'y_pred': y_pred
    }

    df = pd.DataFrame(data, columns=['y_true','y_pred'])
    confusion_matrix = pd.crosstab(df['y_true'], df['y_pred'], rownames=['True'], colnames=['Prediction'])
    confusion_matrix /= confusion_matrix.sum()

    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

def report(y_true_e, y_pred_e, y_true_n, y_pred_n, show_on_nodes=True):
    print('Eval Edges Classification:')
    print(classification_report(y_true_e, y_pred_e, target_names=['normal', 'malicious']))
    plot_heatmap(y_true_e, y_pred_e)
    if show_on_nodes:
        print('Eval Nodes Classification:')
        print(classification_report(y_true_n, y_pred_n, target_names=['normal', 'malicious']))
        plot_heatmap(y_true_n, y_pred_n)

def classification_report_graph(model, g, labels):
    y_true_e = labels['edge'] == CTU13Dataset.label_malicious
    y_true_n = labels['node'] == CTU13Dataset.label_malicious
    with torch.no_grad():
        y_pred_n, y_pred_e = model(g, g.edata['features'])
        y_pred_n = y_pred_n.max(dim=-1)[1].bool()
        y_pred_e = y_pred_e.max(dim=-1)[1].bool()
    report(y_true_n=y_true_n, y_pred_n=y_pred_n, y_true_e=y_true_e, y_pred_e=y_pred_e)

def classification_report_dataset(model, dt, show_on_nodes=True):
    model.eval()
    tot_y_true_n, tot_y_true_e, tot_y_pred_n, tot_y_pred_e = [],[],[],[]
    for idx in tqdm(range(len(dt)), leave=False):
        g, labels = dt[idx]
        y_true_e = labels['edge'] == CTU13Dataset.label_malicious
        y_true_n = labels['node'] == CTU13Dataset.label_malicious
        with torch.no_grad():
            y_pred_n, y_pred_e = model(g, g.edata['features'])
            if show_on_nodes:
                y_pred_n = y_pred_n.max(dim=-1)[1].bool()
            y_pred_e = y_pred_e.max(dim=-1)[1].bool()
        if show_on_nodes:
            tot_y_true_n.append(y_true_n)
            tot_y_pred_n.append(y_pred_n)
        tot_y_true_e.append(y_true_e)
        tot_y_pred_e.append(y_pred_e)

    if show_on_nodes:
        tot_y_true_n = np.concatenate(tot_y_true_n)
        tot_y_pred_n = np.concatenate(tot_y_pred_n)
    tot_y_true_e = np.concatenate(tot_y_true_e)
    tot_y_pred_e = np.concatenate(tot_y_pred_e)
    report(y_true_n=tot_y_true_n, y_pred_n=tot_y_pred_n, y_true_e=tot_y_true_e, y_pred_e=tot_y_pred_e, show_on_nodes=show_on_nodes)
