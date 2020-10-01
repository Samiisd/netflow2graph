#!/usr/bin/env python3

import sys
from typing import Set, Optional, Dict
from pathlib import Path
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm


class Label(Enum):
    background = 0
    normal = 1
    malicious = 2


class NetflowDataset:
    _pd_dtype = {
        'ip_src': str,
        'ip_dst': str,
        'port_src': np.uint16,
        'port_dst': np.uint16,
        'protocol': str,
        'length': np.uint16,
    }

    def __init__(self, csvfile: Path, window_time_sec: int = 60, chunksize: int = int(1e6),
            ip_normal: Optional[Set] = None, ip_malicious: Optional[Set] = None):
        self._csvfile = csvfile
        self._window_time_sec = window_time_sec
        self._chunksize = chunksize
        self._ip_normal = ip_normal or set()
        self._ip_malicious = ip_malicious or set()

    def _annotate(self, df):
        df['label'] = Label.background.value
        if self._ip_malicious:
            df.loc[(df.ip_src.isin(self._ip_malicious) | df.ip_dst.isin(self._ip_normal)), 'label'] = Label.malicious.value
        if self._ip_normal:
            df.loc[(df.ip_src.isin(self._ip_normal) | df.ip_dst.isin(self._ip_normal)), 'label'] = Label.normal.value
        df.label = df.label.astype(np.uint8)
        return df

    def _feature_window(self, df, normalize=True):
        df = df.groupby(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocol', 'label'])\
               .agg({
                   'time': ['max', 'min'],
                   'length': ['mean', 'median', 'count', 'sum']
                })\
                .reset_index()

        # get flow duration
        duration = df.time['max'] - df.time['min']
        df = df.drop('time', axis=1, level=0)

        # normalize by time window
        if normalize:
            duration /= self._window_time_sec
            df.length /= self._window_time_sec

        # create features list
        df['features'] = np.concatenate([df.length.values, np.expand_dims(duration, axis=-1)], axis=-1)\
                           .astype(np.float32)\
                           .tolist()

        df = df.drop('length', axis=1, level=0)

        return df

    @staticmethod
    def _pd2graph(df):
        g: nx.Graph = nx.from_pandas_edgelist(
                df,
                source='ip_src',
                target='ip_dst',
                edge_attr=['features', 'protocol', 'label'],
                create_using=nx.DiGraph,
                # FIXME: should consider using `edge_key` to have multigraph for the different protocols (TCP/UDP/ICMP)
        )

        nx.set_node_attributes(g, Label.background.value, 'label')
        nx.set_node_attributes(g, {k: Label.normal.value for k in df[df.label == Label.normal.value].ip_src}, 'label')
        nx.set_node_attributes(g, {k: Label.malicious.value for k in df[df.label == Label.malicious.value].ip_src}, 'label')

        return g

    def _iter_csv(self, annotate=True):
        df = pd.read_csv(self._csvfile,
                         compression='gzip',
                         dtype=NetflowDataset._pd_dtype, chunksize=self._chunksize)
        for chunk in df:
            if annotate:
                chunk = self._annotate(chunk)
            yield chunk

    @staticmethod
    def interval_range(a, b, freq):
        it = a + freq
        while a < b:
            yield a, min(it, b)
            a, it = a + freq, it + freq

    def __iter__(self):
        for df in self._iter_csv(annotate=True):
            # FIXME: URGENT: should take into account the overlap with previous chunk
            for (left, right) in self.interval_range(df.time.iloc[0], df.time.iloc[-1], freq=self._window_time_sec):
                local_n_labels = df.label.value_counts()
                n_labels = np.zeros(len(Label))
                n_labels[local_n_labels.index] += local_n_labels.values

                df_window = df[df.time.between(left, right)]
                df_window = self._feature_window(df_window)

                yield self._pd2graph(df_window), n_labels.astype(np.uint64)

    def get_features(self) -> Optional[Dict]:
        n_iterations = 0
        n_labels = np.zeros(len(Label))
        flow_quantity = None
        unique_ip = set()

        for df in tqdm(self._iter_csv(annotate=True)):
            local_n_labels = df.label.value_counts()
            n_labels[local_n_labels.index] += local_n_labels.values
            unique_ip |= set(pd.unique(df[['ip_src', 'ip_dst']].values.ravel('K')))

            local_flow_quantity = df.groupby('label').agg({'length': ['sum', 'mean', 'std']})
            flow_quantity = local_flow_quantity if flow_quantity is None else flow_quantity.add(local_flow_quantity,
                                                                                                fill_value=0)

            n_iterations += 1

        if n_iterations == 0:
            return None

        flow_quantity.length['mean'] /= n_iterations
        flow_quantity.length['std'] /= n_iterations

        nf_background, nf_normal, nf_malicious = n_labels[Label.background.value], n_labels[Label.normal.value], n_labels[Label.malicious.value]

        return {
            'malicious': {
                'n_nodes': len(unique_ip & self._ip_malicious),
                'n_flows': int(nf_malicious),
                'flows:': {
                    'sum': flow_quantity.loc[Label.malicious.value, ('length', 'sum')] if nf_malicious else 0,
                    'mean': flow_quantity.loc[Label.malicious.value, ('length', 'mean')] if nf_malicious else 0,
                    'mean_std': flow_quantity.loc[Label.malicious.value, ('length', 'std')] if nf_malicious else 0,
                },
            },
            'normal': {
                'n_nodes': len(unique_ip & self._ip_normal),
                'n_flows': int(nf_normal),
                'flows:': {
                    'sum': flow_quantity.loc[Label.normal.value, ('length', 'sum')] if nf_normal else 0,
                    'mean': flow_quantity.loc[Label.normal.value, ('length', 'mean')] if nf_normal else 0,
                    'mean_std': flow_quantity.loc[Label.normal.value, ('length', 'std')] if nf_normal else 0,
                },
            },
            'background': {
                'n_nodes': len(unique_ip - self._ip_normal - self._ip_malicious),
                'n_flows': int(nf_background),
                'flows:': {
                    'sum': flow_quantity.loc[Label.background.value, ('length', 'sum')] if nf_background else 0,
                    'mean': flow_quantity.loc[Label.background.value, ('length', 'mean')] if nf_background else 0,
                    'mean_std': flow_quantity.loc[Label.background.value, ('length', 'std')] if nf_background else 0,
                },
            },
        }


def plot_flow_graph(g, malicious=None, normal=None):
    def edge_filter(label: Label):
        return list(filter(lambda e: g[e[0]][e[1]]['label'] == label.value, g.edges))

    malicious, normal = malicious or set(), normal or set()

    for (s, t) in g.edges:
        e = g[s][t]
        e['weight'] = e['features'][2]

    pos = nx.spring_layout(g)
    valid_ip = set(pos.keys())

    options_big_nodes = {"node_size": 200, "alpha": 1.0}
    nx.draw_networkx_nodes(g, pos, node_color='black', alpha=0.3, node_size=options_big_nodes['node_size']//10, label='background')
    nx.draw_networkx_nodes(g, pos, nodelist=valid_ip & malicious, node_color="r", **options_big_nodes, label='malicious')
    nx.draw_networkx_nodes(g, pos, nodelist=valid_ip & normal, node_color="b", **options_big_nodes, label='normal')

    nx.draw_networkx_edges(g, pos, width=0.5, style='dashed')
    nx.draw_networkx_edges(g, pos, edgelist=edge_filter(Label.malicious), width=2, alpha=0.8, edge_color='r')
    nx.draw_networkx_edges(g, pos, edgelist=edge_filter(Label.normal), width=2, alpha=0.8, edge_color='b')

    plt.axis('off')
    return plt.show()


def main():
    dt = NetflowDataset(
            Path(sys.argv[1]),
            chunksize=int(1e5),
            ip_malicious={'147.32.84.165', '147.32.84.191', '147.32.84.192'},
            ip_normal={'147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11'}
        )
    print("Time window size:", dt._window_time_sec, "seconds")
    for (i, (g, n_labels)) in enumerate(dt):
        print("Graph", i)
        print(nx.classes.function.info(g))
        print('n_flow_background =', n_labels[Label.background.value])
        print('n_flow_normal     =', n_labels[Label.normal.value])
        print('n_flow_malicious  =', n_labels[Label.malicious.value])
        if n_labels[Label.malicious.value] > 50:
            print('ploting graph')
            plot_flow_graph(g, dt._ip_malicious, dt._ip_normal)
            break
    return 0


if __name__ == '__main__':
    sys.exit(main())