from abc import ABC, abstractmethod
import os
import nltk
from typing import Tuple
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from vulexp.data_models.interface import CustomSet
from vulexp.data_processing.tokenize_node import symbolic_tokenize
from vulexp.data_processing.parse_data_point import joern_to_networkx

from pytorch_lightning import LightningDataModule

cwd = os.getcwd()
w2v_path = os.path.join(cwd, 'data/Word2Vec/li_et_al_wv')


class BaseDataModule(LightningDataModule):

    def __init__(self, root, over_write=False):
        print(f'over_write: {over_write}')
        self.root = root
        self.over_write = over_write
        self.word2vec = Word2Vec.load(w2v_path)
        self.map_id_to_graph_file = []
        self.error_files = []
        self.over_sampling_data = None
        self.non_data = []
        self.vul_data = []
        self.count = 0
        self.feat_dim = 64

        # read
        self.handle()

    def _reader(self, nodes, edges, label):
        """
        given 1 data point after joern parsing, create a graph networkx object
        :param nodes:
        :param edges:
        :param label:
        :return:
        """
        g = joern_to_networkx(nodes, edges)
        G = g[0]['graph']
        G.graph['label'] = label

        # now is to vectorize node feature
        sym_map = {}
        for ci in G.nodes():
            cnode = G.nodes[ci]
            source = cnode['code']

            symbolic_tokens = []
            original_tokens = []

            actual_line_tokens = nltk.wordpunct_tokenize(source)
            original_tokens.append(actual_line_tokens)
            symbolic_line_tokens, sym_map = symbolic_tokenize(source, sym_map)
            symbolic_tokens.append(symbolic_line_tokens)
            nrp = np.zeros(64, dtype=np.float32)
            for token in symbolic_line_tokens:
                try:
                    embedding = self.word2vec.wv[token]
                except ValueError:
                    embedding = np.zeros(self.feat_dim)
                nrp = np.add(nrp, embedding)
                # line_feat.append(embedding.tolist())
            if len(actual_line_tokens) > 0:
                fNrp = np.divide(nrp, len(symbolic_line_tokens))
            else:
                fNrp = nrp
            cnode['feat'] = fNrp
            cnode['symbolic'] = symbolic_line_tokens
        return G

    def _serialize_n_count_graph(self, graph, save_path):
        try:
            nx.write_gpickle(graph, save_path)
            self.count += 1
            self.map_id_to_graph_file.append((self.count, save_path, graph.graph['label']))
        except Exception as e:
            print(e)

    def len(self) -> int:
        return len(self.map_id_to_graph_file)

    @abstractmethod
    def generate_train_test(self) -> Tuple[CustomSet, CustomSet, CustomSet]:
        raise NotImplementedError

    @abstractmethod
    def handle(self):
        raise NotImplementedError

