import os.path as osp
import random
import os
from abc import ABC
from os import walk
from typing import Optional

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm.auto import tqdm

from vulexp.data_models.abstract_dataset import BaseDataModule
from vulexp.data_models.interface import CustomSet
from vulexp.data_models.helpers import relabel_nodes
from vulexp.data_processing.parse_data_point import joern_to_networkx
from vulexp.data_processing.tokenize_node import symbolic_tokenize

magic_number = 12345
cwd = os.getcwd()
w2v_path = os.path.join(cwd, 'data/Word2Vec/li_et_al_wv')  # should be vulnerability-explanation/data/...


class Reveal(BaseDataModule, ABC):
    feature_dim = 64
    n_class = 2

    def __init__(self, root, over_write=False, absolute_path=None, to_undirected=False,
                 seed: Optional[int] = None):
        super().__init__(root, over_write)
        self.absolute_path = absolute_path
        self.to_undirected = to_undirected
        self.seed = seed

    def _loop_over_folder(self, raw_folder, parsed_folder, processed_folder, label=0, min_line_number_acceptable=2):
        """
        :param raw_folder:
        :param parsed_folder:
        :param processed_folder:
        :param label:
        :param min_line_number_acceptable: if a program has the number line of code is smaller this, reject it
        :return:
        """
        target_files = []
        for (_, _, filenames) in walk(raw_folder):
            target_files.extend(filenames)
            break
        for i in tqdm(target_files):
            edges = osp.join(parsed_folder, i, 'edges.csv')
            nodes = osp.join(parsed_folder, i, 'nodes.csv')
            save_path = osp.join(processed_folder, f'{i}.gpickle')

            if (self.over_write ^ (not osp.exists(save_path))) or (self.over_write and (not osp.exists(save_path))):
                # (A xor B) or (A and B)
                # if file not exist -> write
                # if file exist, over_write is true -> write
                try:
                    G = self._reader(nodes, edges, label=label)
                    max_line_number = -1
                    for inode in G.nodes(data=True):
                        max_line_number = max(max_line_number, inode[-1]['line_num'])
                    if max_line_number <= min_line_number_acceptable:
                        raise Exception('data has less then 2 line of code')
                    G = relabel_nodes(G)
                    self._serialize_n_count_graph(G, save_path)
                except Exception as e:
                    raise e
            else:
                print('skip generate new file')

    def handle(self):
        raw_folder = osp.join(self.root, 'raw', 'raw_data')
        parsed_folder = osp.join(self.root, 'raw', 'parsed_reveal')
        non_raw_folder = osp.join(raw_folder, 'non')
        non_parsed_folder = osp.join(parsed_folder, 'non')
        vul_raw_folder = osp.join(raw_folder, 'vul')
        vul_parsed_folder = osp.join(parsed_folder, 'vul')

        processed_folder = osp.join(self.root, 'processed')

        self._loop_over_folder(non_raw_folder, non_parsed_folder, processed_folder, label=0)
        self._loop_over_folder(vul_raw_folder, vul_parsed_folder, processed_folder, label=1)

        if self.over_write:
            self.map_id_to_graph_file = pd.DataFrame(self.map_id_to_graph_file, columns=['id', 'path', 'gt'])
            self.map_id_to_graph_file.to_csv(osp.join(self.root, 'map_id_to_graph.tsv'), sep='\t', index=False)
            self.error_files = pd.DataFrame(self.error_files, columns=['file_error', 'reason'])
            self.error_files.to_csv(osp.join(self.root, 'error_file.tsv'), sep='\t', index=False)

        else:
            self.map_id_to_graph_file = pd.read_csv(osp.join(self.root, 'map_id_to_graph.tsv'), sep='\t')
            self.error_files = pd.read_csv(osp.join(self.root, 'error_file.tsv'), sep='\t')

    def generate_train_test(self):
        seed = self.seed

        self.map_id_to_graph_file.index = range(0, len(self.map_id_to_graph_file.index))

        non_data = self.map_id_to_graph_file[self.map_id_to_graph_file['gt'] == 0]
        vul_data = self.map_id_to_graph_file[self.map_id_to_graph_file['gt'] == 1]

        # test
        ratio = len(vul_data) / len(self.map_id_to_graph_file)
        test_01 = vul_data.sample(int(ratio * len(vul_data)), random_state=seed)
        test_02 = non_data.sample(int(ratio * len(non_data)), random_state=seed)
        test_set = pd.concat([test_01, test_02])
        # remove test from database
        non_data = non_data.drop(test_set[test_set['gt'] == 0].index)
        vul_data = vul_data.drop(test_set[test_set['gt'] == 1].index)

        # val
        val_ratio = 0.1
        val_01 = vul_data.sample(int(val_ratio * len(vul_data)), random_state=seed)
        val_02 = non_data.sample(int(val_ratio * len(non_data)), random_state=seed)
        val_set = pd.concat([val_01, val_02])
        # remove val from database
        non_data = non_data.drop(val_set[val_set['gt'] == 0].index)
        vul_data = vul_data.drop(val_set[val_set['gt'] == 1].index)

        # train
        train_set = pd.concat([non_data, vul_data])
        # due to imbalance, oversampling to use
        max_size = train_set['gt'].value_counts().max()
        lst = [train_set]
        for class_index, group in train_set.groupby('gt'):
            lst.append(group.sample(max_size - len(group), replace=True, random_state=seed))
        over_sampling_data = pd.concat(lst)

        assert len(pd.merge(train_set, test_set, on=['path', 'gt'])) == 0, 'data leakage!'
        assert len(pd.merge(train_set, val_set, on=['path', 'gt'])) == 0, 'data leakage!'

        return CustomSet(over_sampling_data, self.absolute_path, self.to_undirected), \
               CustomSet(val_set, self.absolute_path, self.to_undirected), \
               CustomSet(test_set, self.absolute_path, self.to_undirected)
