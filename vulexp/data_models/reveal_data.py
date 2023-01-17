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


def do_oversampling(df: pd.DataFrame, seed) -> pd.DataFrame:
    # due to imbalance, oversampling to use
    max_size = df['gt'].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby('gt'):
        lst.append(group.sample(max_size - len(group), replace=True, random_state=seed))
    over_sampling_data = pd.concat(lst)
    return over_sampling_data


class Reveal(BaseDataModule, ABC):
    # feature_dim = 64
    n_class = 2

    def __init__(self, root, gtype, over_write=False, absolute_path=None, to_undirected=False,
                 seed: Optional[int] = None):
        super().__init__(root, gtype, over_write)
        self.absolute_path = absolute_path
        self.to_undirected = to_undirected
        self.seed = seed
        self.gtype = gtype

    def _loop_over_folder(self, raw_folder, parsed_folder, processed_folder, label=0, min_line_number_acceptable=2):
        """
        :param raw_folder:
        :param parsed_folder:
        :param processed_folder:
        :param label:
        :param min_line_number_acceptable: if a program has the number line of code is smaller this, reject it
        :return:
        """
        error_files = []
        error_reasons = []
        success_files = []
        target_files = []
        for (_, _, filenames) in walk(raw_folder):
            target_files.extend(filenames)
            break
        for i in tqdm(target_files):
            edges = osp.join(parsed_folder, i, 'edges.csv')
            nodes = osp.join(parsed_folder, i, 'nodes.csv')
            save_path = osp.join(processed_folder, self.gtype, f'{i}.gpickle')

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
                    BaseDataModule._serialize_n_count_graph(G, save_path)
                    success_files.append(osp.join(raw_folder, i))
                except Exception as e:
                    error_files.append(osp.join(raw_folder, i))
                    error_reasons.append(str(e))
            else:
                print('skip generate new file')
        return success_files, error_files, error_reasons

    def handle(self):
        raw_folder = osp.join(self.root, 'raw', 'raw_data')
        parsed_folder = osp.join(self.root, 'raw', 'parsed_reveal')
        non_raw_folder = osp.join(raw_folder, 'non')
        non_parsed_folder = osp.join(parsed_folder, 'non')
        vul_raw_folder = osp.join(raw_folder, 'vul')
        vul_parsed_folder = osp.join(parsed_folder, 'vul')

        processed_folder = osp.join(self.root, 'processed')

        if self.over_write:
            df_error_files = pd.DataFrame(columns=['path', 'reason', 'gt'])
            df_success_files = pd.DataFrame(columns=['path', 'gt'])

            a0, b0, c0 = self._loop_over_folder(non_raw_folder, non_parsed_folder, processed_folder, label=0)
            a1, b1, c1 = self._loop_over_folder(vul_raw_folder, vul_parsed_folder, processed_folder, label=1)

            df_error_files['path'] = b0 + b1
            df_error_files['reason'] = c0 + c1
            df_error_files['gt'] = [0] * len(b0) + [1] * len(b1)
            df_success_files['path'] = a0 + a1
            df_success_files['gt'] = [0] * len(a0) + [1] * len(a1)

            df_error_files.to_csv(osp.join(self.root, 'error_files.tsv'), sep='\t', index=False)
            df_success_files.to_csv(osp.join(self.root, 'success_files.tsv'), sep='\t', index=False)
            self.map_id_to_graph_file = df_success_files

        else:
            self.map_id_to_graph_file = pd.read_csv(osp.join(self.root, 'success_files.tsv'), sep='\t')
            # self.error_files = pd.read_csv(osp.join(self.root, 'error_files.tsv'), sep='\t')

    def generate_train_test(self):

        seed = self.seed

        if not osp.isfile(osp.join(self.root, 'split_sets.tsv')):

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
            over_sampling_data = do_oversampling(train_set, seed)

            assert len(pd.merge(train_set, test_set, on=['path', 'gt'])) == 0, 'data leakage!'
            assert len(pd.merge(train_set, val_set, on=['path', 'gt'])) == 0, 'data leakage!'

            tmp = self.map_id_to_graph_file.merge(test_set, how='left', on=['path', 'gt'], indicator=True)
            tmp['subset'] = ''
            tmp.loc[tmp._merge == 'both', 'subset'] = 'test'
            tmp = tmp[['path', 'gt', 'subset']]
            tmp = tmp.merge(train_set, how='left', on=['path', 'gt'], indicator=True)
            tmp.loc[tmp._merge == 'both', 'subset'] = 'train'
            tmp = tmp[['path', 'gt', 'subset']]
            tmp = tmp.merge(val_set, how='left', on=['path', 'gt'], indicator=True)
            tmp.loc[tmp._merge == 'both', 'subset'] = 'val'
            tmp = tmp[['path', 'gt', 'subset']]
            tmp.to_csv(osp.join(self.root, 'split_sets.tsv'), sep='\t', index=False)
        else:
            df = pd.read_csv(osp.join(self.root, 'split_sets.tsv'), sep='\t')
            train_set = df[df.subset == 'train']
            test_set = df[df.subset == 'test'].sort_values(by=['path', 'gt'], ascending=False)
            val_set = df[df.subset == 'val']
            over_sampling_data = do_oversampling(train_set, seed)

        return CustomSet(over_sampling_data, self.gtype, self.absolute_path, self.to_undirected), \
            CustomSet(val_set, self.gtype, self.absolute_path, self.to_undirected), \
            CustomSet(test_set, self.gtype, self.absolute_path, self.to_undirected)
