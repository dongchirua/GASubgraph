import glob
import os
import os.path as osp

import networkx as nx

import pandas as pd
from tqdm import tqdm

from vulexp.data_models.abstract_dataset import BaseDataModule
from vulexp.data_models.helpers import relabel_nodes
from vulexp.data_models.interface import CustomSet

magic_number = 12345


class NVD(BaseDataModule):

    def __init__(self, root, over_write=False):
        super().__init__(root, over_write)

    def _loop_over_folder(self, target_files, parsed_folder, processed_folder, label, min_line_number_acceptable=2):
        """
        :param target_files:
        :param parsed_folder:
        :param processed_folder:
        :param label:
        :param min_line_number_acceptable: if a program has the number line of code is smaller this, reject it
        :return:
        """
        for i in tqdm(target_files):
            file_name = i.split('/')[-1]
            edges = osp.join(parsed_folder, file_name, 'edges.csv')
            nodes = osp.join(parsed_folder, file_name, 'nodes.csv')
            save_path = osp.join(processed_folder, f'{file_name}.gpickle')

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
                    _serialize_n_count_graph(G, save_path)
                except Exception as e:
                    self.error_files.append((file_name, str(e)))

    def handle(self):
        raw_folder = osp.join(self.root, 'raw', 'raw_data')
        parsed_folder = osp.join(self.root, 'raw', 'parsed_nvd')
        processed_folder = osp.join(self.root, 'processed')
        if not osp.exists(processed_folder):
            os.makedirs(processed_folder)

        non_files = glob.glob(f'{raw_folder}/CVE_*_PATCHED_*.c')
        vul_files = glob.glob(f'{raw_folder}/CVE_*_VULN_*.c')

        self._loop_over_folder(non_files, parsed_folder, processed_folder, label=0)
        self._loop_over_folder(vul_files, parsed_folder, processed_folder, label=1)

        if self.over_write:
            self.map_id_to_graph_file = pd.DataFrame(self.map_id_to_graph_file, columns=['id', 'path', 'gt'])
            self.map_id_to_graph_file.to_csv(osp.join(self.root, 'map_id_to_graph.tsv'), sep='\t', index=False)
            self.error_files = pd.DataFrame(self.error_files, columns=['file_error', 'reason'])
            self.error_files.to_csv(osp.join(self.root, 'error_file.tsv'), sep='\t', index=False)

        else:
            self.map_id_to_graph_file = pd.read_csv(osp.join(self.root, 'map_id_to_graph.tsv'), sep='\t')
            self.error_files = pd.read_csv(osp.join(self.root, 'error_file.tsv'), sep='\t')

    def generate_train_test(self):
        """
        NDV data set currently doesn't need
        """
        pass

    def get_pair_with_name(self, name):
        records = self.map_id_to_graph_file
        records['name'] = records.apply(lambda r: r['path'].split('/')[-1], axis=1)
        if 'PATCHED' in name:
            vul_name = name.replace('PATCHED', 'VULN')
            non_name = name
        else:
            non_name = name.replace('VULN', 'PATCHED')
            vul_name = name
        vul_file = records[records['name'] == f'{vul_name}.gpickle']['path'].to_list()[0]
        non_file = records[records['name'] == f'{non_name}.gpickle']['path'].to_list()[0]
        vul_idx = records[records['name'] == f'{vul_name}.gpickle'].index.to_list()[0]
        non_idx = records[records['name'] == f'{non_name}.gpickle'].index.to_list()[0]
        vul_g = nx.read_gpickle(vul_file)
        non_g = nx.read_gpickle(non_file)
        return vul_g, vul_idx, non_g, non_idx

    def generate_dataset(self):
        return CustomSet(self.map_id_to_graph_file)


if __name__ == "__main__":
    data_module = NVD(root='data/NVD', over_write=False)
    vul, vul_id, non, non_id = data_module.get_pair_with_name("CVE_2010_4342_PATCHED_aun_incoming.c")
    myset = data_module.generate_dataset()
    myset.get(vul_id)

