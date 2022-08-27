from vulexp.data_models.interface import CustomSet
import os.path as osp
import os
from typing import Optional
import pandas as pd
from vulexp.data_models.reveal_data import Reveal


magic_number = 12345
cwd = os.getcwd()
w2v_path = os.path.join(cwd, 'data/Word2Vec/li_et_al_wv')  # should be vulnerability-explanation/data/...


class Adhoc(Reveal):

    def __init__(self, root, over_write=False, absolute_path=None, to_undirected=False,
                 seed: Optional[int] = None):
        super().__init__(root, over_write)
        self.absolute_path = absolute_path
        self.to_undirected = to_undirected
        self.seed = seed

    def handle(self):
        raw_folder = osp.join(self.root, 'raw')
        parsed_folder = osp.join(self.root, 'parsed')

        self._loop_over_folder(raw_folder, parsed_folder, self.root, label=0)

        self.map_id_to_graph_file = pd.DataFrame(self.map_id_to_graph_file, columns=['id', 'path', 'gt'])
        self.map_id_to_graph_file.to_csv(osp.join(self.root, 'map_id_to_graph.tsv'), sep='\t', index=False)

    def generate_train_test(self):
        pass

    def generate(self):
        records = self.map_id_to_graph_file
        # vul_file = records.iloc[id]['path']
        # vul_g = nx.read_gpickle(vul_file)
        return CustomSet(records, self.absolute_path, self.to_undirected)
