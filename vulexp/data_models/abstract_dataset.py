from abc import ABC, abstractmethod
import os
import nltk
from pathlib import Path
from typing import Tuple
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from vulexp.data_models.interface import CustomSet
from vulexp.data_processing.tokenize_node import symbolic_tokenize
from vulexp.data_processing.parse_data_point import joern_to_networkx

from pytorch_lightning import LightningDataModule

# uncomment to resolve path issue
# cwd = '/workspace' 
cwd = '.'
w2v_path = os.path.join(cwd, 'data/Word2Vec/li_et_al_wv')

type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}
type_one_hot = np.eye(len(type_map))


class BaseDataModule(LightningDataModule):
    """
        data -> handle() -> _loop_over_folder() -> _reader()
    """
    map_id_to_graph_file = None

    def __init__(self, root, gtype='smt', over_write=False):
        self.root = root
        self.over_write = over_write
        self.word2vec = Word2Vec.load(w2v_path)
        self.error_files = []
        self.over_sampling_data = None
        self.non_data = []
        self.vul_data = []
        self.count = 0
        self.feat_dim = 64

        self.gtype = gtype
        if gtype == 'cpg':
            self.etype = (
                'FLOWS_TO', 'CONTROLS',
                'DEF', 'USE', 'REACHES',
                'IS_AST_PARENT_OF'
            )
            self.node_feature = True
        elif gtype == 'smg':
            # https://joern.readthedocs.io/en/latest/databaseOverview.html
            self.etype = ('FLOWS_TO', 'REACHES')
            self.node_feature = False
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
        g = joern_to_networkx(nodes, edges, types=self.etype)
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
                except KeyError:
                    embedding = np.zeros(self.feat_dim)
                except ValueError:
                    embedding = np.zeros(self.feat_dim)
                nrp = np.add(nrp, embedding)
                # line_feat.append(embedding.tolist())
            if len(actual_line_tokens) > 0:
                fNrp = np.divide(nrp, len(symbolic_line_tokens))
            else:
                fNrp = nrp
            if self.node_feature:
                node_type = cnode['type']
                node_feature = type_one_hot[type_map[node_type] - 1].tolist()
                node_feature.extend(fNrp.tolist())
            else:
                node_feature = fNrp
            # as by reference, node ci in G will be modified
            cnode['feat'] = node_feature
            cnode['symbolic'] = symbolic_line_tokens
        return G

    def len(self) -> int:
        if self.map_id_to_graph_file is None:
            raise Exception('Data is empty')
        return len(self.map_id_to_graph_file)

    @abstractmethod
    def generate_train_test(self) -> Tuple[CustomSet, CustomSet, CustomSet]:
        raise NotImplementedError

    @abstractmethod
    def handle(self):
        raise NotImplementedError

    @classmethod
    def _serialize_n_count_graph(cls, graph, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # create parent folder if missed
        try:
            nx.write_gpickle(graph, save_path)
        except Exception as e:
            print(e)
