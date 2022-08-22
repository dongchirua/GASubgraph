from dataclasses import dataclass
from random import sample
import time
from pathlib import Path
from queue import Empty
from typing import Optional, Set
import sys, traceback

import networkx as nx
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GNNExplainer

from vulexp.explanation.subgraphx import SubgraphX
from vulexp.ml_models.gin import GIN
from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.data_models.reveal_data import Reveal

from ga_subgraph.explainer import GASubX
from ga_subgraph.fitness import classifier
from ga_subgraph.individual import Individual
from ga_subgraph.utils import helper, extract_node_from_mask


def explainers(blackbox_model, classifier, sample, k_node, device, n_generation, CXPB, MUTPB, tournsize, subgraph_building_method, **kwargs):
    ga_explainer = GASubX(blackbox_model, classifier, device, Individual,
                            n_generation, CXPB, MUTPB,
                          tournsize, subgraph_building_method)
    subgraphx = SubgraphX(model=blackbox_model,
                          min_nodes=k_node, n_rollout=n_generation)
    gnn_explainer = GNNExplainer(blackbox_model, epochs=n_generation, return_type='raw', log=False)

    start = time.time()
    ga_subgraph, _ = ga_explainer.explain(sample, k_node, verbose=False)
    end = time.time()
    ga_dt = end - start

    start = time.time()
    sub_explainer = subgraphx.explain(x=sample.x.to(device), edge_index=sample.edge_index.to(device), max_nodes=k_node)
    end = time.time()
    sub_dt = end - start

    start = time.time()
    _, gnn_edge_mask = gnn_explainer.explain_graph(
        sample.x.to(device), sample.edge_index.to(device))
    end = time.time()
    gnn_explainer_nodes = extract_node_from_mask(gnn_edge_mask, k_node, sample)
    gnn_dt = end - start

    return ga_subgraph, ga_dt, list(sub_explainer.coalition), sub_dt, gnn_explainer_nodes, gnn_dt

def get_method():
    saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                      checkpoint_path="weights/Reveal-GIN-auc_pos=0.78-optimal_t=0.560-f1=0.34-epoch=04.ckpt")
    return saved_model


def get_output_string(sample_id, num_nodes, output_y, output_pred, 
                        ga_dtime, ga_explain_prob, inv_ga_explain_prob, ga_fidelity, 
                        sub_dtime, sub_explain_prob, inv_sub_explain_prob, sub_fidelity,
                        gnn_dtime, gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity):

    part_01 = f'{sample_id} \t {num_nodes} \t {output_y} \t {output_pred:.4f} \t {ga_explain_prob:.4f} \t {inv_ga_explain_prob:.4f} \t {ga_fidelity:.4f} \t {ga_dtime} \t'
    part_02 = f'{sub_explain_prob:.4f} \t {inv_sub_explain_prob:.4f} \t {sub_fidelity:.4f} \t {sub_dtime} \t'
    part_03 = f'{gnn_explain_prob:.4f} \t {inv_gnn_explain_prob:.4f} \t {gnn_fidelity:.4f} \t {gnn_dtime} \t'
    return part_01 + part_02 + part_03 + '\n'


def print_qsize(event, precv_pipe, queue):
    try:
        pbar = tqdm(bar_format="{desc}")
        while not (event.is_set() and queue.empty()):
            if not precv_pipe.poll():
                continue
            remaining = precv_pipe.recv()
            qsize = queue.qsize()

            pbar.desc = f"rem : {remaining:4}, " + \
                        f"qsize : {qsize:2},"
            pbar.update()
            time.sleep(0.1)
        pbar.close()
    except NotImplementedError as err:
        print("JoinableQueue.qsize has not been implemented;" +
              "remainging can't be shown")


def handle_output(sample_id, num_nodes, sample_label, output_prediction, lock, file, 
                    dtime, ga_explain_prob, inv_ga_explain_prob, ga_fidelity, 
                    sub_dtime, sub_explain_prob, inv_sub_explain_prob, sub_fidelity,
                  gnn_dtime, gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity):
    """
    Obtains the output string from `path` and `output` and writes
    to `file` by acquiring a `lock`
    """
    output_string = get_output_string(sample_id, num_nodes, sample_label, output_prediction, 
                                        dtime, ga_explain_prob, inv_ga_explain_prob, ga_fidelity, 
                                        sub_dtime, sub_explain_prob, inv_sub_explain_prob, sub_fidelity,
                                        gnn_dtime, gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity)
    lock.acquire()
    file.write(output_string)
    file.flush()
    lock.release()


def load_data(data_dir, queue, event, psend_pipe,
              to_undirected, seed, wait_time=0.1, ):
    reveal_dataset = Reveal(data_dir, to_undirected=to_undirected, seed=seed)
    _, _, reveal_test = reveal_dataset.generate_train_test()

    # n_items = len(reveal_test)
    n_items = 165

    count = 0
    while count < n_items:
        if queue.full():
            time.sleep(wait_time)
            continue
        else:
            sample = reveal_test.get(count)
            queue.put((sample, count))
            psend_pipe.send((n_items - count))
            count += 1

    event.set()
    queue.join()


def main(queue, event, model, device, lock, output_path, k_node, n_generation, CXPB, MUTPB,
                tournsize, subgraph_building_method):
    file = open(output_path.as_posix(), "a")
    model.eval().to(device)
    while not (event.is_set() and queue.empty()):
        try:
            graph, sample_id = queue.get(block=True, timeout=0.1)
        except Empty:
            continue
        y = int(graph.y)

        if k_node <= graph.num_nodes:
            try:
                predict_prod = classifier(graph, model, device)
                ga_explain, ga_dtime, sub_explain, sub_dtime, gnn_explainer, gnn_dtime = explainers(model, classifier, graph, k_node, device, n_generation, CXPB, MUTPB,
                                        tournsize, subgraph_building_method)
                ga_explain_prob, inv_ga_explain_prob, ga_fidelity = helper(ga_explain, graph, model, predict_prod, device)
                sub_explain_prob, inv_sub_explain_prob, sub_fidelity = helper(
                    sub_explain, graph, model, predict_prod, device)
                gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity = helper(gnn_explainer, graph, model, predict_prod, device)
                
                handle_output(sample_id, graph.num_nodes, y, predict_prod, lock, file, ga_dtime, ga_explain_prob,
                            inv_ga_explain_prob, ga_fidelity, sub_dtime, sub_explain_prob, inv_sub_explain_prob, sub_fidelity,
                            gnn_dtime, gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity)
            except Exception as e:
                print(f'error at {sample_id}')
                print(e)
            
        queue.task_done()
    file.close()


if __name__ == "__main__":
    args = sys.argv
    node_constraint = int(args[1])
    device = torch.device(args[2])

    @dataclass
    class Args:
        """A class containing arguments used for setting up the dataset and model."""
        batch_size: int = 32  # Batch size for the training loop.
        num_workers: int = 2  # Number of workers to use for the data loader.
        learning_rate: float = 0.001  # Learning rate.
        weight_decay: float = 5e-4  # Weight decay.
        num_epochs: int = 300  # Number of training epochs.
        num_layers: int = 3  # Number of message passing layers in the GNN model.
        hidden_features: int = 32  # Dimensionality of the hidden layers in the GNN.
        dropout: float = 0.2  # Dropout probability.
        seed: int = 27  # Random seed.
        pre_train: bool = True  # Change to False if want to retrain
        CXPB = 0.55
        MUTPB = 0.25
        tournsize = 11
        to_undirected = True
        subgraph_building_method = "zero_filling"
        n_generation = 150


    args = Args()

    mp.set_start_method("spawn")
    data_dir = 'data/reveal/'
    output_path = Path(f'statistic_{node_constraint}.tsv')
    n_handler_workers = 4

    queue = mp.JoinableQueue(10)
    event = mp.Event()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()

    # Initialize processes
    reader_process = mp.Process(
        target=load_data,
        args=(data_dir, queue, event, psend_pipe, args.to_undirected, args.seed)
    )
    detector_processes = [mp.Process(target=main, args=(queue, event, get_method(), device, lock, output_path, node_constraint,
                                                        args.n_generation, args.CXPB, args.MUTPB, args.tournsize, args.subgraph_building_method))
                          for i in range(n_handler_workers)]

    try:
        # Starting processes
        reader_process.start()
        [dp.start() for dp in detector_processes]

        print_qsize(event, precv_pipe, queue)

        # Waiting for processes to complete
        [dp.join() for dp in detector_processes]
        reader_process.join()
    except Exception as e:
        print(e)

    print('Closing everything')
    [c.close() for c in closables]
