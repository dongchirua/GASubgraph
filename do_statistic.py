from dataclasses import dataclass
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


def inference(sample, model, device):
    batch = torch.zeros(sample.x.shape[0], dtype=int, device=device)
    output = model(sample.x.to(device), sample.edge_index.to(device), batch)
    pred = torch.sigmoid(output).item()
    return pred


def get_method():
    saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                      checkpoint_path="weights/Reveal-GIN-auc_pos=0.78-optimal_t=0.560-f1=0.34-epoch=04.ckpt")
    return saved_model


def get_output_string(sample_id, num_nodes, output_y, output_pred):
    return str(sample_id) + "\t" + str(num_nodes) + '\t' + str(output_y) + '\t' + str(output_pred) + "\n"


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


def handle_output(sample_id, num_nodes, sample_label, output_prediction, lock, file):
    """
    Obtains the output string from `path` and `output` and writes
    to `file` by acquiring a `lock`
    """
    output_string = get_output_string(sample_id, num_nodes, sample_label, output_prediction)
    lock.acquire()
    file.write(output_string)
    file.flush()
    lock.release()


def load_data(data_dir, queue, event, psend_pipe,
              to_undirected, seed, wait_time=0.1, ):
    reveal_dataset = Reveal(data_dir, to_undirected=to_undirected, seed=seed)
    _, _, reveal_test = reveal_dataset.generate_train_test()

    # n_items = len(reveal_test)  # todo: remove when debug finished
    n_items = 1000
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


def main(queue, event, detector, device, lock, output_path):
    time.sleep(3)
    file = open(output_path.as_posix(), "a")
    detector.eval().to(device)
    while not (event.is_set() and queue.empty()):
        try:
            graph, sample_id = queue.get(block=True, timeout=0.1)
        except Empty:
            continue
        y = int(graph.y)
        batch = torch.zeros(graph.x.shape[0], dtype=int, device=device)
        output = detector(graph.x.to(device), graph.edge_index.to(device), batch)
        predict_prod = torch.sigmoid(output).item()
        queue.task_done()
        handle_output(sample_id, graph.num_nodes, y, predict_prod, lock, file)
    file.close()


if __name__ == "__main__":

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/reveal/'
    output_path = Path('statistic.txt')
    n_handler_workers = 2

    queue = mp.JoinableQueue(4)
    event = mp.Event()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()

    # Initialize processes
    reader_process = mp.Process(
        target=load_data,
        args=(data_dir, queue, event, psend_pipe, args.to_undirected, args.seed)
    )
    detector_processes = [mp.Process(target=main, args=(queue, event, get_method(), device, lock, output_path))
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
