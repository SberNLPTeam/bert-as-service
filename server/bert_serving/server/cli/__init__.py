import os
import tensorflow as tf
from termcolor import colored
from bert_serving.server import BertServer
from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_run_args, set_logger, get_benchmark_parser, get_shutdown_parser, \
    get_optimizer_args_parser
from bert_serving.server.benchmark import run_benchmark


def main():
    with BertServer(get_run_args()) as server:
        server.join()


def benchmark():
    args = get_run_args(get_benchmark_parser)
    run_benchmark(args)


def terminate():
    args = get_run_args(get_shutdown_parser)
    BertServer.shutdown(args)


def optimize():
    logger = set_logger(colored('OPTIMIZER', 'magenta'))

    arg = get_run_args(get_optimizer_args_parser)

    temporary_file_name, config = optimize_graph(arg)
    optimized_graph_filepath = os.path.join(arg.model_dir, "optimized_graph.pbtxt")

    tf.gfile.Rename(
        temporary_file_name,
        optimized_graph_filepath,
        overwrite=True
    )
    logger.info(f"Serialized graph to {optimized_graph_filepath}")
