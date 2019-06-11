import os
import tensorflow as tf
from termcolor import colored

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_run_args, set_logger

if __name__ == '__main__':
    logger = set_logger(colored('optimizer', 'magenta'))

    arg = get_run_args()

    temporary_file_name, config = optimize_graph(arg)
    optimized_graph_filepath = os.path.join(arg.model_dir, "optimized_graph.pbtxt")

    tf.gfile.Rename(
        temporary_file_name,
        optimized_graph_filepath,
        overwrite=True
    )
    logger.info(f"Serialized graph to {optimized_graph_filepath}")
