from mid import MID
import argparse
import os
import yaml
import numpy as np
# from easydict import EasyDict
import seml
from seml.experiment import Experiment
# from sacred import Experiment

ex = Experiment()


@ex.automain
def main(dataset, lr, epochs, batch_size, diffnet, encoder_dim, tf_layer, eval_batch_size, k_eval, seed, eval_every, data_dir, eval_mode, 
         eval_at, sampling, conf, debug, preprocess_workers, offline_scene_graph, dynamic_edges, edge_state_combine_method, edge_influence_combine_method,
         edge_addition_filter, edge_removal_filter, override_attention_radius, incl_robot_node, map_encoding, augment, node_freq_mult_train,
         node_freq_mult_eval, scene_freq_mult_train, scene_freq_mult_eval, scene_freq_mult_viz, no_edge_encoding, device, eval_device):
    agent = MID(epochs, augment, eval_every, dataset, eval_at, encoder_dim, eval_mode, diffnet, tf_layer, 
                 override_attention_radius, scene_freq_mult_train, incl_robot_node, batch_size, preprocess_workers,
                 scene_freq_mult_eval, eval_batch_size, data_dir, lr)

    sampling = "ddim"
    steps = 5

    if eval_mode:
        agent.eval(sampling, 100//steps)
    else:
        agent.train()


if __name__ == '__main__':
    main()
