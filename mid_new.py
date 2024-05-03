from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import dill
import torch
import os
import numpy as np
import os
import argparse
import pdb
import dill
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

#from .models import AutoEncoder, CustomEncoder


# Define the path to your training data
data_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/processed_data_no_social'
train_data_path = 'tdrive_edge_coordinates_traincustom.pkl'
# Define the batch size
batch_size = 1

# Step 1: Load Data with Dill
with open(os.path.join(data_path, train_data_path), 'rb') as f:
    train_data = dill.load(f)
    f.close()

# Step 2: Define Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Access row by index using .iloc
        # Process the row and return the data
        return row.values  # Assuming you want to return the values of the row

# Step 3: Create DataLoader
train_dataset = CustomDataset(train_data)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Iterate through the DataLoader
for batch in tqdm(dataloader):
    print(batch.size()[1])
    
class MID_new():
    def __init__(self, epochs, augment, eval_every, dataset, eval_at, encoder_dim, eval_mode, diffnet, tf_layer, 
                 override_attention_radius, scene_freq_mult_train, incl_robot_node, batch_size, preprocess_workers,
                 scene_freq_mult_eval, eval_batch_size, data_dir, lr, output_dir):
        self.epochs = epochs
        self.augment = augment
        self.eval_every = eval_every
        self.dataset = dataset
        self.eval_at = eval_at
        self.encoder_dim = encoder_dim
        self.eval_mode = eval_mode
        self.diffnet = diffnet
        self.tf_layer = tf_layer
        self.override_attention_radius = override_attention_radius
        self.scene_freq_mult_train = scene_freq_mult_train
        self.incl_robot_node = incl_robot_node
        self.batch_size = batch_size
        self.preprocess_workers = preprocess_workers
        self.scene_freq_mult_eval = scene_freq_mult_eval
        self.eval_batch_size = eval_batch_size
        self.data_dir = data_dir
        self.lr = lr
        self.output_dir = output_dir
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                train_loss = self.model.get_loss(batch)
                batch.set_description(f"Epoch {epoch}, MSE: {train_loss.item():.2f}")
                train_loss.backward()
                self.optimizer.step()
                
    def _build(self):
        self._build_dir()

        # self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        print("> Everything built. Have fun :)")
        
    def _build_dir(self):
        self.exp_name = self.output_dir
        self.model_dir = osp.join("./experiments",self.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        attributes = sorted(vars(self).items())
        for attr, value in attributes:
            if not attr.startswith('__') and not callable(value):
                self.log.info(f"{attr}: {value}")
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.data_dir, self.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.data_dir, self.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")
        
    def _build_encoder(self):
        self.encoder = CustomEncoder(self.hyperparams, 'cuda')
        
    def _build_model(self):
        """ Define Model """
        model = AutoEncoder(self.diffnet, self.tf_layer, self.encoder_dim, encoder=self.encoder)

        self.model = model.cuda()
        if self.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")
        
    def _build_train_loader(self):
        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='utf-8')
        
        self.train_dataset = CustomDataset(train_env,
                                           hyperparams=self.hyperparams,
                                           )
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print("> Training Dataset loaded!")
        
    def _build_eval_loader(self):
        with open(self.eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='utf-8')
        
        self.eval_dataset = CustomDataset(eval_env,
                                          hyperparams=self.hyperparams,
                                          )
        self.eval_data_loader = DataLoader(self.eval_dataset, batch_size=self.eval_batch_size, shuffle=True)
        
        print("> Eval Dataset loaded!")