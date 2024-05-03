import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from shapely import Polygon, Point

from trajectory_gnn.data.base import *
from trajectory_gnn.utils.utils import get_suggested_number_dataloader_workers
from trajectory_gnn.config import DataConfig
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingDataset     
from trajectory_gnn.task.trajectory_forecasting import TrajectoryForecastingTask


class TrajectoryForecastingDataModule(pl.LightningDataModule):
    """DataModule for trajectory forecasting. """
    
    test_split_seed: int = 1337 # the test set is fixed
    
    def __init__(self, base_dataset: BaseDataset, config: DataConfig, rng: np.random.RandomState):
        super().__init__()
        self.base_dataset = base_dataset
        
        self.overfit = config.overfit
        if self.overfit:
            self.train_size, self.val_size, self.test_size = 1.0, 0.0, 0.0
        else:
            assert np.allclose(config.train_portion + config.val_portion + config.test_portion, 1.0), f'Split portions should add up to 1.0'
            self.test_size = int(max(1, len(self.base_dataset) * config.test_portion))
            self.val_size = int(max(1, len(self.base_dataset) * config.val_portion))
            self.train_size = len(self.base_dataset) - (self.test_size + self.val_size)
            assert self.train_size > 0, f'Not enough samples for given dataset split portions'
        
        self.num_workers = get_suggested_number_dataloader_workers() if config.num_workers is None else config.num_workers
        self.config = config
        self._rng = rng
        
        if self.overfit: # Overfit to the training set
            # We create disctintive datasets that all share the same indices
            idx_train, idx_val, idx_test = None, None, None
        elif config.split == 'random':
            idx_train, idx_val, idx_test = self.random_split()
        elif config.split == 'polygons':
            idx_train, idx_val, idx_test = self.polygon_split(Polygon(config.split_polygons['train']), 
                                                              Polygon(config.split_polygons['val']),
                                                              Polygon(config.split_polygons['test']) if 'test' in config.split_polygons else None)
        else:
            raise ValueError(f'Unsupported dataset split {config.split}')
            
        self.data_train = TrajectoryForecastingDataset(self.base_dataset, self.config, self._rng, idxs_subset=idx_train)
        self.data_val = TrajectoryForecastingDataset(self.base_dataset, self.config, self._rng, idxs_subset=idx_val)
        self.data_test = TrajectoryForecastingDataset(self.base_dataset, self.config, self._rng, idxs_subset=idx_test)
    
    
    def polygon_split(self, polygon_train: Polygon, polygon_val: Polygon, 
                      polygon_test: Polygon | None) -> Tuple[TensorType[int, 'num_train'], 
                                                                        TensorType[int, 'num_val'], TensorType[int, 'num_test']]:
        """ Splits according to pre-defined polygons. If no test polygon is given, it is simply all nodes not contained in train or val. """
        assert self.base_dataset.node_coordinates is not None, f'Polygon split can only be applied to datasets with coordinates'
        
        mask_nodes_train = torch.tensor([polygon_train.contains(Point(*p)) for p in self.base_dataset.node_coordinates.tolist()], dtype=bool)
        mask_nodes_val = torch.tensor([polygon_val.contains(Point(*p)) for p in self.base_dataset.node_coordinates.tolist()], dtype=bool)
        if polygon_test is not None:
            mask_nodes_test = torch.tensor([polygon_test.contains(Point(*p)) for p in self.base_dataset.node_coordinates.tolist()], dtype=bool)
        else:
            mask_nodes_test = ~(mask_nodes_train | mask_nodes_val)
        assert ((mask_nodes_train.long() + mask_nodes_val.long() + mask_nodes_test.long()) <= 1).all(), f'Polygons should be non-overlapping'
        
        # Categorize edges
        self.mask_edges_train = mask_nodes_train[self.base_dataset.edge_idxs].all(1)
        self.mask_edges_val = mask_nodes_val[self.base_dataset.edge_idxs].all(1)
        self.mask_edges_test = mask_nodes_test[self.base_dataset.edge_idxs].all(1)
        
        idxs_train, idxs_val, idxs_test = [], [], []
        for idx in range(len(self.base_dataset)):
            if self.mask_edges_train[self.base_dataset[idx]['edge_idxs']].all():
                idxs_train.append(idx)
            elif self.mask_edges_val[self.base_dataset[idx]['edge_idxs']].all():
                idxs_val.append(idx)
            elif self.mask_edges_test[self.base_dataset[idx]['edge_idxs']].all():
                idxs_test.append(idx)
        return torch.tensor(idxs_train), torch.tensor(idxs_val), torch.tensor(idxs_test) 
    
    def random_split(self) -> Tuple[TensorType[int, 'num_train'], TensorType[int, 'num_val'], TensorType[int, 'num_test']]:
        """ Randomly splits into train, val and test, where test is always according to the fixed test_split_seed """
        idx_non_test, idx_test = random_split(range(len(self.base_dataset)), [self.train_size + self.val_size, self.test_size], 
                                                  generator=torch.Generator().manual_seed(self.test_split_seed))
        idx_non_test, idx_test = idx_non_test.indices, idx_test.indices
        idx_train, idx_val = random_split(idx_non_test, [self.train_size, self.val_size])
        idx_train, idx_val = idx_train.indices, idx_val.indices
        return idx_train, idx_val, idx_test
        
    
    def register_task(self, task: TrajectoryForecastingTask):
        """ Registers a task to train, validation and test datasets, e.g. sets up edge complex reduction wrappers. """
        self.data_train.register_task(task, which='train')
        self.data_val.register_task(task, which='val')
        self.data_test.register_task(task, which='test')
        
    def train_dataloader(self):
        if not self.overfit:
            self.data_train.randomize_masked_edge_idxs()
        return DataLoader(self.data_train, batch_size=self.config.batch_size, collate_fn=self.data_train.collate, num_workers=self.num_workers, 
                          shuffle=self.config.shuffle, pin_memory=self.config.pin_memory, multiprocessing_context=self.config.multiprocessing_context)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.config.batch_size, collate_fn=self.data_train.collate, num_workers=self.num_workers, 
                          shuffle=False, pin_memory=self.config.pin_memory, multiprocessing_context=self.config.multiprocessing_context)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size, collate_fn=self.data_train.collate, num_workers=self.num_workers, 
                          shuffle=False, pin_memory=self.config.pin_memory, multiprocessing_context=self.config.multiprocessing_context)
    
    