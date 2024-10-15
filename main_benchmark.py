# from seml.experiment import Experiment
import numpy as np
import seml
import torch
from torch.utils.data import DataLoader
from models.rnn import EdgeRNN
from models.run_rnn import Trainer
from dataset.trajectory_dataset_rnn import TrajectoryDataset
from dataset.trajectory_dataset_rnn import collate_fn


experiment = seml.Experiment()

@experiment.automain
def main(data_config, wandb_config, model_config, training_config, testing_config, eval):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TrajectoryDataset(file_path=data_config['train_data_path'],
                                      edge_features=data_config['edge_features'],
                                      device=device,
                                      history_len=data_config['history_len'],
                                      future_len=data_config['future_len'],
                                      true_future=testing_config['true_future'],
                                      mode='train')
    train_dataloader = DataLoader(train_dataset,
                                batch_size=training_config['batch_size'],
                                shuffle=True, collate_fn=collate_fn)

    val_dataset = TrajectoryDataset(file_path=data_config['val_data_path'],
                                edge_features=data_config['edge_features'],
                                device=device,
                                history_len=data_config['history_len'],
                                future_len=data_config['future_len'],
                                true_future=testing_config['true_future'],
                                mode='train')
    val_dataloader = DataLoader(val_dataset,
                            batch_size=testing_config['batch_size'],
                            shuffle=True, 
                            collate_fn=collate_fn)

    output_size = train_dataset.stop_token + 2
    rnn_model = EdgeRNN(num_edge_features_rnn=1, 
                    num_edge_features=data_config['num_edge_features'], 
                    hidden_size=model_config['hidden_size'], 
                    output_size=output_size, 
                    num_layers=model_config['num_layers'], 
                    dropout=model_config['dropout'], 
                    model_type=model_config['model_type'])

    model = Trainer(wandb_config,
                data_config,
                rnn_model,
                model_type=model_config['model_type'],
                dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                padding_value=train_dataset.padding_value,
                stop_token=train_dataset.stop_token,
                num_edge_features=data_config['num_edge_features'], 
                future_len=data_config['future_len'], history_len=data_config['history_len'],
                baseline_acc=1 / train_dataset.avg_degree, 
                device=device, 
                true_future=testing_config['true_future'],
                learning_rate=training_config['lr'])

    model.train(epochs=training_config['num_epochs'])

    test_dataset = TrajectoryDataset(file_path=data_config['test_data_path'],
                                edge_features=data_config['edge_features'],
                                device=device,
                                history_len=data_config['history_len'],
                                future_len=data_config['future_len'],
                                true_future=testing_config['true_future'],
                                mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=testing_config['batch_size'], shuffle=True, collate_fn=collate_fn)
    model.test(test_dataloader, mode='fixed', max_prediction_length=data_config['future_len'])
    print("Stop token:", train_dataset.stop_token)
    print("Padding value:", train_dataset.padding_value)