import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
import os
import time
from torch.profiler import profile, record_function, ProfilerActivity

class Graph_Diffusion_Model(nn.Module):
    def __init__(self, data_config, diffusion_config, model_config, train_config, test_config, wandb_config, model):
        super(Graph_Diffusion_Model, self).__init__()
        
        # Data
        self.data_config = data_config
        self.train_data_path = self.data_config['train_data_path']
        self.test_data_path = self.data_config['test_data_path']
        self.history_len = self.data_config['history_len']
        self.future_len = self.data_config['future_len']
        self.num_classes = self.data_config['num_classes']
        self.edge_features = self.data_config['edge_features']
        
        # Diffusion
        self.diffusion_config = diffusion_config
        
        # Model
        self.model_config = model_config
        self.model = model # Edge_Encoder
        self.hidden_channels = self.model_config['hidden_channels']
        self.time_embedding_dim = self.model_config['time_embedding_dim']
        self.condition_dim = self.model_config['condition_dim']
        self.num_layers = self.model_config['num_layers']
        
        # Training
        self.train_config = train_config
        self.lr = self.train_config['lr']
        self.lr_decay_parameter = self.train_config['lr_decay']
        self.learning_rate_warmup_steps = self.train_config['learning_rate_warmup_steps']
        self.num_epochs = self.train_config['num_epochs']
        self.gradient_accumulation = self.train_config['gradient_accumulation']
        self.gradient_accumulation_steps = self.train_config['gradient_accumulation_steps']
        self.batch_size = self.train_config['batch_size'] if not self.gradient_accumulation else self.train_config['batch_size'] * self.gradient_accumulation_steps
        
        # Testing
        self.test_config = test_config
        self.test_batch_size = self.test_config['batch_size']
        self.model_path = self.test_config['model_path']
        self.eval_every_steps = self.test_config['eval_every_steps']
        
        # WandB
        self.wandb_config = wandb_config
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.wandb_config['project'],
            entity=self.wandb_config['entity'],
            notes=self.wandb_config['notes'],
            job_type=self.wandb_config['job_type'],
            config={**self.data_config, **self.diffusion_config, **self.model_config, **self.train_config}
        )
        self.exp_name = self.wandb_config['exp_name']
        wandb.run.name = self.exp_name

        # Logging
        self.dataset = self.data_config['dataset']
        self.model_dir = os.path.join("experiments", self.exp_name)
        os.makedirs(self.model_dir, exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.dataset}_{log_name}"
        
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        log_dir = os.path.join(self.model_dir, log_name)
        file_handler = logging.FileHandler(log_dir)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.log.addHandler(file_handler)
        
        self.log_loss_every_steps = self.train_config['log_loss_every_steps']        
        
        # Build Components
        self._build_train_dataloader()
        self._build_test_dataloader()
        self._build_model()
        self._build_optimizer()
        
    def train(self):
        """
        Trains the diffusion-based trajectory prediction model.

        This function performs the training of the diffusion-based trajectory prediction model. It iterates over the specified number of epochs and updates the model's parameters based on the training data. The training process includes forward propagation, loss calculation, gradient computation, and parameter updates.

        Returns:
            None
        """
        torch.autograd.set_detect_anomaly(True)
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges, future_len=self.future_len)
        def model_fn(x, edge_index, t, condition=None):
            if self.model_config['name'] == 'edge_encoder':
                return self.model.forward(x, edge_index, t, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_residual':
                return self.model.forward(x, edge_index, t, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(x, t=t, condition=condition, mode='future')
                
        for epoch in tqdm(range(self.num_epochs)):
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"epoch": epoch, "learning_rate": current_lr})
            
            total_loss = 0
            ground_truth_fut = []
            pred_fut = []
            
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_training"):
                    if self.gradient_accumulation:
                        for data in self.train_data_loader:
                            history_edge_features = data["history_edge_features"]
                            future_edge_features = data["future_edge_features"]
                            future_edge_indices_one_hot = future_edge_features[:, :, 0]
                            # Check if any entry in future_edge_indices_one_hot is not 0 or 1
                            if not torch.all((future_edge_indices_one_hot == 0) | (future_edge_indices_one_hot == 1)):
                                continue  # Skip this datapoint if the condition is not met

                            self.optimizer.zero_grad()
                            for i in range(min(self.gradient_accumulation_steps, history_edge_features.size(0))):
                                # Calculate history condition c
                                self.line_graph.x = history_edge_features[i].unsqueeze(0)   # unsqueeze to be able to handle different batch sizes when using Edge Encoders that do not rely on GAT layers
                                if self.model_config['name'] == 'edge_encoder':
                                    c = self.model.forward(x=self.line_graph.x, edge_index=self.line_graph.edge_index, mode='history')
                                elif self.model_config['name'] == 'edge_encoder_residual':
                                    c = self.model.forward(x=self.line_graph.x, edge_index=self.line_graph.edge_index, mode='history')
                                elif self.model_config['name'] == 'edge_encoder_mlp':
                                    c = self.model.forward(x=self.line_graph.x, mode='history')

                                x_start = future_edge_indices_one_hot[i].unsqueeze(0)   # (1, num_edges)
                                # Get loss and predictions
                                loss, preds = dif.training_losses(model_fn, c, x_start=x_start, line_graph=self.line_graph)   # preds are of shape (num_edges,)

                                ground_truth_fut.append(x_start)
                                pred_fut.append(preds)

                                total_loss += loss / self.gradient_accumulation_steps
                                (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                            self.optimizer.step()

                    else:
                        for data in self.train_data_loader:
                            history_edge_features = data["history_edge_features"]
                            future_edge_features = data["future_edge_features"]
                            future_edge_indices_one_hot = future_edge_features[:, :, 0]
                            # Check if any entry in future_edge_indices_one_hot is not 0 or 1
                            if not torch.all((future_edge_indices_one_hot == 0) | (future_edge_indices_one_hot == 1)):
                                continue

                            batch_size = future_edge_indices_one_hot.size(0)
                            if self.model_config['name'] == 'edge_encoder_mlp':
                                if batch_size == self.batch_size:
                                    future_edge_indices_one_hot = future_edge_indices_one_hot.view(self.batch_size, self.num_edges)
                                else:
                                    future_edge_indices_one_hot = future_edge_indices_one_hot.view(batch_size, self.num_edges)

                            self.optimizer.zero_grad()
                            # Calculate history condition c
                            self.line_graph.x = history_edge_features
                            if self.model_config['name'] == 'edge_encoder':
                                c = self.model.forward(x=self.line_graph.x, edge_index=self.line_graph.edge_index, mode='history')
                            elif self.model_config['name'] == 'edge_encoder_residual':
                                c = self.model.forward(x=self.line_graph.x, edge_index=self.line_graph.edge_index, mode='history')
                            elif self.model_config['name'] == 'edge_encoder_mlp':
                                c = self.model.forward(x=self.line_graph.x, mode='history')

                            x_start = future_edge_indices_one_hot
                            # Get loss and predictions
                            loss, preds = dif.training_losses(model_fn, c, x_start=x_start, line_graph=self.line_graph)

                            #x_start = x_start.squeeze(-1)   # (bs, num_edges, 1) -> (bs, num_edges)
                            ground_truth_fut.append(x_start)
                            pred_fut.append(preds)

                            total_loss += loss
                            loss.backward()
                            self.optimizer.step()

                    self.scheduler.step()
            
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            
            avg_loss = total_loss / len(self.train_data_loader)
            f1_score = F1Score(task='binary', average='macro', num_classes=2)
            f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach(), torch.flatten(torch.cat(ground_truth_fut)).detach())
            # Logging
            if epoch % self.log_loss_every_steps == 0:
                wandb.log({"epoch": epoch, "average_loss": avg_loss.item()})
                wandb.log({"epoch": epoch, "average_F1_score": f1_epoch.item()})
                self.log.info(f"Epoch {epoch} Average Loss: {avg_loss.item()}")
                print("Epoch:", epoch)
                print("Loss:", avg_loss.item())
                print("F1:", f1_epoch.item())
                
            if (epoch + 1) % self.eval_every_steps == 0:
                print("Evaluating on test set...")
                sample_list, ground_truth_hist, ground_truth_fut = self.get_samples(task='predict')
                fut_ratio, f1, avg_sample_length = self.eval(sample_list, ground_truth_hist, ground_truth_fut)
                wandb.log({"Test F1 Score": f1.item()})
                wandb.log({"Test Future ratio": fut_ratio})
                wandb.log({"Average test sample length": avg_sample_length})
                        
            if self.train_config['save_model'] and (epoch + 1) % self.train_config['save_model_every_steps'] == 0:
                self.save_model()
