# Master's Thesis Code: Diffusion-Based Trajectory Prediction on Graphs

This repository contains the code for my master's thesis on diffusion-based trajectory prediction using denoising diffusion models on graphs. The project focuses on predicting future trajectories of agents (e.g., vehicles) in road networks represented as graphs.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
  - [Understanding the Configuration File](#understanding-the-configuration-file)
- [Logging and Monitoring](#logging-and-monitoring)
- [Qualitative Evaluation](#qualitative-evaluation)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a diffusion-based approach for trajectory prediction on graphs. The key components include:

- **Data Preparation**: Splitting the dataset into training, validation, and test sets.
- **Model Training**: Training the diffusion model using the specified configurations.
- **Evaluation**: Evaluating the trained model on the test set and logging metrics.
- **Qualitative Analysis**: Visualizing the predicted trajectories for qualitative assessment.

## Prerequisites

- **Operating System**: Linux
- **Python**: Version 3.12
- **Conda**: For managing the environment
- **CUDA**: For GPU acceleration (optional but recommended)

## Installation

1. **Clone the Repository**:

  ```bash
  git clone https://gitlab.com/yourusername/your-repo-name.git
  cd your-repo-name
  ```

2. **Create and Activate Conda Environment**:

  ```bash
  conda create -n d3pm_new python=3.12
  conda activate d3pm_new
  ```

3. **Install Dependencies**:

Install required packages using requirements.txt:

  ```bash
  pip install -r requirements.txt
  ```

4. **Set Up Weights & Biases (W&B)**:

  - Sign up for a W&B account at wandb.ai.

  - Log in from the command line:

    ```bash
    wandb login
    ```

## Data Preparation
The first step is to prepare the dataset by creating the train, validation, and test splits.

1. **Data Location**:
- The data should be stored in .h5 files located in a data/ directory.

2. **Create Train/Val/Test Splits**:

- Use the provided Jupyter notebook:

    ```bash
    notebooks/train_val_test_split.ipynb
    ```
- This notebook processes the raw data and splits it into training, validation, and test sets. There additionally is the option to split the sets based on trajectory coordinates.
- Run the notebook to generate the split datasets.

## Training
To train the model, run the main_diffusion.py script with the appropriate configuration file.

1. **Select Configuration**:

- Configuration files are located in the config/ directory.
- Choose the configuration file corresponding to your dataset (e.g., d3pm_pneuma.yaml).

2. **Set Training Mode**:

In the configuration file, ensure that the eval entry is set to False:

  ```yaml
  eval: False
  ```

3. **Run Training Script**:
Run training with seml.
  ```bash
  seml training_run_name add config/D3PM/pneuma_config.yaml
  seml training_run_name start
  ```
- Replace pneuma_config.yaml with your chosen configuration file.

4. **Monitor Training**:
- Training logs and metrics (e.g., loss, F1 score) are logged using Weights & Biases.
- You can monitor the training progress on your W&B dashboard.

## Evaluation
To evaluate the trained model on the test set:

1. **Update Configuration**:

- Set the eval entry to True in the configuration file:
  ```yaml
  eval: True
  ```
- Set the val_data_path entry to the desired set of data (validation or test).
  ```yaml
  val_data_path: path/to/test_data.h5
  ```
2. **Run Evaluation Script**:
Run evaluation with seml.
  ```bash
  seml testing_run_name add config/D3PM/pneuma_config.yaml
  seml testing_run_name start
  ```

3. **Results**:

- Evaluation metrics such as F1 score, ADE (Average Displacement Error), and FDE (Final Displacement Error) are logged to W&B.
- Samples generated during evaluation are saved for qualitative analysis (Adapt saving location in d3pm_graph_diffusion_model.py).

## Configuration
The behavior of the training and evaluation scripts is controlled via YAML configuration files. Below is an explanation of the entries in a sample configuration file.

**Understanding the Configuration File**
  ```yaml
  seml:
    executable: main_diffusion.py
    name: pneuma_d3pm_residual
    output_dir: experiments/pneuma_d3pm_residual
    project_root_dir: ../..
    conda_environment: /path/to/your/conda/environment/d3pm_new

  slurm:
    experiments_per_job: 1
    sbatch_options:
      partition: gpu_gtx1080
      gres: gpu:1
      mem: 64G
      cpus-per-task: 5
      time: 0-22:00 # D-hh-mm

  fixed:
    data:
      dataset: pneuma
      train_data_path: '/path/to/pneuma_train.h5'
      val_data_path: '/path/to/pneuma_val.h5'  # Change to validation set during training
      history_len: 5
      num_classes: 2

    diffusion_config:
      start: 1e-5   # beta_0
      stop: 0.09    # beta_T
      num_timesteps: 100  # T

    model:
      name: edge_encoder_residual # edge_encoder (GAT only), edge_encoder_mlp (MLP only), edge_encoder_residual (GAT+MLP)
      hidden_channels: 64
      time_embedding_dim: 16
      num_heads: 3
      num_layers: 3
      pos_encoding_dim: 16
      theta: 1.0    # Strength of residual layer

    training:
      lr: 0.009
      batch_size: 64
      optimizer: adam
      learning_rate_warmup_steps: 40
      num_epochs: 50
      lr_decay: 0.99
      log_loss_every_steps: 1
      log_metrics_every_steps: 2
      save_model: True
      save_model_every_steps: 2
      pretrained: False

    testing:
      batch_size: 32
      model_path: '/path/to/trained_model.pth'
      number_samples: 10  # Number of samples to generate
      eval_every_steps: 10  # Validation step during training

    wandb:
      exp_name: "pneuma_residual"
      project: "trajectory_prediction_using_denoising_diffusion_models"
      entity: "your_wandb_username"
      job_type: "test"
      notes: ""
      tags: ["pneuma", "edge_encoder_residual"]

    eval: True

  grid:
    data.future_len:
      type: choice
      options:
        - 0  # For conditional model. Works only with 'future_len' in edge_features
        - 2
        - 10

    data.edge_features:
      type: choice
      options:
        # - ['one_hot_edges', 'coordinates', 'pos_encoding', 'pw_distance', 'edge_length', 'edge_angles', 'num_pred_edges']
        - ['one_hot_edges', 'coordinates', 'pos_encoding', 'pw_distance', 'edge_length', 'edge_angles', 'num_pred_edges', 'future_len']
        # - ['one_hot_edges', 'coordinates', 'pos_encoding', 'pw_distance', 'edge_length', 'edge_angles', 'num_pred_edges', 'future_len', 'start_end']  # for trajectory planning

    diffusion_config.type:
      type: choice
      options:
        - 'linear'
        - 'cosine'

    model.transition_mat_type:
      type: choice
      options:
        - 'custom'  # for uniform transition matrix
        - 'marginal_prior'

    testing.conditional_future_len: # when using the conditional model, we can modify the sample length with differenc conditional future lengths. With the standard model, simply use None
      type: choice
      options:
        - None
        - 15
        - 20
  ```
Sections Explained
seml: Contains settings for experiment management, such as the executable script, experiment name, and environment paths.

slurm: Configurations for submitting jobs to a SLURM cluster (if applicable).

fixed: Parameters that are kept constant during experiments.

grid: Parameters that are varied during hyperparameter search or experimentation.

## Qualitative Evaluation
After evaluation, the generated samples are saved and can be visualized for qualitative assessment.

1. **Locate Saved Samples**:
- Samples are saved in thelocation specified in d3pm_graph_diffusion_model.py.

2. **Use the Evaluation Script**:
  ```bash
  notebooks/evaluation_script.ipynb
  ```
Run the notebook to visualize the predicted trajectories alongside the ground truth.

3. **Customize Visualizations**:
Modify the plotting functions to adjust visualization styles, focus on specific trajectories, or compare different models.

## Acknowledgments
This project was developed as part of my master's thesis. I would like to thank my supervisors Dominik Fuchsgruber, Marcel Kollovieh, David Lüdke, and Viktoria Dahmen for their guidance and support.

Institution: Technische Universität München