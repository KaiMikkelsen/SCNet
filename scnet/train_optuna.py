import os
import torch
from torch.utils.data import DataLoader
from .wav import get_wav_datasets
from .SCNet import SCNet
from .solver import Solver
import argparse
import yaml
from ml_collections import ConfigDict
from accelerate import Accelerator
from .log import logger
import wandb
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler  # You can choose other samplers if needed
from optuna.trial import TrialState
import uuid
import sys

accelerator = Accelerator()



def wandb_init(args: argparse.Namespace, config) -> None:
    """
    Initialize the Weights & Biases (wandb) logging system.

    Args:
        args: Parsed command-line arguments containing the wandb key.
        config: Configuration dictionary for the experiment.
        device_ids: List of GPU device IDs used for training.
        batch_size: Batch size for training.
    """

    # if args.wandb_key is None or args.wandb_key.strip() == '':
    #     wandb.init(mode='disabled')
    date_str = datetime.now().strftime('%Y-%m-%d')
    wandb.login(key="689bb384f0f7e0a9dbe275c4ba6458d13265990d")
    wandb.init(
        project='SCNet_optuna',
        name=f"SCNet_optuna{date_str}",
        config={'config': config, 'args': args}
    )

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    return model


def get_solver(args, trial):
    with open(args.config_path, 'r') as file:
          config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

        # Hyperparameters to optimize

    lr = trial.suggest_float("optim.lr", 1e-5, 1e-3)
    decay_rate = trial.suggest_float("optim.decay_rate", 0.85, 0.99)
    decay_step = trial.suggest_int("optim.decay_step", 5, 30)
    momentum = trial.suggest_float("optim.momentum", 0.7, 0.99)
    beta2 = trial.suggest_float("optim.beta2", 0.9, 0.999)
    weight_decay = trial.suggest_int('optim.weight_decay', 0, 1) # Example: Suggest integer 0 or 1
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    # Dims: control model size
    dims_base = trial.suggest_categorical("model.dims_base", [16, 32, 64])
    dims = [dims_base // (2 ** i) for i in reversed(range(4))]  # e.g. [4, 8, 16, 32]
    # STFT settings
    nfft = trial.suggest_categorical("model.nfft", [2048, 4096, 8192])
    hop_size = nfft // 4
    win_size = nfft
    # Band split settings
    sr_0 = trial.suggest_float('band_SR_0', 0.1, 0.3)      # Original: 0.175
    sr_1 = trial.suggest_float('band_SR_1', 0.3, 0.6)      # Original: 0.392
    sr_2 = trial.suggest_float('band_SR_2', 0.35, 0.7)     # Original: 0.433
    band_SR = [sr_0, sr_1, sr_2]
    stride_0 = trial.suggest_int('band_stride_0', 1, 3)     # Original: 1
    stride_1 = trial.suggest_int('band_stride_1', 2, 8)     # Original: 4
    stride_2 = trial.suggest_int('band_stride_2', 8, 24)    # Original: 16
    band_stride = [stride_0, stride_1, stride_2]
    kernel_0 = trial.suggest_int('band_kernel_0', 2, 5)     # Original: 3
    kernel_1 = trial.suggest_int('band_kernel_1', 3, 8)     # Original: 4
    kernel_2 = trial.suggest_int('band_kernel_2', 8, 24)    # Original: 16
    band_kernel = [kernel_0, kernel_1, kernel_2]
    # Conv + RNN
    conv_depths_0 = trial.suggest_int('model.conv_depths_0', 1, 4)     # Original: 3
    conv_depths_1 = trial.suggest_int('model.conv_depths_1', 1, 4)     # Original: 2                            
    conv_depths_2 = trial.suggest_int('model.conv_depths_2', 1, 4)     # Original: 1
    conv_depths = [conv_depths_0, conv_depths_1, conv_depths_2]
    compress = trial.suggest_int("model.compress", 2, 8)
    conv_kernel = trial.suggest_int("model.conv_kernel", 1, 5)
    num_dplayer = trial.suggest_int("model.num_dplayer", 3, 8)
    expand = trial.suggest_int("model.expand", 1, 3)
                               
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    augment_remix_proba = trial.suggest_int("augment.remix.proba", 1, 4)
    augment_remix_group_size = trial.suggest_int("augment.remix.group_size", 2)
    augment_scale_proba = trial.suggest_int("augment.scale.proba", 1, 4)
    augment_scale_min = trial.suggest_float("augment.scale.min", 0.1, 0.5)
    augment_scale_max = trial.suggest_float("augment.scale.max", 1.0, 1.5)
    augment_flip = trial.suggest_categorical("augment.flip", [True, False])
    # Update the config with the suggested hyperparameters
    config.optim.lr = lr
    config.optim.decay_rate = decay_rate
    config.optim.decay_step = decay_step       
    config.optim.momentum = momentum
    config.optim.beta2 = beta2
    config.optim.weight_decay = weight_decay
    config.optim.optimizer = optimizer
    config.model.dims = dims
    config.model.nfft = nfft
    config.model.hop_size = hop_size
    config.model.win_size = win_size
    config.model.band_SR = band_SR
    config.model.band_stride = band_stride
    config.model.band_kernel = band_kernel
    config.model.conv_depths = conv_depths
    config.model.compress = compress       
    config.model.conv_kernel = conv_kernel
    config.model.num_dplayer = num_dplayer
    config.model.expand = expand
    config.batch_size = batch_size
    config.augment.remix.proba = augment_remix_proba
    config.augment.remix.group_size = augment_remix_group_size
    config.augment.scale.proba = augment_scale_proba
    config.augment.scale.min = augment_scale_min
    config.augment.scale.max = augment_scale_max   
    config.augment.flip = augment_flip
    

    torch.manual_seed(config.seed)
    model = get_model(config)
  
    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if config.optim.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)
    elif config.optim.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)

    print("goung into get_wav_datasets")
    train_set, valid_set = get_wav_datasets(config.data)

    #use this to limit for testing
    # train_set = torch.utils.data.Subset(train_set, range(min(10, len(train_set))))  # First 10 samples
    # valid_set = torch.utils.data.Subset(valid_set, range(min(2, len(valid_set))))    # First 2 samples

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.misc.num_workers, drop_last=True)
    train_loader = accelerator.prepare_data_loader(train_loader)

    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=config.misc.num_workers)
    valid_loader = accelerator.prepare_data_loader(valid_loader)

    loaders = {"train": train_loader, "valid": valid_loader}


    model, optimizer = accelerator.prepare(model, optimizer)
    
    return Solver(loaders, model, optimizer, config, args)


def objective(trial: Trial):

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    parser.add_argument("--config_path", type=str, default='./conf/config.yaml', help="path to save checkpoint")
    args = parser.parse_args()


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isfile(args.config_path):
        print(f"Error: config file {args.config_path} does not exist.")
        sys.exit(1)

    with open(args.config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        wandb_init(args, config_dict)

    solver = get_solver(args, trial)
    accelerator.wait_for_everyone()
    return solver.train()


def main():


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    # parser.add_argument("--config_path", type=str, default='./conf/config.yaml', help="path to save checkpoint")
    # args = parser.parse_args()

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)

    # if not os.path.isfile(args.config_path):
    #     print(f"Error: config file {args.config_path} does not exist.")
    #     sys.exit(1)

    # with open(args.config_path, 'r') as file:
    #     config_dict = yaml.safe_load(file)
    #     wandb_init(args, config_dict)

    # solver = get_solver(args)
    # accelerator.wait_for_everyone()
    # solver.train()


    db_folder = "optunadb"
    os.makedirs(db_folder, exist_ok=True)  # Create folder if it doesn't exist

    unique_id = uuid.uuid4().hex[:8]  # Generate a short unique ID
    # Generate a unique filename for each study
    db_path = os.path.join(db_folder, f"scnet_optimization_{datetime.now().strftime('%Y-%m-%d')}_{unique_id}.sqlite3")

    # Create the study with the new database path
    study = optuna.create_study(
        direction="maximize",  # Change to "minimize" if optimizing a loss
        sampler=TPESampler(),  # TPE sampler for efficient search
        storage=f"sqlite:///{db_path}",  # Save in "optunadb" folder
        study_name=f"scnet_optimization_{datetime.now().strftime('%Y-%m-%d')}_{unique_id}"
    )
    

    study.optimize(lambda trial: objective(trial), n_trials=300)



    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()

