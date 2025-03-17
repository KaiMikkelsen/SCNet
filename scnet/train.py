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
        project='SCNet',
        name=f"SCNet_{date_str}",
        config={'config': config, 'args': args}
    )

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    return model


def get_solver(args):
    with open(args.config_path, 'r') as file:
          config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

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


def main():
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

    solver = get_solver(args)
    accelerator.wait_for_everyone()
    solver.train()



if __name__ == "__main__":
    main()

