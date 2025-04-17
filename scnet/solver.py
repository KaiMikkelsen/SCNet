import torch
from pathlib import Path
from .utils import copy_state, EMA, new_sdr
from .apply import apply_model
from .ema import ModelEMA
from . import augment
from .loss import spec_rmse_loss
from tqdm import tqdm
from .log import logger
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast
import wandb

def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())

class Solver(object):
    def __init__(self, loaders, model, optimizer, config, args):
        self.config = config
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.device = next(iter(self.model.parameters())).device
        self.accelerator = Accelerator()
        self.scaler = GradScaler()

        self.stft_config = {
            'n_fft': config.model.nfft,
            'hop_length': config.model.hop_size,
            'win_length': config.model.win_size,
            'center': True,
            'normalized': config.model.normalized
        }
        # Exponential moving average of the model
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(config.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(config.data.samplerate * config.data.shift),
                                  same=config.augment.shift_same)]
        if config.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(config.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        self.folder = args.save_path
        # Checkpoints
        self.checkpoint_file = Path(args.save_path) / 'checkpoint.th'
        self.best_state = None
        self.best_nsdr = 0
        self.epoch = -1
        self._reset()

    def _serialize(self, epoch, steps=0):
        package = {}
        package['state'] = self.model.state_dict()
        package['best_nsdr'] = self.best_nsdr
        package['best_state'] = self.best_state
        package['optimizer'] = self.optimizer.state_dict()
        package['epoch'] = epoch
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        if steps: 
            checkpoint_with_steps = Path(self.checkpoint_file).with_name(f'checkpoint_{epoch+1}_{steps}.th')
            self.accelerator.save(package, checkpoint_with_steps)
        else:
            self.accelerator.save(package, self.checkpoint_file)

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, map_location=self.accelerator.device)
            self.model.load_state_dict(package['state'])
            self.best_nsdr = package['best_nsdr']
            self.best_state = package['best_state']
            self.optimizer.load_state_dict(package['optimizer'])
            self.epoch = package['epoch']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.config.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self, trial=None):

        if trial is not None:
            lr = trial.suggest_loguniform("optim.lr", 1e-5, 1e-3)
            decay_rate = trial.suggest_uniform("optim.decay_rate", 0.85, 0.99)
            decay_step = trial.suggest_int("optim.decay_step", 5, 30)
            momentum = trial.suggest_uniform("optim.momentum", 0.7, 0.99)
            beta2 = trial.suggest_uniform("optim.beta2", 0.9, 0.999)
            weight_decay = trial.suggest_loguniform("optim.weight_decay", 1e-6, 1e-2)
            optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

            # Dims: control model size
            dims_base = trial.suggest_categorical("model.dims_base", [16, 32, 64])
            dims = [dims_base // (2 ** i) for i in reversed(range(4))]  # e.g. [4, 8, 16, 32]

            # STFT settings
            nfft = trial.suggest_categorical("model.nfft", [2048, 4096, 8192])
            hop_size = nfft // 4
            win_size = nfft

            # Band split settings
            band_SR = trial.suggest_categorical("model.band_SR", [
                [0.175, 0.392, 0.433], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]
            ])
            band_stride = trial.suggest_categorical("model.band_stride", [
                [1, 4, 16], [1, 2, 8], [2, 4, 16]
            ])
            band_kernel = trial.suggest_categorical("model.band_kernel", [
                [3, 4, 16], [3, 3, 8], [5, 5, 16]
            ])

            # Conv + RNN
            conv_depths = trial.suggest_categorical("model.conv_depths", [[3, 2, 1], [2, 2, 2], [4, 3, 2]])
            compress = trial.suggest_int("model.compress", 2, 8)
            conv_kernel = trial.suggest_int("model.conv_kernel", 1, 5)
            num_dplayer = trial.suggest_int("model.num_dplayer", 3, 8)
            expand = trial.suggest_int("model.expand", 1, 3)
                                       
            batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

            augment_remix_proba = trial.suggest_uniform("augment.remix.proba", 0.5, 1.0)
            augment_remix_group_size = trial.suggest_int("augment.remix.group_size", 2, 6)

            augment_scale_proba = trial.suggest_uniform("augment.scale.proba", 0.5, 1.0)
            augment_scale_min = trial.suggest_uniform("augment.scale.min", 0.1, 0.5)
            augment_scale_max = trial.suggest_uniform("augment.scale.max", 1.0, 1.5)

            augment_flip = trial.suggest_categorical("augment.flip", [True, False])


            # Update the config with the suggested hyperparameters
            self.config.optim.lr = lr
            self.config.optim.decay_rate = decay_rate
            self.config.optim.decay_step = decay_step       
            self.config.optim.momentum = momentum
            self.config.optim.beta2 = beta2
            self.config.optim.weight_decay = weight_decay
            self.config.optim.optimizer = optimizer
            self.config.model.dims = dims
            self.config.model.dims_base = dims_base
            self.config.model.nfft = nfft
            self.config.model.hop_size = hop_size
            self.config.model.win_size = win_size
            self.config.model.band_SR = band_SR
            self.config.model.band_stride = band_stride
            self.config.model.band_kernel = band_kernel
            self.config.model.conv_depths = conv_depths
            self.config.model.compress = compress       
            self.config.model.conv_kernel = conv_kernel
            self.config.model.num_dplayer = num_dplayer
            self.config.model.expand = expand
            self.config.data.batch_size = batch_size
            self.config.augment.remix.proba = augment_remix_proba
            self.config.augment.remix.group_size = augment_remix_group_size
            self.config.augment.scale.proba = augment_scale_proba
            self.config.augment.scale.min = augment_scale_min
            self.config.augment.scale.max = augment_scale_max   
            self.config.augment.flip = augment_flip




        # Optimizing the model
        for epoch in range(self.epoch + 1, self.config.epochs):
            #Adjust learning rate
            for param_group in self.optimizer.param_groups:
              param_group['lr'] = self.config.optim.lr * (self.config.optim.decay_rate**((epoch)//self.config.optim.decay_step))
              logger.info(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']}")

            # Train one epoch
            self.model.train()
            metrics = {}
            logger.info('-' * 70)
            logger.info(f'Training Epoch {epoch + 1} ...')


            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}')
            
            # Log metrics to WandB after each epoch
            wandb.log({
            'train_loss': metrics['train']['loss'],
            'train_sdr': metrics['train'].get('sdr', None),  # Use get() to avoid errors if missing
            'train_nsdr': metrics['train'].get('nsdr', None),
            'epoch': epoch + 1,
            })



            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid['nsdr']
                        b = bvalid['nsdr']
                        if a > b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname



            formatted = self._format_train(metrics['valid'])
            logger.info(
                f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}')
            
            valid_nsdr = metrics['valid']['nsdr']
            # Save the best model
            if valid_nsdr > self.best_nsdr:
              logger.info('New best valid nsdr %.4f', valid_nsdr)
              self.best_state = copy_state(state)
              self.best_nsdr = valid_nsdr

            wandb.log({
                'best_valid_nsdr': valid_nsdr,
                'epoch': epoch + 1
            })

            if self.accelerator.is_main_process:
                self._serialize(epoch)
            if epoch == self.config.epochs - 1:
                break


    def _run_one_epoch(self, epoch, train=True):
        config = self.config
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        data_loader.sampler.epoch = epoch

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)

        averager = EMA()

        if self.accelerator.is_main_process:
            data_loader = tqdm(data_loader)

        for idx, sources in enumerate(data_loader):
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]

            if not train:
                estimate = apply_model(self.model, mix, split=True, overlap=0)
            else:
                with autocast():
                   estimate = self.model(mix)

            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)

            loss = spec_rmse_loss(estimate, sources, self.stft_config)

            losses = {}

            losses['loss'] = loss
            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                nsdrs = self.accelerator.reduce(nsdrs, reduction="mean")
                total = 0
                for source, nsdr in zip(self.config.model.sources, nsdrs):
                    losses[f'nsdr_{source}'] = nsdr
                    total += nsdr
                losses['nsdr'] = total / len(self.config.model.sources)

            # optimize model in training mode
            if train:
                scaled_loss = self.scaler.scale(loss)
                self.accelerator.backward(scaled_loss)
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
                if self.config.save_every and (idx+1) % self.config.save_every == 0:
                    self._serialize(epoch, idx+1)

            losses = averager(losses)
            
            del loss, estimate

        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return losses
