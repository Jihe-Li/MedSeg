import os
import math
import wandb
import torch
import shutil
import logging
import torch.nn.functional as F

from collections import defaultdict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import get_scheduler
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

import networks
import datasets
import losses
import metrics
import utils
from utils import io
from utils.log_buffer import LogBuffer
from datasets.collate import collate_fn

logger = get_logger(__name__, log_level="INFO")


class Segmentator:
    """Base trainer class."""

    def __init__(self, config, is_train):
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.resume = config.resume

        self.output_root = HydraConfig.get().run.dir
        self.ckpt_dir = os.path.join(self.output_root, 'checkpoints')
        self.save_dir = os.path.join(self.output_root, 'results')
        self.log_file = os.path.join(self.output_root, 'train_log.txt')

        self._init_accelerator()
        self._init_logger()
        self._init_dataloader()
        self._init_networks()
        if is_train:
            self._init_optimizer()
            if config.use_wandb:
                self.wandbfigs = {}
        else:
            self.test_log_file = os.path.join(self.output_root, 'test_log.txt')

    def _init_accelerator(self):
        self.accelerator = Accelerator(log_with='wandb' if self.is_train and self.config.use_wandb else None)
        self.device = self.accelerator.device
        self.mixed_precision = self.accelerator.mixed_precision
        self.num_proc = self.accelerator.num_processes
        self.acc_steps = self.accelerator.state.deepspeed_plugin.gradient_accumulation_steps \
            if hasattr(self.accelerator.state, 'deepspeed_plugin') and self.accelerator.state.deepspeed_plugin is not None else self.config.gradient_accumulation_steps
        self.max_grad_norm = self.accelerator.state.deepspeed_plugin.gradient_clipping \
            if hasattr(self.accelerator.state, 'deepspeed_plugin') and self.accelerator.state.deepspeed_plugin is not None else self.config.gradient_clipping

        set_seed(3407)

    def _init_logger(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)
            self.accelerator.init_trackers(
                project_name='Segmentation',
                # config=vars(self.config),
                init_kwargs={"wandb": {"name": self.config.name,
                                       "dir": self.output_root}},
            )

        self.accelerator.wait_for_everyone()

        if self.is_train:
            self.log_buffer = LogBuffer()
        self.test_buffer = LogBuffer()

    def _init_networks(self):
        """Initialize networks."""
        self.network = networks.__dict__[f'get_{self.config.network.name}'](self.config.network)

    def _init_dataloader(self):
        """Initialize dataloader."""
        config = self.config.datasets
        if self.is_train:
            self.train_dataset = datasets.__dict__[f'{config.name}Dataset'](config, 'train', self.mixed_precision)
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=config.batch_size_per_gpu, 
                num_workers=config.num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn
            )
        self.test_dataset = datasets.__dict__[f'{config.name}Dataset'](config, config.inference, self.mixed_precision)
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=config.batch_size_per_gpu, 
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
    def _init_optimizer(self):
        """Initialize optimizers and schedulers."""
        config = self.config
        self.optimizer = AdamW(self.network.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

        # before accelerate.prepare, we need to consider multi-process
        # after accelerate.prepare, len(dataloader) is divided by num_proc
        num_update_per_epoch = math.ceil(
            len(self.train_loader) / self.acc_steps / self.num_proc
        )
        self.max_steps = config.max_epochs * num_update_per_epoch

        self.scheduler = get_scheduler(
            name=config.scheduler.lr_policy,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.max_steps * self.num_proc * config.scheduler.warmup_ratio),
            num_training_steps=self.max_steps * self.num_proc,
        )
        
    def load_ckpt(self):
        config = self.config
        if config.resume_from != "latest":
            path = os.path.basename(config.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{config.resume_from}' does not exist. Starting a new training run."
            )
            config.resume_from = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.ckpt_dir, path))
            self.epoch = int(path.split("-")[1])
            self.step = int(path.split("-")[2])

    def save_ckpt(self):
        config = self.config
        if config.ckpt_max_num is not None:
            ckpts = os.listdir(self.ckpt_dir)
            ckpts = [d for d in ckpts if d.startswith("ckpt")]
            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(ckpts) >= config.ckpt_max_num:
                num_to_remove = len(ckpts) - config.ckpt_max_num + 1
                removing_checkpoints = ckpts[0:num_to_remove]

                logger.info(
                    f"{len(ckpts)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(
                        self.ckpt_dir, removing_checkpoint
                    )
                    if self.accelerator.is_main_process:
                        shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(self.ckpt_dir, f"ckpt-{self.epoch}-{self.step}")
        self.accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def train(self):
        """Train the model."""
        config = self.config
        total_batch_size = config.datasets.batch_size_per_gpu * self.num_proc * self.acc_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {config.max_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {config.datasets.batch_size_per_gpu}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {self.acc_steps}")
        logger.info(f"  Total optimization steps = {self.max_steps}")
        logger.info(f"  Number of parameters of {config.network.name}: {self.count_params(self.network)/1e6:.2f}M")

        self.network, self.optimizer, self.scheduler, self.train_loader, self.test_loader = self.accelerator.prepare(
                self.network, self.optimizer, self.scheduler, self.train_loader, self.test_loader
            )

        self.epoch = 0
        self.step = 0
        if self.resume:
            self.load_ckpt()

        self.progress_bar = tqdm(
            range(self.step, self.max_steps),
            disable=not self.accelerator.is_local_main_process,
            ncols=80
        )
        self.progress_bar.set_description("Steps")

        for epoch in range(self.epoch, config.max_epochs):
            self.epoch = epoch + 1
            self._train_epoch()
            if self.epoch % config.save_freq == 0:
                self.save_ckpt()
                self._test_epoch()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def _train_epoch(self):
        """Train the model for one epoch."""
        config = self.config
        self.network.train()
        acc_metrics = defaultdict(float)

        for data in self.train_loader:
            with self.accelerator.accumulate(self.network):
                acc_metrics = self._train_step(data, acc_metrics)

            if self.accelerator.sync_gradients:
                self.step += 1
                self.scheduler.step()
                self.log_buffer.update(acc_metrics)
                acc_metrics = defaultdict(float)

                if self.accelerator.is_main_process:
                    if self.step % config.print_freq == 0:
                        summary = self.log_buffer.summary()
                        summary.update(
                            {
                                "lr": self.optimizer.param_groups[0]['lr'],
                            }
                        )
                        self.progress_bar.set_postfix(**summary)
                        self.progress_bar.update(config.print_freq)
                        
                        message = "(epoch: %d, iters: %d) " % (self.epoch, self.step)
                        message += " ".join(
                            ["%s: %.5f" % (k, v) for k, v in summary.items()]
                        )
                        with open(self.log_file, "a") as log_file:
                            log_file.write("%s\n" % message)

                        summary.update(
                            {
                                "epoch": self.epoch,
                                "step": self.step,
                            }
                        )
                        self.accelerator.log(summary, self.step)

                if self.step >= self.max_steps:
                    break

    def network_forward(self, data):
        inputs = torch.cat([data['ct_ten'], data['pt_ten']], dim=1)
        pred_logits = self.network(inputs).float()
        return pred_logits

    def _train_step(self, data, acc_metrics):
        config = self.config
        pred_logits = self.network_forward(data)

        seg_gt = data['gtv_ten'].bool().float()
        loss = 0
        if config.loss.lambda_bce > 0:
            bce = losses.bce_loss_with_logits(pred_logits, seg_gt)
            loss += config.loss.lambda_bce * bce
            acc_metrics["bce_loss"] = bce.detach().cpu().clone()

        if config.loss.lambda_focal > 0:
            focal = losses.focal_loss_with_logits(pred_logits, seg_gt)
            loss += config.loss.lambda_focal * focal
            acc_metrics["focal_loss"] = focal.detach().cpu().clone()

        if config.loss.lambda_dice > 0:
            pred_probab = torch.sigmoid(pred_logits)
            dice = losses.dice_loss(pred_probab, seg_gt)
            loss += config.loss.lambda_dice * dice
            acc_metrics["dice_loss"] = dice.detach().cpu().clone()

        acc_metrics["loss"] = loss.detach().clone()
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        acc_metrics["loss"] = loss.detach().clone()
        return acc_metrics

    @torch.no_grad()
    def test(self):
        """Test the model's performance."""
        config = self.config
        total_batch_size = config.datasets.batch_size_per_gpu * self.num_proc * self.acc_steps

        logger.info("***** Running testing *****")
        logger.info(f"  Num examples = {len(self.test_dataset)}")
        logger.info(
            f"  Instantaneous batch size per device = {config.datasets.batch_size_per_gpu}"
        )
        logger.info(
            f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {self.acc_steps}")
        logger.info(f"  Number of parameters of {config.network.name}: {self.count_params(self.network)/1e6:.2f}M")

        self.network, self.test_loader = self.accelerator.prepare(self.network, self.test_loader)

        self.epoch = 0
        self.step = 0
        self.load_ckpt()

        self.network.eval()

        acc_metrics = defaultdict(float)
        for data in tqdm(self.test_loader, ncols=80, desc="Steps", 
                            total=len(self.test_loader),
                            disable=not self.accelerator.is_local_main_process):
            acc_metrics = self._test_step(data, acc_metrics)

        if self.accelerator.is_main_process:
            summary = self.test_buffer.summary()
            message = "(Test --> epoch: %d, iters: %d)\n" % (self.epoch, self.step)
            message += "\n".join(
                ["%s: %.5f" % (k, v) for k, v in summary.items()]
            )
            with open(self.test_log_file, "a") as log_file:
                log_file.write("%s\n" % message)

        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _test_epoch(self):
        self.network.eval()
        acc_metrics = defaultdict(float)

        for data in self.test_loader:
            acc_metrics = self._test_step(data, acc_metrics)

        if self.accelerator.is_main_process:
            summary = self.test_buffer.summary()
            message = "(Test --> epoch: %d, iters: %d) " % (self.epoch, self.step)
            message += " ".join(
                ["%s: %.5f" % (k, v) for k, v in summary.items()]
            )
            with open(self.log_file, "a") as log_file:
                log_file.write("%s\n" % message)
                
            if hasattr(self, 'wandbfigs') and len(self.wandbfigs) > 0:
                summary.update(self.wandbfigs)
                self.wandbfigs = {}
            self.accelerator.log(summary, self.step)

    @torch.no_grad()
    def _test_step(self, data, acc_metrics):
        pred_logits = self.network_forward(data)
        seg_pred = (torch.sigmoid(pred_logits) > 0.5).float()
        seg_gt = data['gtv_ten'].bool().float()

        dice = metrics.comp_dice(seg_pred, seg_gt)
        acc_metrics["dice"] = dice
        count = seg_gt.shape[0]

        self.test_buffer.update(acc_metrics, count)
        acc_metrics = defaultdict(float)
        return acc_metrics

    @torch.no_grad()
    def inference(self):
        """Inference registration results."""
        config = self.config
        total_batch_size = config.datasets.batch_size_per_gpu * self.num_proc

        logger.info("***** Running inference *****")
        logger.info(f"  Num examples = {len(self.test_dataset)}")
        logger.info(
            f"  Instantaneous batch size per device = {config.datasets.batch_size_per_gpu}"
        )
        logger.info(
            f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Number of parameters of {config.network.name}: {self.count_params(self.network)/1e6:.2f}M")

        self.network, self.test_loader = self.accelerator.prepare(
            self.network, self.test_loader
        )

        self.epoch = 0
        self.step = 0
        self.load_ckpt()

        self.network.eval()
        acc_metrics = defaultdict(float)

        for data in tqdm(self.test_loader, ncols=80, desc="Steps", 
                            total=len(self.test_loader),
                            disable=not self.accelerator.is_local_main_process):
            acc_metrics = self._inference_step(data, acc_metrics)
            self.test_buffer.update(acc_metrics)
            acc_metrics = defaultdict(float)

        if self.accelerator.is_main_process:
            summary = self.test_buffer.summary()
            message = "(Test --> epoch: %d, iters: %d)\n" % (self.epoch, self.step)
            message += "\n".join(
                ["%s: %.5f" % (k, v) for k, v in summary.items()]
            )
            with open(self.test_log_file, "a") as log_file:
                log_file.write("%s\n" % message)

        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _inference_step(self, data, acc_metrics):
        pred_logits = self.network_forward(data)
        seg_pred = (torch.sigmoid(pred_logits) > 0.5).float()
        ct_ten = data['ct_ten'].float()
        pt_ten = data['pt_ten'].float()
        seg_gt = data['gtv_ten'].bool().float()

        count = seg_gt.shape[0]
        dice = metrics.comp_dice(seg_pred, seg_gt)
        acc_metrics["dice"] = dice
        self.test_buffer.update(acc_metrics, count)
        acc_metrics = defaultdict(float)

        for i in range(count):
            dataset_name, sample_id, params = data['dataset_name'][i], data['sample_id'][i], data['params'][i]
            save_folder = os.path.join(self.save_dir, f'Epoch{self.epoch}_Step{self.step}', dataset_name)
            os.makedirs(save_folder, exist_ok=True)
            save_path_ct = os.path.join(save_folder, sample_id, sample_id + '_ct.nii.gz')
            io.save_image(ct_ten[i][0], save_path_ct, params, is_tensor=True)
            save_path_pt = os.path.join(save_folder, sample_id, sample_id + '_pt.nii.gz')
            io.save_image(pt_ten[i][0], save_path_pt, params, is_tensor=True)
            save_path_pred = os.path.join(save_folder, sample_id, sample_id + '_pred.nii.gz')
            io.save_image(seg_pred[i][0], save_path_pred, params, is_tensor=True)
            save_path_gt = os.path.join(save_folder, sample_id, sample_id + '_gt.nii.gz')
            io.save_image(seg_gt[i][0], save_path_gt, params, is_tensor=True)

        return acc_metrics

    def count_params(self, network, optimizable_only=True):
        if optimizable_only:
            total = sum(p.numel() for p in network.parameters() if p.requires_grad)
        else:
            total = sum(p.numel() for p in network.parameters())
        return total
