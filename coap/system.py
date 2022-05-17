import logging
import os
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .coap_body_model import COAPBodyModel
from .dataset import build_datasets, build_loaders
from .utils import get_world_size, get_rank

log = logging.getLogger(__name__)


class ImplicitBody(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hparams.update(cfg)
        self.model = COAPBodyModel(os.path.join(self.hparams.bm_path, 'neutral', 'model.npz'))

        self.datasets: Dict[str, Optional[List[Tuple[str, DictConfig, Dataset]]]] = {
            'train': None,  # is built when calling self.train_dataloader
            'val': None,  # is built when calling self.val_dataloader
            'test': None  # is built when calling self.test_dataloader
        }

        self.iou = nn.ModuleDict({
            stage: nn.ModuleList([
                tm.SumMetric() for i in range(len(self.hparams.datasets[stage]))
            ]) for stage in ['val', 'test']
        })
        print(self.iou)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch: Dict, batch_idx: int):
        occ = self.forward(batch['points'], batch['surface_part_posed_points'], batch['backward_transform'])
        loss = F.mse_loss(occ, batch['occ'])
        iou = self.compute_iou(occ, batch['occ'])
        iou = iou.mean()
        self.log(f'train_loss', loss, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'train_iou', iou, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: Dict, batch_idx: int, dataloader_idx: Optional[int], stage: str):
        occ = self.forward(batch['points'], batch['surface_part_posed_points'], batch['backward_transform'])
        iou = self.compute_iou(occ, batch['occ'])
        self.iou[stage][dataloader_idx].update(iou)

    def _shared_eval_epoch_end(self, _, stage: str):
        for i, (name, _, _) in enumerate(self.datasets[stage]):
            self.log(f"{stage}/{name}_iou", self.iou[stage][i].compute(), prog_bar=True, logger=True)
            self.iou[stage][i].reset()

    def validation_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def dataloader(self, stage: str, concat=False):
        self.datasets[stage] = build_datasets(self.hparams, stage)
        datasets = [d for _, _, d in self.datasets[stage]]
        if concat:
            return build_loaders(self.hparams, [torch.utils.data.ConcatDataset(datasets)], stage)[0]
        return build_loaders(self.hparams, datasets, stage)

    def train_dataloader(self):
        num_gpus = get_world_size()
        rank = get_rank()
        pl.seed_everything((3407 + self.current_epoch) * num_gpus + rank, workers=True)
        return self.dataloader('train', concat=True)

    def val_dataloader(self):
        return self.dataloader('val')

    def test_dataloader(self):
        return self.dataloader('test')

    @staticmethod
    def compute_iou(occ1, occ2):
        """ Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.
        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        """
        # Also works for 1-dimensional data
        if len(occ1.shape) >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if len(occ2.shape) >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union)

        return iou
