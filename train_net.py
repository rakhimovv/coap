import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from coap import ImplicitBody

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    pl.seed_everything(3407, workers=True)

    model: pl.LightningModule = ImplicitBody(cfg)

    trainer: pl.Trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            ModelCheckpoint(save_last=True, save_weights_only=False, monitor='val/movi_val_iou', save_top_k=2,
                            mode='max')
        ],
    )

    if cfg.weights_path is not None:
        weights_absolute_path = hydra.utils.to_absolute_path(cfg.weights_path)
        model.load_state_dict(torch.load(weights_absolute_path, map_location=model.device)['state_dict'], strict=False)

    # train / test
    if not cfg.eval_only:
        trainer.fit(model)
    else:
        # PossibleUserWarning: Using `DistributedSampler` with the dataloaders.
        # During `trainer.test()`, it is recommended to use `Trainer(devices=1)`
        # to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings
        # use `DistributedSampler` that replicates some samples to make sure all devices
        # have same batch size in case of uneven inputs.
        trainer.test(model)


if __name__ == "__main__":
    main()
