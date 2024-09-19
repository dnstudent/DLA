import os
import shutil
from pathlib import Path
from typing import Type, Optional, List, Any, Dict, Union, Tuple

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from src.tools.dicts import dict_hash


def model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags: Optional[List[str]]) -> Path:
    if ulterior_tags:
        root_dir = os.path.join(root_dir, *(list(map(str, ulterior_tags))))
    root_dir = Path(root_dir)
    return root_dir / model_class.__name__ / f"{autoencoder_version}" / f"{train_frac}" / f"{dict_hash(hparams_set)}"

def model_checkpoints_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) -> Path:
    return model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) / "checkpoints"

def model_logs_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) -> Path:
    return model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) / "logs"

def best_model(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) -> Path:
    return model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags) / "best_model.ckpt"

def train_model(root_dir: Path, model_class: Type[L.LightningModule], autoencoder_version: Optional[str], train_frac: float, hparams_set: Dict[str, Union[str, int, float, bool]], ulterior_tags: Optional[List[str]], train_ds: Dataset, val_ds: Dataset, log: bool, max_epochs: int, batch_size: int, trainer_kwargs: Optional[Dict[str, Any]]):
    root_dir = model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    checkpoint_dir = root_dir / "checkpoints"
    log_dir = root_dir / "logs"
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor="valid/score/t", mode="max")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=TensorBoardLogger(save_dir=log_dir) if log else None,
        log_every_n_steps=1,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        callbacks=[EarlyStopping(monitor="valid/score/t", mode="max", patience=100), checkpoint],
        enable_model_summary=False,
        **trainer_kwargs
    )
    model = model_class(**hparams_set)
    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7, persistent_workers=True),
        val_dataloaders=DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=7, persistent_workers=True)
    )
    shutil.copy(checkpoint.best_model_path, root_dir / "best_model.ckpt")
    return model_class.load_from_checkpoint(root_dir / "best_model.ckpt"), root_dir / "best_model.ckpt"

def train_or_load_best(root_dir, model_class: Type[L.LightningModule], autoencoder_version: Optional[str], train_frac: float, hparams_set: Dict[str, Union[str, int, float, bool]],
                       ulterior_tags: Optional[List[str]], train_ds: Dataset, val_ds: Dataset, log: bool, max_epochs: int, batch_size: int, force_retrain: bool, trainer_kwargs: Optional[Dict[str, Any]]) -> Tuple[L.LightningModule, Union[str, Path]]:
    if ulterior_tags is None:
        ulterior_tags = ["base"]
    if trainer_kwargs is None:
        trainer_kwargs = {}
    best_model_path = best_model(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags)
    if best_model_path.exists() and not force_retrain:
        return model_class.load_from_checkpoint(best_model_path), best_model_path
    if best_model_path.exists() and force_retrain:
        shutil.rmtree(model_dir(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags))
    return train_model(root_dir, model_class, autoencoder_version, train_frac, hparams_set, ulterior_tags, train_ds, val_ds, log, max_epochs, batch_size, trainer_kwargs)

