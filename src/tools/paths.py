from pathlib import Path
from typing import Optional


def embedding_path(root, ds: str, embedding_version: Optional[str]):
    return Path(root) / "data" / ds / "encoded_features" / f"{embedding_version}.csv" if embedding_version else None

def ds_dir(root):
    return Path(root) / "PGA_LSTM" / "Datasets"

def drivers_path(root, ds):
    if ds == "fcr":
        return Path(root) / "PGA_LSTM" / "Datasets" / "FCR_2013_2018_Drivers.csv"