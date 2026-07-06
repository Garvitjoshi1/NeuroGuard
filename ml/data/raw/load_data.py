from pathlib import Path
import logging

import pandas as pd
import yaml

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)

logger =logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuarion file not found: \n{config_file}"
        )
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    logger.info("Configuation loaded successfully.")
    return config

#==============DATASET LOADER=======================

def load_dataset(config: dict) -> pd.DataFrame:
    dataset_path = Path(config["dataset"]["raw_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )
    if dataset_path.suffix != ".csv":
        raise ValueError(
            "Dataset must be CSV file."
        )
    
    logger.info(f"Loading dataset: \n{dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info("Dataset loaded successfully.")

    return df

#==============DATASET SUMMARY=======================

