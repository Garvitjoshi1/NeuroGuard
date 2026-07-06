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

def dataset_summary(df: pd.DataFrame) -> None:
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn Names")

    for col in df.columns:
        print(f"- {col}")

    print("\nData Types")
    print(df.dtypes)

    print("\nMemory Usage")
    print(
        f"{df.memory_usage(deep=True).sum() /1024:.2f} KB"
    )

#==============DDISPLAY SAMPLE RECORD=======================

def preview_data(df: pd.DataFrame,rows: int = 5) -> None:
    print("\nPreview\n")
    print(df.head(rows))

#==============SCHEMA EXTRACTION=======================

def get_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema =pd.DataFrame({
        "Columns": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non null Count": df.count().values,
        "Null Count": df.isna().sum().values,
        "UNique Values": df.nunique().values
    })

    return schema

#==============TARGET COLUMN VALIDATION=======================

