import pandas as pd
from pathlib import Path
from .utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "ave_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace("-","_")
    )
    
    if "id" in df.columns:
        df = df.drop(columns = ["id"])
        logger.info("Dropped 'id' column")
        
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Missing values: \n{df.isnull.sum()}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return df