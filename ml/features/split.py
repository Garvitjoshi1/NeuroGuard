from pathlib import Path
import logging
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class DatasetSplitter:
    def __init__(
            self,
            df: pd.DataFrame,
            target: str,
            random_state: int = 42,
            test_size: float = 0.20,
            drop_columns: list = None
            ):
        self.df = df.copy()