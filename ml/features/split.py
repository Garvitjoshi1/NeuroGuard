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
        self.target = target
        self.random_state = random_state
        self.test_size = test_size
        self.drop_columns = drop_columns or []


    def prepare_features(self):
        logger.info("Preparing feature matrix...")
        X = self.df.drop(
            columns = self.drop_columns + [self.target]
        )

        y = self.df[self.target]

        logger.info(
            f"feature Matrix shape: {X.shape}"
        )

        logger.info(
            f"Target Shape: {y.shape}"
        )

        return X, y
    
    def split(self):
        X, y = self.prepare_features()
        logger.info("Performing Train/Test Split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size= self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info("Dataset split completed.")
        logger.info(f"Training Samples: {len(X_train)}")
        logger.info(f"Testing Samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def save_split(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir="artifacts/datasets"
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            X_train,
            output_dir / "X_train.pkl"
        )

        joblib.dump(
            X_test,
            output_dir / "X_test.pkl"
        )

        joblib.dump(
            y_train,
            output_dir / "y_train.pkl"
        )

        joblib.dump(
            y_test,
            output_dir / "y_test.pkl"
        )

        logger.info(
            f"Datasets saved to {output_dir.resolve()}"
        )

def split_dataset(
        df,
        config
):
    splitter = DatasetSplitter(
        df = df,
        target= config["model"]["target"],
        random_state= config["training"]["random_state"],
        test_size=config["training"]["test_size"],
        drop_columns=['id']
    )

    X_train, X_test, y_train, y_test = splitter.split()

    splitter.save_split(
        X_train,
        X_test,
        y_train,
        y_test
    )

    return X_train, X_test, y_train, y_test