from pathlib import Path
import pandas as pd


class DatasetValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def dataset_shape(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
        }

    def missing_values(self):

        report = pd.DataFrame(
            {
                "Missing Count": self.df.isnull().sum(),
                "Missing %": (
                    self.df.isnull().sum() / len(self.df) * 100
                ).round(2),
            }
        )

        return report.sort_values(
            by="Missing Count",
            ascending=False,
        )

    def duplicate_rows(self):
        return self.df.duplicated().sum()

    def data_types(self):

        return pd.DataFrame(
            {
                "Data Type": self.df.dtypes.astype(str),
                "Non Null": self.df.count(),
                "Unique": self.df.nunique(),
            }
        )

    def target_distribution(self, target):

        counts = self.df[target].value_counts()

        percentages = (
            self.df[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        return pd.DataFrame(
            {
                "Count": counts,
                "Percentage": percentages,
            }
        )

    def numerical_summary(self):
        return self.df.describe().T

    def categorical_summary(self):

        categorical = self.df.select_dtypes(
            include=["object", "string"]
        )

        summary = {}

        for column in categorical.columns:
            summary[column] = categorical[column].value_counts()

        return summary

    def generate_report(self, target):

        report_dir = Path("artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "shape": self.dataset_shape(),
            "duplicates": self.duplicate_rows(),
            "missing": self.missing_values(),
            "types": self.data_types(),
            "target": self.target_distribution(target),
            "numerical": self.numerical_summary(),
            "categorical": self.categorical_summary(),
        }

        report["missing"].to_csv(
            report_dir / "missing_values.csv"
        )

        report["types"].to_csv(
            report_dir / "data_types.csv"
        )

        report["target"].to_csv(
            report_dir / "target_distribution.csv"
        )

        report["numerical"].to_csv(
            report_dir / "numerical_summary.csv"
        )

        print("\n" + "=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)

        print("\nDataset Shape")
        print(report["shape"])

        print("\nDuplicate Rows")
        print(report["duplicates"])

        print("\nMissing Values")
        print(report["missing"])

        print("\nData Types")
        print(report["types"])

        print("\nTarget Distribution")
        print(report["target"])

        print("\nNumerical Summary")
        print(report["numerical"])

        print("\nCategorical Summary")

        for column, values in report["categorical"].items():
            print(f"\n{column}")
            print(values)

        print("\nValidation report saved to:")
        print(report_dir.resolve())

        print("\nValidation Complete.\n")

        return report