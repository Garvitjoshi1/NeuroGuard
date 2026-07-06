import pandas as pd

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def dataset_shape(self):
        return{
            "rows": self.df.shape[0],
            "columns": self.df.shape[1]
        }
    
    def missing_values(self):
        report = pd.DataFrame({
            "Missing Count": self.df.isnull().sum() ,
             "Missing %": (
                self.df.isnull().sum() / len(self.df)* 100).round(2)
        })

        return report.sort_values(
            "Missing Count",
            ascending=False
        )
    
    def duplicate_rows(self):
        return self.df.duplicated().sum()
    
    def data_types(self):
        return pd.DataFrame({
            "Data type": self.df.dtypes.astype(str),
            "Non Null": self.df.count(),
            "Unique": self.df.nunique()
        })
    
    def target_distribution(self, target):
        counts = self.df[target].value_counts()
        percentages = (
            self.df[target].value_counts(normalize = True) * 100
        ).round(2)

        report = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages
        })

        return report
    
    def numerical_summary(self):
        return self.df.describe().T
    
    def categorical_summary(self):
        categorical = self.df.select_dtypes(
            include = ["object", "string"]
        )

        summary = {}
        for column in categorical.columns:
            summary[column] = categorical[column].value_counts()

        return summary
    
    def invalid_zero_values(self):
        numerical = self.df.select_dtypes(include = "number")
        report  = {}
        for col in numerical.columns:
            report[col] = (numerical[col] == 0).sum()

        return pd.Series(report)
    
    def generate_report(self, target):
        print("\n", "="*20, "DATA VALIDATION REPORT ", "="*20 )

        print("\n Shape")
        print(self.dataset_shape())

        print("\nDuplicate Rows")
        print(self.duplicate_rows())

        print("\nMissing Values")
        print(self.missing_values())

        print("\nData Types")
        print(self.data_types())

        print("\nTarget Distribution")
        print(self.target_distribution(target))

        print("\nNumerical Summary")
        print(self.numerical_summary())

        print("\nCategorical Summary")

        categorical = self.categorical_summary()

        for col, values in categorical.items():
            print(f"\n{col}")
            print(values)

        print("\n Zero value count")
        print(self.invalid_zero_values())
        print("\n Validation Complete.")