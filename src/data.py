import pandas as pd

def load_dataset(features_path: str, labels_path: str) -> pd.DataFrame:
    """
    Loads features.csv and labels.csv and returns a single dataframe:
      columns = [strain, strain_rate, stress]
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    if len(features) != len(labels):
        raise ValueError("Mismatch between features and labels rows")

    if "stress" not in labels.columns:
        raise ValueError("labels.csv must contain a 'stress' column")

    df = features.copy()
    df["stress"] = labels["stress"].values

    required_cols = {"strain", "strain_rate", "stress"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found={list(df.columns)}")

    return df
