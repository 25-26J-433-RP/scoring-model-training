import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/aes_dataset/training_set_rel3.tsv"):
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    return df

def train_val_split(df, target="domain1_score", test_size=0.2, random_state=42):
    X = df["essay"].values
    y = df[target].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
