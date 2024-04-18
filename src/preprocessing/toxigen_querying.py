import os

import pandas as pd


def load_toxigen_train_and_validation():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "./../../data/toxigen/annotated_train.csv")
    df = pd.read_csv(filename)[["label", "text", "target_group"]]

    train_len = int(df.shape[0] * 0.8)
    train_df = df.iloc[:train_len]
    validation_df = df.iloc[train_len:]
    print(f"Loaded {train_len}/{df.shape[0]} train samples and {df.shape[0] - train_len}/{df.shape[0]} validation samples")

    return train_df, validation_df


def load_big_toxigen_train_set():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "./../../data/toxigen/toxigen.csv")

    df = pd.read_csv(filename)[["label", "text", "target_group"]]
    return df
