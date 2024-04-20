import os

import numpy as np
import pandas as pd


def prepare_hatespeech_v2_dataset(save=True):
    hate_df = pd.read_csv("data/toraman22_hate_speech_v2/Toraman22_hate_speech_v2.tsv", sep="\t")
    dataset_v2 = pd.read_csv("data\hatespeech_v2\hate_speech_dataset_v2.csv")

    print("Starting size of full dataset", hate_df.shape)
    hate_df = hate_df.drop_duplicates()
    print("Removed duplicates size of dataset", hate_df.shape)

    indexes_to_drop = []
    for index, row in hate_df.iterrows():
        try:
            favourite_int = int(row["tweet_id"])
        except:
            indexes_to_drop.append(index)

    print(f"Found {len(indexes_to_drop)} rows to drop")
    hate_df = hate_df.drop(indexes_to_drop)
    hate_df["tweet_id"] = hate_df['tweet_id'].astype(str)
    new_size = hate_df.shape
    print(f"New size is {new_size}")

    # verify that the IDs are present in the dataset on Github
    dataset_v2["TweetID"] = dataset_v2['TweetID'].astype(str)
    out_df = hate_df[hate_df["tweet_id"].isin(dataset_v2["TweetID"])]
    assert new_size == out_df.shape, "Problems occured while filtering! Sizes should match!"
    assert 128907 == out_df.shape[
        0], f"Size doesn't match with the required one! It should be 128907, but is {out_df.shape[0]}!"

    # filter by english
    out_df = out_df[out_df["language"] == 1].reset_index(drop=True)
    assert out_df.shape[0] == 68597, "Size of the final processed dataset isn't correct, errors occured!"
    print(f"Final dataset size is {out_df.shape}")

    # remove large spaces
    out_df["text"] = out_df["text"].str.replace(r"\s+", " ", regex=True)

    if save:
        save_file = "data/hatespeech_v2/prepared_hatespeech_v2.csv"
        print("Saving data")
        out_df.to_csv(save_file, index=False)
        print(f"Saved preprocessed data to: {save_file}")


def load_hatespeech_v2_dataset(file_path="data/hatespeech_v2/prepared_hatespeech_v2.csv", filter=True):
    df = pd.read_csv(file_path)

    if filter:
        # we are working on an NLP task, hence we will take only the tweet id, text, label, topic
        df = df[["tweet_id", "text", "label", "topic"]]
        # enforce data types
        # they can't be categories, because datasets library doesn't support them
        df["label"] = df["label"].astype(int)
        df["topic"] = df["topic"].astype(int)
        df["tweet_id"] = df["tweet_id"].astype(np.int64)

    return df


# open the file prepared_hatespeech_v2.csv, shuflle the data split it into training and testing and save them into two separate files
def split_hatespeech_v2_dataset(file_path="data/hatespeech_v2/prepared_hatespeech_v2.csv", test_size=0.2, save=True):
    df = pd.read_csv(file_path)
    # shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int((1-test_size) * df.shape[0])
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    if save:
        train_file = "data/hatespeech_v2/train_hatespeech_v2.csv"
        test_file = "data/hatespeech_v2/test_hatespeech_v2.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"Saved train data to: {train_file}")
        print(f"Saved test data to: {test_file}")

    return train_df, test_df


def hatespeech_v2_load_train_and_validation_set(large=False):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "./../../data/hatespeech_v2/train_hatespeech_v2.csv")
    if not os.path.exists(filename):
        split_hatespeech_v2_dataset()

    df = load_hatespeech_v2_dataset(file_path=filename)[["text", "label"]]
    train_len = int(df.shape[0] * 0.8)
    validation_df = df.iloc[train_len:]

    if large:
        filename = os.path.join(dirname, "./../../data/large_merged_training_set_toxigen_and_hate.csv")
        train_df = pd.read_csv(filename)
    else:
        train_df = df.iloc[:train_len]

    return train_df, validation_df


def hatespeech_v2_load_test_set():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "./../../data/hatespeech_v2/test_hatespeech_v2.csv")
    if not os.path.exists(filename):
        split_hatespeech_v2_dataset()

    df = load_hatespeech_v2_dataset(file_path=filename)[["text", "label"]]
    return df


def map_predicted_to_label_hatesv2(y_pred):
    y_pred_mapped = []
    for json_pair in y_pred:
        if json_pair["label"] == "Normal":
            y_pred_mapped.append(0)
        if json_pair["label"] == "Offensive":
            y_pred_mapped.append(1)
        if json_pair["label"] == "Hate speech":
            y_pred_mapped.append(2)
    return y_pred_mapped


# if __name__ == '__main__':
#    prepare_hatespeech_v2_dataset()
