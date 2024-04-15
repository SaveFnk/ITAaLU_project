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


if __name__ == '__main__':
    prepare_hatespeech_v2_dataset()
