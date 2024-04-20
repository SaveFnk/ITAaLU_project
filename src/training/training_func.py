import pickle
from datetime import datetime

import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocessing.hatespeech_dataset_querying import hatespeech_v2_load_train_and_validation_set, \
    hatespeech_v2_load_test_set


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def convert_train_and_validation_to_dataset(train_df, validation_df, test_df):
    train_dataset = datasets.Dataset.from_pandas(df=train_df, split="train")
    validation_dataset = datasets.Dataset.from_pandas(df=validation_df, split="validation")
    test_dataset = datasets.Dataset.from_pandas(df=test_df, split="test")

    ds = datasets.DatasetDict()
    ds["train"] = train_dataset
    ds["validation"] = validation_dataset
    ds["test"] = test_dataset
    return ds


def predict_text(text, tokenizer):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")


def prepare_dataset_dict():
    train_df, validation_df = hatespeech_v2_load_train_and_validation_set()
    test_df = hatespeech_v2_load_test_set()
    print(f"Train set size {len(train_df)}, validation set {len(validation_df)}, and test set {len(test_df)}")

    # prepare the dataset in the correct format
    hate_dataset_dict = convert_train_and_validation_to_dataset(train_df, validation_df, test_df=test_df)
    return hate_dataset_dict


def save_model(model, model_name):
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"{model_name.replace('/', '_')}_{time}"

    print(f"Saving model to {model_name}")
    with open(f"./{model_name}.pickle", 'wb') as f:
        pickle.dump(model, f)
    print("Saved model !!!")
