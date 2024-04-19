from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import tqdm
import os
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from evaluate import evaluator
import torch
import pickle 



from src.preprocessing.hatespeech_dataset_querying import hatespeech_v2_load_train_and_validation_set, map_predicted_to_label_hatesv2, hatespeech_v2_load_test_set
from src.preprocessing.toxigen_querying import load_toxigen_train_and_validation

import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

def convert_train_and_validation_to_dataset(train_df, validation_df):
    train_dataset = datasets.Dataset.from_pandas(df=train_df, split="train")
    validation_dataset = datasets.Dataset.from_pandas(df=validation_df, split="validation")
    ds = datasets.DatasetDict()
    ds["train"] = train_dataset
    ds["validation"] = validation_dataset
    return ds


# def main():
#     print("Hello World!")
#     train_df, validation_df = hatespeech_v2_load_train_and_validation_set()
#     # print(train_df)

#     # prepare the dataset in the correct format
#     hate_dataset_dict = convert_train_and_validation_to_dataset(train_df, validation_df)

#     tokenizer = AutoTokenizer.from_pretrained("unhcr/hatespeech-detection")
#     model = AutoModelForSequenceClassification.from_pretrained("unhcr/hatespeech-detection")

#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True)

#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         return metric.compute(predictions=predictions, references=labels)

#     tokenized_datasets = hate_dataset_dict.map(tokenize_function, batched=True)
#     training_args = TrainingArguments(output_dir="test_trainer")
#     metric = evaluate.load("accuracy")

#     training_args = TrainingArguments(output_dir="test_trainer", 
#                                   evaluation_strategy="epoch")

#     small_train_dataset = tokenized_datasets["train"]   # .shuffle(seed=42).select(range(1000))
#     small_eval_dataset = tokenized_datasets["validation"]   # .shuffle(seed=42).select(range(1000))

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=small_train_dataset,
#         eval_dataset=small_eval_dataset,
#         compute_metrics=compute_metrics,
#     )
#     trainer.train()


def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return inputs


def main_pp():
    print("PP time")
    train_df, validation_df = hatespeech_v2_load_train_and_validation_set()
    test_df = hatespeech_v2_load_test_set()
    test_dataset = datasets.Dataset.from_pandas(df=test_df, split="test")
    print(f"Train set size {len(train_df)}, validation set {len(validation_df)}, and test set {len(test_df)}")
    # print(train_df)

    # prepare the dataset in the correct format
    hate_dataset_dict = convert_train_and_validation_to_dataset(train_df, validation_df)
    hate_dataset_dict["test"] = test_dataset

    model_name="unhcr/hatespeech-detection"     # "IMSyPP/hate_speech_en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = hate_dataset_dict.map(tokenize_function, batched=True)
    training_args = TrainingArguments(output_dir="test_trainer")
    metric = evaluate.load("accuracy")

    training_args = TrainingArguments(output_dir="test_trainer", 
                                  evaluation_strategy="epoch", num_train_epochs=10)

    print("Dataset size", len(train_df))
    small_train_dataset = tokenized_datasets["train"]   # .shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_datasets["validation"]   # .shuffle(seed=42).select(range(10))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("ended")

    with open(f"./model_checkpoint_{model_name.replace('/', '_')}.pickle", 'wb') as f:
        pickle.dump(model, f)
    print("Saved model !!!")
    
    evaluation_results = trainer.evaluate()
    print("Evaluation results:", evaluation_results)
    test_results = trainer.predict(hate_dataset_dict["test"].map(tokenize_function, batched=True))
    print("Test results", test_results)

# if "__name__" == "__main__":
main_pp()
