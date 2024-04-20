from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from src.training.training_func import prepare_dataset_dict, compute_metrics, save_model


def train_model_and_predict(model_name):
    start_time = datetime.now()
    print(f"Training model {model_name}!\n Training started at {start_time}")

    # prepare the dataset in the correct format
    hate_dataset_dict = prepare_dataset_dict(large=True)
    model_name = "unhcr/hatespeech-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "IMSyPP/hate_speech_en":
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=3,
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # def tokenize_function(examples):
    #    return tokenizer(examples["text"], padding="max_length", truncation=True)

    # tokenized_datasets = hate_dataset_dict.map(tokenize_function, batched=True)
    tokenized_datasets = hate_dataset_dict.map(lambda examples: tokenizer(examples["text"], batched=False))
    training_args = TrainingArguments(output_dir="test_trainer",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=10
                                      )

    small_train_dataset = tokenized_datasets["train"]  # .shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_datasets["validation"]  # .shuffle(seed=42).select(range(10))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(f"Ended training in {datetime.now() - start_time} seconds!")
    save_model(model, model_name)

    evaluation_results = trainer.evaluate()
    print("Evaluation results:\n", evaluation_results, "\n")

    # hate_dataset_dict["test"].shuffle(seed=42).select(range(10))
    test_results = trainer.predict(hate_dataset_dict["test"])
    print("Test results\n", test_results)


# "IMSyPP/hate_speech_en"
train_model_and_predict("unhcr/hatespeech-detection")
