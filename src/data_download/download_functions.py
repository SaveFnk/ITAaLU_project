import os
from pathlib import Path

import requests
from datasets import load_dataset_builder
from dotenv import load_dotenv


def download_hatespeech_version_2():
    """
    Download the Hatespeech dataset version 2 from the original source.
    """
    download_path = "data/hatespeech_v2"
    download_urls = ["https://raw.githubusercontent.com/avaapm/hatespeech/master/dataset_v2/hate_speech_dataset_v2.csv",
                     "https://raw.githubusercontent.com/avaapm/hatespeech/master/dataset_v2/hate_speech_dataset_v2_labeler.csv",
                     "https://raw.githubusercontent.com/avaapm/hatespeech/master/dataset_v2/readme.md"]

    for link in download_urls:
        downloaded_data = requests.get(link)
        file_name = link.split("/")[-1]

        with open(f"{download_path}/{file_name}", "wb") as file:
            file.write(downloaded_data.content)


def download_toxify():
    dotenv_path = Path(".env")
    load_dotenv(dotenv_path=dotenv_path)
    # the toke should be generated on hugging face
    token = os.environ.get("HUGGING_FACE_TOKEN")

    # # 250k training examples
    builder = load_dataset_builder("skg/toxigen-data", name="train", token=token, trust_remote_code=True)
    builder.download_and_prepare("./data/toxigen-data-train")

    # Human study
    builder = load_dataset_builder("skg/toxigen-data", name="annotated", token=token, trust_remote_code=True)
    builder.download_and_prepare("./data/toxigen-data-annotated")


if __name__ == "__main__":
    # download_hatespeech_version_2()
    download_toxify()