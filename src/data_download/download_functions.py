import requests


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



if __name__ == "__main__":
    download_hatespeech_version_2()