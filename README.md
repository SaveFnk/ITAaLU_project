# ITAaLU_project
Project for a university course


## Data sources used

1. https://github.com/avaapm/hatespeech
2. https://huggingface.co/datasets/skg/toxigen-data


Back up dataset: 
https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech/data


## Loading model from hugging face

Some models have gated access which means you have to authenticate first. Do this do the following:
1. Install the `transformers` library with `pip install transformers`
2. Log in through `huggingface-cli login`, or add this to a notebook:
```python
from huggingface_hub import login
login()
```
3. Pytorch is required as well:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```