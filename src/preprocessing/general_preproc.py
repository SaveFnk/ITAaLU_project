import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess_text(text_series):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    tokens = text_series.apply(word_tokenize)
    tokens = tokens.apply(lambda x: [word.lower().strip() for word in x])

    stop_words = set(stopwords.words("english"))
    tokens = tokens.apply(lambda x: [word for word in x if word.lower() not in stop_words])

    tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(word, pos="v") for word in x])
    tokens = tokens.apply(lambda x: [stemmer.stem(word) for word in x])
    return tokens.apply(lambda x: " ".join(x))