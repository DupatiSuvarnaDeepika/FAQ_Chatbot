import os
import re
import string
import pandas as pd
import nltk

# Create local NLTK data folder
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Tell NLTK where to look
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Download required datasets if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
data = pd.read_csv("faq.csv")

questions = data["Question"]
answers = data["Answer"]

stop_words = set(stopwords.words("english"))

# Common words to ignore
remove_words = [
    "what", "is", "are", "define", "explain",
    "tell", "me", "about", "can", "you",
    "please", "how"
]

# Expand abbreviations
def normalize(text):
    text = text.lower()

    replacements = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision"
    }

    for short, full in replacements.items():
        text = re.sub(r"\b" + short + r"\b", full, text)

    return text


# Clean text
def preprocess(text):
    text = normalize(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words and word not in remove_words
    ]

    return " ".join(words)


# Preprocess FAQ questions
clean_questions = [preprocess(q) for q in questions]

# TF-IDF
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(clean_questions)


def get_answer(user_question):

    original_question = user_question.lower()

    keywords = [
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "dl",
        "nlp",
        "natural language processing",
        "chatbot",
        "computer vision",
        "cv"
    ]

    if not any(word in original_question for word in keywords):
        return "Sorry, I don't know the answer. Please ask an AI-related question."

    user_question = preprocess(user_question)

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    score = similarity[0][index]

    if score >= 0.60:
        return answers[index]

    return "Sorry, I don't know the answer. Please ask an AI-related question."