import nltk
import streamlit as stl
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
ps = PorterStemmer()


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def transformText(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

stl.title("Email/SMS Spam Classifier")
msgInput = stl.text_input("Please enter the SMS / Email received here.")

if stl.button("Classify"):
    transformedMsgInput = transformText(msgInput)
    vectorInput = tfidf.transform([transformedMsgInput])
    result = model.predict(vectorInput)[0]
    if result == 1:
        stl.header("Spam")
    else:
        stl.header("Not Spam")
