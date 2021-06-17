from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
import pickle

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def makecalc():
    data = request.get_json()
    tmp=tf1.transform([data['text']])
    prediction=np.array2string(Encoder.inverse_transform(model.predict(tmp)))
    return jsonify(prediction)

if __name__ == '__main__':
    filename = 'SVM.sav'
    model = p.load(open(filename, 'rb'))


#Set Random seed
np.random.seed(500)

tf1 = pickle.load(open("tfidf_SVM.pkl", 'rb'))

from sklearn.model_selection import train_test_split
Encoder = LabelEncoder()
df = pd.read_csv('bbc-text.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'],test_size=0.3, random_state = 0)
Train_Y = Encoder.fit_transform(y_train)

app.run(debug=True, port=5555)