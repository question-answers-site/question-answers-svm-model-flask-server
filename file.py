#!/usr/bin/python
# -*- coding: utf-8 -*-
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
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def makecalc():
    return jsonify(request.json)
    data = request.json
    tmp = tf1.transform([data['text']])
    prediction = \
        np.array2string(Encoder.inverse_transform(model.predict(tmp)))

    # prediction = np.array2string(model.predict(tmp))

    return jsonify(prediction)

if __name__ == '__main__':
    filename = '4kSVM_py.sav'
    model = p.load(open(filename, 'rb'))

    # Set Random seed

np.random.seed(500)

# Set Random seed

np.random.seed(500)

# Testing phase

tf1 = pickle.load(open('tfidflMerged.pkl', 'rb'))

# Create new tfidfVectorizer with old vocabulary
# tf1_new = TfidfVectorizer(max_features = 500000, vocabulary = tf1.vocabulary_)

filename = 'Merged_py.sav'
loaded_model = pickle.load(open(filename, 'rb'))

tmp = \
    tf1.transform(['i am going to eat pittza with my friends in garden '
                  , 'lets go watching a new movie next week'])

from sklearn.model_selection import train_test_split
Encoder = LabelEncoder()
df = pd.read_csv('newFile.csv')
(X_train, X_test, y_train, y_test) = train_test_split(df['text'],
        df['label'], test_size=0.3, random_state=0)
Train_Y = Encoder.fit_transform(y_train)

print Encoder.inverse_transform(loaded_model.predict(tmp))
app.run(debug=True, port=5555)
