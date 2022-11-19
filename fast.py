from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
#from scipy.spatial.distance import cosine
#import json
#from datetime import datetime
import os
from typing import List
#from functools import wraps, update_wrapper
#import re
#from py_thesaurus import Thesaurus
#from thesaurus import Word
#import nltk
# nltk.download('stopwords')
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score 
import sys
import pickle
from scipy.stats import entropy
import spacy

app = FastAPI()
"""class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float"""
lookup = None

nlp = spacy.load("en_core_web_sm")
def load_phonetic_embedding():
    global lookup
    # read phonetic embedding pickle file
    path = ""
    with open('phonetic_embd.pickle', 'rb') as handle:
        lookup = pickle.load(handle)
    print("Phonetic embedding loaded !")
    return "success"


class ItemList(BaseModel):
    easy: List[str]
    diff: List[str]
@app.post('/learn')
async def learn(inputdata: ItemList):
    load_phonetic_embedding()
    easy = inputdata.easy
    diff = inputdata.diff

    
    X, y = [], []
    for w in easy:
        word = w.upper()
        if word in lookup:
            X.append(lookup[word])
            y.append(0)

    for w in diff:
        word = w.upper()
        if word in lookup:
            X.append(lookup[word])
            y.append(1)

    clf = svm.SVC(probability=True, random_state=0)
    clf.fit(X, y)
    pickle.dump(clf, open('clf.pkl', 'wb'))
    y_pred = clf.predict(X)

    return {
    'accuracy': accuracy_score(y,y_pred)*100

    }
def uncertainity_sampling():
    clf = pickle.load(open('clf.pkl', 'rb'))
    X = list(lookup.values())
    prob = clf.predict_proba(X)
    ent = entropy(prob.T)
    # sort in descending order so minus sign
    sorted_ind = (-ent).argsort()
    return sorted_ind

@app.post('/explicit_learn')
def explicit_learn(inputdata: ItemList):
    load_phonetic_embedding()
    easy_words = inputdata.easy
    diff_words = inputdata.diff
    label_words = easy_words + diff_words
    #print("Label Words: ", label_words)
    all_words = list(lookup.keys())
    sorted_ind = uncertainity_sampling()
    #print("Sorted ind: ", sorted_ind[:10])
    for i in sorted_ind:
        word = all_words[i]
        if word not in label_words:
            break
    next_word = all_words[i]
    #print("Next word:  ", next_word)
    return {'Next_word' :next_word }

class Words(BaseModel):
    words: List[str]
@app.post('/get_hard_text')
def get_hard_text(data: Words):
    text_words = data.words
    thresh = 0.7
    res = []
    word_list = []
    for w in  text_words:
        w = w.upper()
        if w not in lookup:
            continue
        vec = lookup[w]
        p = round(clf.predict_proba([vec])[0][1], 2)
        #print("word: ", w, "  p val: ",p)
        if p >= thresh and w not in word_list:
            res.append((w, p))
            word_list.append(w)
    #print("Hard Words:  ", res)
    return {'Result' :res}

'''@app.post('/check_if_word_difficult')
def check_if_word_difficult():
    
    synonyms = request.args.getlist("synonyms[]")
    thresh = float(request.args.get("thresh"))/100

    print("synonyms:  ", synonyms)
    print("threshold:  ", thresh)

    res = []
    for w in synonyms:
        w = w.upper()
        if w not in lookup:
            continue
        vec = lookup[w]
        p = round(clf.predict_proba([vec])[0][1], 2)
        print("word: ", w, "  p val: ",p)
        if p <= thresh:
            print(w,p)
            res.append((w, p))
    print("check_if_word_difficult res:  ", res)
    return jsonify(res)   '''

