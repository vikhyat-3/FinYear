import streamlit as st
import joblib
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import numpy as np
import matplotlib.pyplot as plt

def cleanData(row):
  row = row.lower() #Convert all text to lowercase
  row = re.sub('[^a-zA-Z]', ' ', row) #Remove punctuation, special characters etc
  token = row.split()
  removeStop= [i for i in token if i not in stopwords] #Remove all stopwords
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in removeStop] #Lemmatize the words
  cleaned_string = ""
  for word in lemm_text:
    cleaned_string+=word
    cleaned_string+= ' '
  return cleaned_string


model = joblib.load('MNB')
st.title("Fake News Detector")
ip = st.text_input("Enter the news: ")
op = model.predict([cleanData(ip)])
final  = ''
if st.button('Predict'):
  if op == 1:
    final = 'Real News'
  else:
    final = 'Fake News'
st.title(final)
