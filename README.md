
# Disaster Recovery Response

During disaster situation, twitter gets flooded with messages. Response team need to go through these message quickly so that they can forward the message to concerned team.

Response team get flooded with such messaged when they have least capacity to handle such load.

With this objective, here I have created an application which can allow team to understand which response team should act. 

Here is I have used text mining and machine learning modelling to undetstand the theme of message and creating flags to tell which team should act.


## Installation

Install my-project with npm

```bash
import json
import plotly
import pandas as pd

# For Web Development
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# For Text Mining NLTK library has been used
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For Machine Learning Development
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer



```
    
## How Model has been developed?

### How messages are cleaned?
- All Words in messages have been tokenised
- In Tokenisation, each word is standardised to its root word so that different version of same word are treated equally
- NLTK English Stopword is also to remove some common English words which have no significance in identification of themes

### What features have been created?
- Scikit Learn - CountVectorizer - used to  matrix of token counts
- Scikit Learn - TfidfTransformer - Transform a count matrix to a normalized tf or tf-idf representation

### What machine learning technique used?
- RandomForestClassification Technique used.
    



## Instructions

### To create a processed sqlite db
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
### To train and save a pkl model
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
### To deploy the application locally
python run.py



# Structure of Code


* app
  * template
    * master.html # main page of web app
    * go.html # classification result page of web app
  * run.py # Flask file that runs app 
* data
  * disaster_categories.csv # data to process
  * disaster_messages.csv # data to process
  * process_data.py
  * DisasterResponse.db # database to save clean data to
* models
  * train_classifier.py
  * classifier.pkl # saved model
* README.md