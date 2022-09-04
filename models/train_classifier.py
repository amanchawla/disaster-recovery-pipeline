import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql('SELECT * FROM Messages', conn)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    # Make text in lower case and remove any special character
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())

    # tokenise text
    words = word_tokenize(text)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__min_samples_split': [2, 10]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function creates dataframe with all evaluation matrics for each output category
    """

    true_array = np.array(Y_test)
    pred_array = model.predict(X_test)

    matrics = []

    for i in range(len(category_names)):
        accuracy = accuracy_score(true_array[:, i], pred_array[:, i])
        precision = precision_score(true_array[:, i], pred_array[:, i], average='micro')
        recall = recall_score(true_array[:, i], pred_array[:, i], average='micro')
        f1_scor = f1_score(true_array[:, i], pred_array[:, i], average='micro')

        matrics.append([accuracy, precision, recall, f1_scor])

    matrics_arr = np.array(matrics)

    matrics_df = pd.DataFrame(matrics_arr, index=category_names,
                              columns=['Accuracy', 'Precision', 'Recall', 'F1_Score'])

    return matrics_df


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()