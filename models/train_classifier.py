

"""
python scripts to import disaster_messages and categories stored in SQLite database, 
build a pipeline and train, test and output a classifier as a pikkle file and

script execution syntax: 
> python train_classifier.py <path to sqlite database> <path to output pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disasterResponse.db classifier.pkl

"""

# Import necessary libraries

import numpy as np
import pandas as pd

import sys
import os
import re
import string

from sqlalchemy import create_engine
import pickle


import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin




def load_data(database_filepath):
    """
    Load the dataset from the sqlite database 

    Arguments:
        database_filepath (string): path to the SQLite source database
    
    Returns:
        X (pandas dataframe): a dataframe containing features 
        Y (pandas dataframe): a dataframe containing labels
        category_names(list): a list of category names

    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'disaster_response_tbl'
    df = pd.read_sql_table(table_name, engine)

    
    X = df['message']
    y = df.iloc[:, 4:]

    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Function to tokenize text

    Arguments:
        text(string): messages extracted from the loaded database
    Returns:
        clean_tokens(list): list of tokenized, cleaned text. 
    """


    # define url_regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    
    # iterate thorugh each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white spaces
        clean_tok = lemmatizer.lemmatize((tok.lower().strip()))
        clean_tokens.append(clean_tok)
    
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting verb extractor class 
    
    Class to extract the starting verb of a sentence into a feature. 
    Adapted from course notes
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Function to build ML model using pipelines, define and run a gridsearch to obtain optimal model parameters

    Retuns:
        an ML pipeline object. 

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier(learning_rate = 1.0, n_estimators=75 ))) #parameters determind from gridsearch study in ML Pipeline Preparation notebook
    ])

    #Grid search to determind optimal model parameters.
    ##parameters = {'classifier__estimator__learning_rate': [0.05, 1, 1.5], 'classifier__estimator__n_estimators': [50, 75, 100]}
    # create grid search object
    ##cv = GridSearchCV(pipeline, param_grid=parameters, scoring = 'f1_macro') #, n_jobs = -1)    

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates a model's performance and prints out the classification_report on all the feature prediction.

    Arguments: 
        model:  fully trained ML model
        X_test: Test features
        Y_test: Test Labels
    
    returns: N/A. prints out classification report.

    """

    preds = model.predict(X_test)
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], preds[:,idx]))


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Arguments:
        model (pipeline or gridsearch object): trained model
        model_filepath (string): destination to write model file to


    Returns: N/A. saves model to disk
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()

def main():
    """
    Classifeir development, training and storage function.
    Calls varios functions defined above to achieve the purpose of this function.   

    """
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()