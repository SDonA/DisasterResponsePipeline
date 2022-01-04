"""
Entry point for app.
Contains:
    webapp routes, data, model import and charting scripts. 

App execution Syntax:
    > python run.py
"""


import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#import plotly.express as px
#from sklearn.externals import joblib
import sklearn.externals
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

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



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_tbl', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Data for 1st graph - provided in template code
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #data for second graph - my custom graph 1
    # creates a sorted bar graph
    cat_names = df.iloc[:,4:].columns
    sum_by_cat = (df.iloc[:,4:] != 0).sum().values
    dict_sum_by_cat = dict(zip(cat_names, sum_by_cat))
    sort_sum_by_cat = sorted(dict_sum_by_cat.values())
    sorted_dict_sum_by_cat = {}
    for i in sort_sum_by_cat:
        for k in dict_sum_by_cat.keys():
            if dict_sum_by_cat[k] == i:
                sorted_dict_sum_by_cat[k] = dict_sum_by_cat[k]
                break



    #data for 3rd graph - my custom graph 2
    # creates a sorted bar graph
    sum_of_sum_by_cat = np.sum(sum_by_cat)
    sum_by_cat_pct = (sum_by_cat / sum_of_sum_by_cat) * 100
    dict_sum_by_cat_pct = dict(zip(cat_names, sum_by_cat_pct))

    sorted_values = sorted(dict_sum_by_cat_pct.values()) # sort the values
    sorted_dict_sum_by_cat_pct = {}
    for i in sorted_values:
        for k in dict_sum_by_cat_pct.keys():
            if dict_sum_by_cat_pct[k] == i:
                sorted_dict_sum_by_cat_pct[k] = dict_sum_by_cat_pct[k]
                break


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # My custome graph 1
        {
            'data': [
                Bar(
                    x = list(sorted_dict_sum_by_cat.values()),
                    y = list(sorted_dict_sum_by_cat.keys()),
                    orientation = 'h'
                    #x = cat_names,
                    #y = sum_by_cat
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories (Counts)',
                'height': 800,
                'margin': dict(l=200, r=50, b=100, t=100, pad=4), 
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count",
#                    'tickangle': 35
                }
            }
        },

        # My custome graph 2
        {
            'data': [
                Bar(
                    x = list(sorted_dict_sum_by_cat_pct.values()),
                    y = list(sorted_dict_sum_by_cat_pct.keys()),
                    orientation = 'h'                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories (Percentages)',
                'height': 800,
                'margin': dict(l=200, r=50, b=100, t=100, pad=4), 
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage Count"                    
                }
            }
        }       


    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()