import json
import os
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Box
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine("sqlite:///{}".format(os.path.join(BASE_DIR, 'data', 'DisasterResponse.db')))
df = pd.read_sql_table('fact_messages', engine)

# load model
model = joblib.load(os.path.join(BASE_DIR, 'models', 'classifier.pkl'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    # Graph 1 Data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Graph 2 Data
    df_ = pd.melt(df.drop(['message','original','genre'], axis=1),['id'])

    categories_sum = df_.groupby('variable')['value'].sum().sort_values(ascending=False)
    categories_names = list(categories_sum.index)

    # Graph 3 Data
    msg_length = df

    msg_length['msg_length'] = msg_length['message'].str.len()

    msg_length = df[['genre','msg_length']]

    max_direct = msg_length[msg_length['genre']=='direct']['msg_length'].max()
    msg_length = msg_length[msg_length['msg_length'] <= max_direct]

    # create visuals

    graphs = [
        # Graph 1
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        # Graph 2
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_sum
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        # Graph 3
        {
            'data': [
                Box(
                    x=msg_length['genre'],
                    y=msg_length['msg_length']
                )
            ],

            'layout': {
                'title': 'Message Length by Genre - Limited by Max Length of Direct Messages'
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
    port = int(os.environ.get('PORT', 3001))
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
