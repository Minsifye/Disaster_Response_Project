import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    df_related = df[df['related']==1]
    news=df_related[df_related['genre']=='news'][['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']].sum()
    direct=df_related[df_related['genre']=='direct'][['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']].sum()
    social=df_related[df_related['genre']=='social'][['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']].sum()
    weather_related_X=['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']
    news2=df_related[df_related['genre']=='news'][['water', 'food', 'shelter', 'clothing','money', 'missing_people']].sum()
    direct2=df_related[df_related['genre']=='direct'][['water', 'food', 'shelter', 'clothing','money', 'missing_people']].sum()
    social2=df_related[df_related['genre']=='social'][['water', 'food', 'shelter', 'clothing','money', 'missing_people']].sum()
    people_related_X=['water', 'food', 'shelter', 'clothing','money', 'missing_people']





    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=weather_related_X,
                    y=news,
                    name='News'
                ),
                Bar(
                    x=weather_related_X,
                    y=direct,
                    name='Direct'
                ),
                Bar(
                    x=weather_related_X,
                    y=social,
                    name='Social'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related to Weather Conditions',
                'yaxis': {
                    'title': "No. of Tweets"
                },
                'xaxis': {
                    'title': "Weather Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=people_related_X,
                    y=news2,
                    name='News'
                ),
                Bar(
                    x=people_related_X,
                    y=direct2,
                    name='Direct'
                ),
                Bar(
                    x=people_related_X,
                    y=social2,
                    name='Social'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related to People Conditions',
                'yaxis': {
                    'title': "No. of Tweets"
                },
                'xaxis': {
                    'title': "Weather Category"
                }
            }
        },
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
