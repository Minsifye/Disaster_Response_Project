# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    load data from ETL pipeline
    Args:
    input_file: first argument passed to script
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'genre'])
    return X, y

def evaluate_model(model, X_test, y_test):
    """
    evaluation precision, recall, accuracy, f1_score for each category and whole model
    Args:
    model: tuned model
    X_test: X test set
    y_test: y test set
    """
    multi_predictions = pd.DataFrame(model.predict(X_test))
    multi_predictions.columns = y_test.columns.copy()

    eval_list = []
    for column in multi_predictions:
        # set each value to be the last character of the string
        #confusion_mat = confusion_matrix(y_test[column], multi_predictions[column])
        report = classification_report(y_test[column],multi_predictions[column])
        accuracy = accuracy_score(y_test[column],multi_predictions[column])
        precision = precision_score(y_test[column],multi_predictions[column], average='weighted')
        recall = recall_score(y_test[column],multi_predictions[column], average='weighted')
        f1 = f1_score(y_test[column],multi_predictions[column], average='weighted')
        print("Label:", column)
        print(report)
        eval_list.append([precision, recall, accuracy, f1])
        #precision_list.append(precision)
        #recall_list.append(recall)
        #f1_list.append(f1)

        print("-----------------------------------------------------------------------")

    evaluation = pd.DataFrame(eval_list)
    evaluation.columns = ['precision','recall','accuracy','f1_score']
    #evaluation['recall'] = recall_list
    #evaluation['accuracy'] = accuracy_list
    #evaluation['f1_score'] = f1_list
    print(evaluation)
    print("*******Overall Evaluation*******\nPrecision:{:.2f}\tRecall:{:.2f}\nAccuracy:{:.2f}\tF1 Score:{:.2f}".format(
        np.mean(evaluation.precision), np.mean(evaluation.recall),
        np.mean(evaluation.accuracy), np.mean(evaluation.f1_score)))

    return evaluation



def tokenize(text):
    """
    a tokenization function to process your text data
    Args:
    text: text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # remove stop words
    STOPWORDS = stopwords.words("english")
    tokens = [word for word in tokens if word not in STOPWORDS]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def main():
    """
    main function: Receive two attributes at run time
    Args
    Arg1: Input DB file name along path example- /home/filename.db
    Arg2: Output pickle file name along with path example- /home/model.sav
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
        pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                           ('tfidf', TfidfTransformer()),
                           ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
        print('Building model...')
        parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                'vect__max_df': (0.75, 1.0),
                'tfidf__use_idf': (True, False)
                }

        cross_validation = GridSearchCV(pipeline, param_grid=parameters, verbose = 3, n_jobs=-1)


        # fit model
        print('Training model...')
        model_tuned = cross_validation.fit(X_train, y_train)
        # output model test results
        print('Evaluating model...')
        the_evaluation = evaluate_model(model_tuned, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #model_tuned = train(X, y, model)  # train model pipeline
        # Export model as a pickle file
        pickle.dump(model_tuned, open(model_filepath, 'wb')) # save model
        print('Trained model saved!')


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')



if __name__ == '__main__':
    main()
