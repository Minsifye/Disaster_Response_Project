# Disaster Response Pipeline Project 

To analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
<p align="center">
  <img src="https://github.com/Minsifye/Disaster_Response_Project/blob/master/title.png?raw=true" width="750" title="title">
</p>


### A Udacity Data Scientist Nanodegree Project


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Execution](#execution)
4. [File Descriptions](#files)
5. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing) 


## Installation <a name="installation"></a>

- Python 3.5+ (I used Python 3.6)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

### Execution <a name="execution"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Project Motivation<a name="motivation"></a>

There are three components for this project.
1. ETL Pipeline

- In a Python script, process_data.py, write a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline

- In a Python script, train_classifier.py, write a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App

- Adding data visualizations using Plotly in the web app.

<p align="center">
  <img src="https://github.com/Minsifye/Disaster_Response_Project/blob/master/weather.png?raw=true" width="750" title="weather">
  <img src="https://github.com/Minsifye/Disaster_Response_Project/blob/master/people.png?raw=true" width="750" title="people">
</p>


## File Descriptions <a name="files"></a>

- [ETL Pipeline Preparation.ipynb](https://github.com/Minsifye/Disaster_Response_Project/blob/master/ETL%20Pipeline%20Preparation.ipynb) : This notebook will help you to go through initial thought process behind ETL part of this project and how different procedure were tried and tested. The final outcome of this notebook is used to create process_data.py file.
- [ML Pipeline Preparation.ipynb](https://github.com/Minsifye/Disaster_Response_Project/blob/master/ML%20Pipeline%20Preparation.ipynb) : This jupyter notebook will help to understand, why and how a particular machine learning algorithm were choosen. This jupyter notebook represent initial thought process for creating a model pipeline. The final outcome of this notebook is used to create train_classifier.py file.

<i>Markdown cells were used to assist in walking through the thought process for individual steps. </i> 


## Results<a name="results"></a>
- The outcome of this project is to start from scratch with a dataset, create a ETL pipeline for data engineering job and create a Machine Learning pipeline to train a model which can read text data and predict 36 classification categories.
- At the end, use that trained and tuned ML model and use to predict any new message and find which disaster category it will fit.
- Create a front-end application using flask to showcase visualization and model disaster category prediction on a webpage.
<p align="center">
  <img src="https://github.com/Minsifye/Disaster_Response_Project/blob/master/result.png?raw=true" width="750" title="result">
</p>


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure 8 for the data.  You can find the Licensing for the data and other descriptive information at the this link available [here](https://www.figure-eight.com/dataset/combined-disaster-response-data/).  Otherwise, feel free to use the code here as you would like! 

