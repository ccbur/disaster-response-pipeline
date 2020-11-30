# Disaster Response Pipeline Project

## Motivation


This project analyzes data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events and are processed by a machine learning pipeline to categorize these events.

The project includes a web app to test the model and display visualizations of the data.

## Warning

I implemented this project as part of the Udacity nanodegree in Datascience and should be considered preliminary. I'm still learning!

## Quick start
### Requirements
* Python >=3.5

### Packages
`pip install pandas sklearn nltk sqlalchemy flask plotly joblib pickle nltk regex`

### Repository
`git clone https://github.com/ccbur/disaster-response-pipeline`

### Run

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/


## Results
- 

## Files
File | Description
------------ | -------------
*README.md* | this README
*LICENSE* | [GNU General Public License v3.0](https://github.com/ccbur/disaster-response-pipeline/LICENSE)
*app/process_data.py* | Read disaster messages from a CSV file, clean data and write to SQLite db file
*models/train_classifier.py* | Use disaster messages from SQLite db file to train a classifier model
*app/run.py* | Web app to test and visualize classifier model 
*app/templates/...* | HTML templates for web app

## License
The code is licensed under the [GNU General Public License v3.0](https://github.com/ccbur/disaster-response-pipeline/LICENSE).

## Acknowledgements
Thanks to Figure Eight for the provided data.




