import sys
import pandas as pd

import pickle
from sqlalchemy import create_engine
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
email_regex = '(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'
stop_words = stopwords.words("english")

def tokenize(text):
    #print("tokenize: {}".format(text))
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        #print('url: {}'.format(url))

    detected_emails = re.findall(email_regex, text)
    for email in detected_emails:
        text = text.replace(email, "emailplaceholder")
        #print('email: {}'.format(email))

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)
#    print("Tokens: {}".format(clean_tokens))
    return clean_tokens

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    print(pd.read_sql('SELECT name FROM sqlite_master WHERE type =\'table\'', engine))

    df = pd.read_sql('SELECT * FROM Message', engine)
    X = df[['message', 'genre']]
    #X = df['message'].values
    Y_df = df.drop(columns = ['id', 'message', 'original', 'genre'])
    for col in Y_df:
        print("Features: {}", col)
        print(Y_df[col].unique())
    Y = Y_df.values
    return X, Y, Y_df.columns


#class StartingVerbExtractor(BaseEstimator, TransformerMixin):
#    def starting_verb(self, text):
#        sentence_list = nltk.sent_tokenize(text)
#        for sentence in sentence_list:
#            pos_tags = nltk.pos_tag(tokenize(sentence))
#            first_word, first_tag = pos_tags[0]
#            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
#                return True
#        return False
#
#    def fit(self, x, y=None):
#        return self
#
#    def transform(self, X):
#        X_tagged = pd.Series(X).apply(self.starting_verb)
#        return pd.DataFrame(X_tagged)

def build_model():

    # Classifier Testing (Classifier: TestSize / Accuracy)
    # 
    # MultinomialNB:                0.9 / 0.931
    # RandomForestClassifier:       0.9 / 0.939
    # GradientBoostingClassifier:   0.9 / 0.939
    # LinearSVC:                    0.9 / 0.943
    # LinearSVC:                    0.2 / 0.949
    pipeline = Pipeline([
        ('ctf', ColumnTransformer(transformers=[
            ('message_feat', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
#                ('starting_verb', StartingVerbExtractor())
            ]), 'message'),
            ('genre_cat', OneHotEncoder(dtype='int'), ['genre']),
        ])),
#        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 314159)))
        ('clf', MultiOutputClassifier(LinearSVC(random_state = 314159)))
#        ('clf', MultiOutputClassifier(GradientBoostingClassifier(random_state = 314159)))
    ])



    # test_size: 0.95
    #parameters = {
    #    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)), # Best: (1, 2), Default: (1, 1)
    #    'features__text_pipeline__vect__max_df': (0.5, 1.0), # Best: 0.5, Default: 1.0
    #    'features__text_pipeline__vect__max_features': (None, 5000), # Best: None, Default: None
    #    'features__text_pipeline__tfidf__use_idf': (True, False), # Best: True, Default: Treu
    #    'clf__estimator__n_estimators': [50, 100], # Best: 50, Default: 100
    #    'clf__estimator__min_samples_split': [2, 4], # Best: 2, Default: 2
    #    'features__transformer_weights': (
    #        {'text_pipeline': 1, 'starting_verb': 0.0}, # Best
    #        {'text_pipeline': 1, 'starting_verb': 0.5},
    #        {'text_pipeline': 0.8, 'starting_verb': 1},
    #    )
    #}

    # test_size: 0.95
    #parameters = {
    #    'features__text_pipeline__vect__ngram_range': ((1, 2), ),
    #    'features__text_pipeline__vect__max_df': (0.5, ), 
    #    'features__text_pipeline__vect__max_features': (None, ),
    #    'features__text_pipeline__tfidf__use_idf': (True, ),
    #    'clf__estimator__n_estimators': (10, 30, 50, 70), # Best: 30, Default: 100
    #    'clf__estimator__min_samples_split': (2, )
    #}

    parameters_full = {
        'ctf__message_feat__text_pipeline__vect__max_df': (0.5, 1.0), # Best: 0.5, Default: 1.0
        'ctf__message_feat__text_pipeline__vect__ngram_range': ((1, 2), (1, 1)), # Best: (1, 2), Default: (1, 1)
        'ctf__message_feat__text_pipeline__vect__max_features': (None, 5000), # Best: None, Default: None
        'ctf__message_feat__text_pipeline__tfidf__use_idf': (True, False), # Best: True, Default: True
        'clf__estimator__n_estimators': (30, 50, 100), # Best: 50, Default: 100
        'clf__estimator__min_samples_split': (2, 4), # Best: 2, Default: 2
        'ctf__transformer_weights': (
            {'message_feat': 1.0, 'genre_cat': 0.0}, # Best -> genre_cat not needed
            {'message_feat': 1.0, 'genre_cat': 0.5},
            {'message_feat': 1.0, 'genre_cat': 1.0},
            {'message_feat': 0.5, 'genre_cat': 1.0},
            {'message_feat': 0.0, 'genre_cat': 1.0},
        ),
        'ctf__message_feat__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.0}, # Best -> starting_verb not needed
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    parameters = {
        'ctf__message_feat__text_pipeline__vect__max_df': (0.5, ), # Best: 0.5, Default: 1.0
        'ctf__message_feat__text_pipeline__vect__ngram_range': ((1, 2), ), # Best: (1, 2), Default: (1, 1)
        'ctf__message_feat__text_pipeline__vect__max_features': (None, ), # Best: None, Default: None
        'ctf__message_feat__text_pipeline__tfidf__use_idf': (True, ), # Best: True, Default: True
#        'clf__estimator__n_estimators': (50, ), # Best: 50, Default: 100
#        'clf__estimator__min_samples_split': (2, ), # Best: 2, Default: 2
        'ctf__transformer_weights': (
            {'message_feat': 1.0, 'genre_cat': 0.0}, # Best
        ),
#        'ctf__message_feat__transformer_weights': (
#            {'text_pipeline': 1, 'starting_verb': 0.0}, # Best
#        )
    }



    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return cv

def column(matrix, i):
    return [row[i] for row in matrix]

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()

    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

    for i in range(len(Y_pred[0])):
        print("Feature: {}".format(category_names[i]))
        Y_pred_f = column(Y_pred, i)
        Y_test_f = column(Y_test, i)
        print(classification_report(Y_test_f, Y_pred_f, zero_division=1))


def save_model(model, model_filepath):
    f = open(model_filepath, 'wb')
    pickle.dump(model, f)
    f.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(category_names)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 314159, shuffle=True) # 0.2

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print(X_train.head())
        model.fit(X_train, Y_train)

        print(model.best_params_)
        
        print('Evaluating model...')
        print(X_test['genre'].unique())
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