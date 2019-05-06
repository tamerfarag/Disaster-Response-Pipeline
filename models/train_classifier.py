import sys
import keep_awake as ka

import pandas as pd
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    df.head()
    X = df['message']
    category_names = df.columns[4:]
    Y = df[category_names].values
    return X, Y, category_names

def tokenize(text):
    words = word_tokenize(text)
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    with ka.active_session():
        model = GridSearchCV(pipeline, param_grid=parameters)
        
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    print("Classification Report")
    for i in range(len(category_names)):
        print("Category: ", category_names[i])
        print(classification_report(Y_test[:, i], y_pred[:, i]))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()