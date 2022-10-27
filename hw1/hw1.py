# Brooke Czerwinski
# Homework 1
# Natural Language Processing - CS 410

# References:
# https://scikit-learn.org/stable/auto_examples/model_selection/gridSearch_text_feature_extraction.html
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html



import numpy as np
import pandas as pd
import itertools
import time
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from nltk.tokenize import word_tokenize

def parseJSON(fname, start_idx, stop_idx):
    with open(fname, 'r') as text_file:
        for line in itertools.islice(text_file, start_idx, stop_idx):
            yield eval(line)     
        

def countLines(fname):
    count = 0
    for line in open(fname, 'r'):
        count += 1
    return count
    


def main():
    # Location of data
    cwd = os.getcwd()
    data_file = cwd + '/News-Headlines-Dataset-For-Sarcasm-Detection/Sarcasm_Headlines_Dataset.json'


    # Split the dataset for training, testing, and validation (8:1:1 split, respectively)
    data_size = countLines(data_file)
    cutoff_1 = int(.80 * data_size)
    cutoff_2 = int(.90 * data_size)
    raw_training_data = parseJSON(data_file, 0, cutoff_1)
    raw_test_data = parseJSON(data_file, cutoff_1, cutoff_2)
    raw_validation_data = parseJSON(data_file, cutoff_2, data_size)

    # Preprocess text for pipeline using panda
    training_data = pd.DataFrame.from_dict(raw_training_data, orient='columns', dtype=None, columns=None)
    test_data = pd.DataFrame.from_dict(raw_test_data, orient='columns', dtype=None, columns=None)

    # Set up a pipeline object for the Naive Bayes baseline model
    baseline_NB = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    # Set up a pipeline object for the Logistic Regression baseline model
    baseline_LR = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])
    
    # Train the baselines
    baseline_NB.fit(training_data.headline, training_data.is_sarcastic)
    baseline_LR.fit(training_data.headline, training_data.is_sarcastic)

    # Get predictions with the set of test headlines
    test_headlines = test_data.headline
    baseline_prediction_NB = baseline_NB.predict(test_headlines)
    baseline_prediction_LR = baseline_LR.predict(test_headlines)

    # Get accuracies for the baselines
    baseline_accuracy_NB = np.mean(baseline_prediction_NB == test_data.is_sarcastic)
    baseline_accuracy_LR = np.mean(baseline_prediction_LR == test_data.is_sarcastic)
    print("Baseline accuracy:")
    print('Naive Bayes: ', baseline_accuracy_NB)
    print('Logistic Regression: ', baseline_accuracy_LR)

    # Grid search for the most accurate features
    # Set parameters for grid search
    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                              (2, 2), (2, 3), (2, 4), (2, 5),
                              (3, 3), (3, 4), (3, 5),
                              (4, 4), (4, 5),
                              (5, 5)
                             ),
        # "vect__max_df": (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        #"clf__alpha": (0.00001, 0.000001),
        #'clf__max_iter': (10, 50, 80),
    }

    print('\n\nPerforming Gridsearch to find best n-gram parameters for baseline models')
    # Create Grid Searches to find the best set of the given parameters
    grid_search_NB = GridSearchCV(baseline_NB, parameters, n_jobs=-1, verbose=1, cv=10)
    grid_search_NB.fit(training_data.headline, training_data.is_sarcastic)
    grid_search_LR = GridSearchCV(baseline_LR, parameters, n_jobs=-1, verbose=1, cv=10)
    grid_search_LR.fit(training_data.headline, training_data.is_sarcastic)

    # Report the results of the grid search:
    print("\n\nBest parameters for baselines:")
    best_parameters_NB = grid_search_NB.best_estimator_.get_params()
    print("Naive Bayes:")
    for parameter_name in sorted(parameters.keys()):
        print("\t%s: %r" % (parameter_name, best_parameters_NB[parameter_name]))
    best_parameters_LR = grid_search_LR.best_estimator_.get_params()
    print("Logistic Regression:")
    for parameter_name in sorted(parameters.keys()):
        print("\t%s: %r" % (parameter_name, best_parameters_LR[parameter_name]))

    ########################## 
    # Models that include punctuation

    # Set up a pipeline object for the Naive Bayes baseline model
    punctuation_NB = Pipeline([
        ('vect', CountVectorizer(tokenizer=word_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    # Set up a pipeline object for the Logistic Regression baseline model
    punctuation_LR = Pipeline([
        ('vect', CountVectorizer(tokenizer=word_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    # Train the punctuateds models
    punctuation_NB.fit(training_data.headline, training_data.is_sarcastic)
    punctuation_LR.fit(training_data.headline, training_data.is_sarcastic)

    # Get predictions with the set of test headlines
    test_headlines = test_data.headline
    punctuation_prediction_NB = baseline_NB.predict(test_headlines)
    punctuation_prediction_LR = baseline_LR.predict(test_headlines)

    # Get accuracies for the baselines
    punctuation_accuracy_NB = np.mean(punctuation_prediction_NB == test_data.is_sarcastic)
    punctuation_accuracy_LR = np.mean(punctuation_prediction_LR == test_data.is_sarcastic)
    print("Punctuation-included accuracy:")
    print('Naive Bayes: ', punctuation_accuracy_NB)
    print('Logistic Regression: ', punctuation_accuracy_LR)

    # Create Grid Searches for models with punctuation
    print('\n\nPerforming Gridsearch to find best n-gram parameters for punctuation-included models')
    grid_search_NB = GridSearchCV(punctuation_NB, parameters, n_jobs=-1, verbose=1, cv=10)
    grid_search_NB.fit(training_data.headline, training_data.is_sarcastic)
    grid_search_LR = GridSearchCV(punctuation_LR, parameters, n_jobs=-1, verbose=1, cv=10)
    grid_search_LR.fit(training_data.headline, training_data.is_sarcastic)

    # Report the results of the grid search:
    print("\n\nBest parameters for punctuation-included:")
    best_parameters_NB = grid_search_NB.best_estimator_.get_params()
    print("Naive Bayes:")
    for parameter_name in sorted(parameters.keys()):
        print("\t%s: %r" % (parameter_name, best_parameters_NB[parameter_name]))
    best_parameters_LR = grid_search_LR.best_estimator_.get_params()
    print("Logistic Regression:")
    for parameter_name in sorted(parameters.keys()):
        print("\t%s: %r" % (parameter_name, best_parameters_LR[parameter_name]))




    # Metrics for all models created
    print('\n\nNaive Bayes (baseline)')
    print(metrics.classification_report(test_data.is_sarcastic, baseline_prediction_NB))

    print('\n\nLogistic Regression (baseline)')
    print(metrics.classification_report(test_data.is_sarcastic, baseline_prediction_LR))
    
    print('\n\nNaive Bayes (WITH punctuation)')
    print(metrics.classification_report(test_data.is_sarcastic, punctuation_prediction_NB))

    print('\n\nLogistic Regression (WITH punctuation)')
    print(metrics.classification_report(test_data.is_sarcastic, punctuation_prediction_LR))
    

    

if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token