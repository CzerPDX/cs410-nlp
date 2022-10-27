# Brooke Czerwinski
# Homework 1
# Natural Language Processing - CS 410

# References:
# https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
# https://stackoverflow.com/questions/34714162/preventing-splitting-at-apostrophies-when-tokenizing-words-using-nltk
# https://nlpforhackers.io/training-pos-tagger/


import numpy as np
import pandas as pd
import pprint
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize

# From https://nlpforhackers.io/training-pos-tagger/
def features(headline, is_sarcastic, index):
    # headline: [w1, w2, ...], index: the index of the word
    return {
        'word': headline[index],
        'is_sarcastic': is_sarcastic,
        'is_first': index == 0,
        'is_last': index == len(headline) - 1,
        # Information about capitals is not helpful because our corpus is all lowercase
        # 'is_capitalized': headline[index][0].upper() == headline[index][0],
        # 'is_all_caps': headline[index].upper() == headline[index],
        # 'is_all_lower': headline[index].lower() == headline[index],
        'prefix-1': headline[index][0],
        'prefix-2': headline[index][:2],
        'prefix-3': headline[index][:3],
        'suffix-1': headline[index][-1],
        'suffix-2': headline[index][-2:],
        'suffix-3': headline[index][-3:],
        'prev_word': '' if index == 0 else headline[index - 1],
        'next_word': '' if index == len(headline) - 1 else headline[index + 1],
        'has_hyphen': '-' in headline[index],
        'is_numeric': headline[index].isdigit(),
        'capitals_inside': headline[index][1:].lower() != headline[index][1:]
    }

def transform_headlines_JSON(headlines_JSON):
    X, y = [], []
 
    for item in headlines_JSON:
        headline_tokens = word_tokenize(item['headline'])

        for index in range(len(headline_tokens)):
            # print('headline_token[index] = ', headline_tokens[index])
            # print('item[\'is_sarcastic\'] = ', item['is_sarcastic'])
            X.append(features(headline_tokens, item['is_sarcastic'], index))
            y.append(item['is_sarcastic'])
 
    return X, y

def parseJason(fname):
    for line in open(fname, 'r'):
        yield eval(line)            


def countLines(fname):
    count = 0
    for line in open(fname, 'r'):
        count += 1
    return count


def main():
    # Location of training and test data sets
    training_file = "Sarcasm_training-set.json"
    test_file = "Sarcasm_test-set.json"

    # Get the data as JSON objects
    raw_training_data = parseJason(training_file)
    raw_test_data = parseJason(test_file)


    # pprint.pprint(features(['This', 'is', 'a', 'headline'], 0, 1))
    X, y = transform_headlines_JSON(raw_training_data)
    
    # for item in X:
    #     print(item)

    baseline_NB = Pipeline([
        ('vect', DictVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    baseline_LR = Pipeline([
        ('vect', DictVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Train the model
    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])
    
    
    baseline_NB.fit(X, y)

    print('Training completed')
    
    X_test, y_test = transform_headlines_JSON(raw_test_data)
    
    print("Accuracy:", baseline_NB.score(X_test, y_test))

    for item in X:
        print(item)

    # # Preprocess text for pipeline using panda
    # training_data = pd.DataFrame.from_dict(raw_training_data, orient='columns', dtype=None, columns=None)
    # test_data = pd.DataFrame.from_dict(raw_test_data, orient='columns', dtype=None, columns=None)

    # # Set up a pipeline object for the Naive Bayes baseline model
    # baseline_NB = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB())
    # ])
    # # Set up a pipeline object for the Logistic Regression baseline model
    # baseline_LR = Pipeline([
    #     ('vect', CountVectorizer(analyzer='word')),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', LogisticRegression())
    # ])

    # # Fit a vocabulary to the pipelines/train them
    # baseline_NB.fit(training_data.headline, training_data.is_sarcastic)
    # baseline_LR.fit(training_data.headline, training_data.is_sarcastic)

    
    # temp = baseline_LR.get_feature_names_out()
    # for line in temp:
    #     print(line)

    # # Get predictions with the set of test headlines
    # test_headlines = test_data.headline
    # baseline_prediction_NB = baseline_NB.predict(test_headlines)
    # baseline_prediction_LR = baseline_LR.predict(test_headlines)

    # # Get accuracies for the baselines
    # baseline_accuracy_NB = np.mean(baseline_prediction_NB == test_data.is_sarcastic)
    # baseline_accuracy_LR = np.mean(baseline_prediction_LR == test_data.is_sarcastic)
    # print("Baseline accuracy:")
    # print('Naive Bayes: ', baseline_accuracy_NB)
    # print('Logistic Regression: ', baseline_accuracy_LR)

    # # Grid search for the most accurate features
    # # Set parameters for grid search
    # parameters = {
    #     "vect__ngram_range": ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    #                           (2, 2), (2, 3), (2, 4), (2, 5),
    #                           (3, 3), (3, 4), (3, 5),
    #                           (4, 4), (4, 5),
    #                           (5, 5)
    #                          ),
    #     # "vect__max_df": (0.5, 0.75, 1.0),
    #     # 'vect__max_features': (None, 5000, 10000, 50000),
    #     # 'tfidf__use_idf': (True, False),
    #     # 'tfidf__norm': ('l1', 'l2'),
    #     #"clf__alpha": (0.00001, 0.000001),
    #     #'clf__max_iter': (10, 50, 80),
    # }
    # # Create Grid Searches to find the best set of the given parameters
    # grid_search_NB = GridSearchCV(baseline_NB, parameters, n_jobs=-1, verbose=1)
    # grid_search_NB.fit(training_data.headline, training_data.is_sarcastic)
    # grid_search_LR = GridSearchCV(baseline_LR, parameters, n_jobs=-1, verbose=1)
    # grid_search_LR.fit(training_data.headline, training_data.is_sarcastic)

    # # Report the results of the grid search:
    # print("\n\nBest parameters for baselines")
    # best_parameters_NB = grid_search_NB.best_estimator_.get_params()
    # print("Naive Bayes baseline best parameters:")
    # for parameter_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (parameter_name, best_parameters_NB[parameter_name]))
    # best_parameters_LR = grid_search_LR.best_estimator_.get_params()
    # print("Logistic Regression baseline best parameters:")
    # for parameter_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (parameter_name, best_parameters_LR[parameter_name]))

    
    
    

    

if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token