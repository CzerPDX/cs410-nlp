# Tutorial from https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def main():
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    

    count_vect = CountVectorizer()

    print('\n\nTokenizing Text')
    # Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, 
    # which builds a dictionary of features and transforms documents to feature vectors:
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(X_train_counts)

    print('target = ', twenty_train.target)
    # This saves every word in the corpus (no duplicates, so not equal to tokens).
    wordsList = count_vect.get_feature_names_out()
    for word in wordsList:
        print(word)


    # Once fitted, the vectorizer has built a dictionary of feature indices:

    # CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the 
    # vectorizer has built a dictionary of feature indices:

    print('Feature index for the word "algorithm": ', count_vect.vocabulary_.get(u'algorithm'))

    # The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

    ################################################
    # From occurrences to frequencies
    # Occurrence count is a good start but there is an issue: longer documents will have higher 
    # average count values than shorter documents, even though they might talk about the same topics.

    # To avoid these potential discrepancies it suffices to divide the number of occurrences of each 
    # word in a document by the total number of words in the document: these new features are called 
    # tf for Term Frequencies.

    # Another refinement on top of tf is to downscale weights for words that occur in many documents 
    # in the corpus and are therefore less informative than those that occur only in a smaller portion 
    # of the corpus.

    # This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
    # Both tf and tf–idf can be computed as follows using TfidfTransformer:
    from sklearn.feature_extraction.text import TfidfTransformer
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print('Term frequency (tf) shape: (n_samples, n_features) = ', X_train_tf.shape)

    #In the above example-code, we firstly use the fit(..) method to fit our estimator to the data and 
    # secondly the transform(..) method to transform our count-matrix to a tf-idf representation. These 
    # two steps can be combined to achieve the same end result faster by skipping redundant processing. 
    # This is done through using the fit_transform(..) method as shown below, and as mentioned in the 
    # note in the previous section:
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print('Term frequency times inverse document frequency shape (tfidf): (n_samples, n_features) = ', X_train_tfidf.shape)




    ##################################################
    print('\n\nTraining a classifier')
    # Now that we have our features, we can train a classifier to try to predict the category of a post. 
    # Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task. scikit-learn 
    # includes several variants of this classifier; the one most suitable for word counts is the multinomial 
    # variant:
    from sklearn.naive_bayes import MultinomialNB
    
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    # To try to predict the outcome on a new document we need to extract the features using almost the same 
    # feature extracting chain as before. The difference is that we call transform instead of fit_transform 
    # on the transformers, since they have already been fit to the training set:

    docs_new = ['God is love', 'OpenGL on the GPU is fast']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))




    #####################################################
    print('\n\nBuilding a pipeline')
    #In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides 
    # a Pipeline class that behaves like a compound classifier:
    from sklearn.pipeline import Pipeline

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()), # Classifier
    ])

    # The names vect, tfidf and clf (classifier) are arbitrary. We will use them to perform grid search for 
    # suitable hyperparameters below. We can now train the model with a single command:

    print('Our pipeline can train the model with a single command: ')
    print(text_clf.fit(twenty_train.data, twenty_train.target))

    

    #######################################################
    print('\n\nEvaluation of the performance on the test set')
    # Evaluating the predictive accuracy of the model is equally easy:
    import numpy as np

    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    docs_test = twenty_test.data
    predicted = text_clf.predict(docs_test)
    accuracy = np.mean(predicted == twenty_test.target)
    accuracy *= 100
    print('Naive Bayes Accuracy: ', accuracy, '%')

    # We achieved 83.5% accuracy. Let’s see if we can do better with a linear support vector machine (SVM), 
    # which is widely regarded as one of the best text classification algorithms (although it’s also a bit 
    # slower than naïve Bayes). We can change the learner by simply plugging a different classifier object 
    # into our pipeline:
    from sklearn.linear_model import SGDClassifier

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])

    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    accuracy = np.mean(predicted == twenty_test.target)
    accuracy *= 100
    print('Support Vector Machine (SVM) Accuracy: ', accuracy, '%')

    # We achieved 91.3% accuracy using the SVM. scikit-learn provides further utilities for more detailed 
    # performance analysis of the results:
    from sklearn import metrics

    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    
    print('PRINTING NAMES OUTTTTT')
    for name in twenty_test.target_names:
        print(name)

    confusionMatrix = metrics.confusion_matrix(twenty_test.target, predicted)

    print('\nConfusion Matrix:')
    for row in confusionMatrix:
        print(row)


    # Confusion Matrix Output:
    # [256  11  16  36]
    # [  4 380   3   2]
    # [  5  35 353   3]
    # [  5  11   4 378]

    # As expected the confusion matrix shows that posts from the newsgroups on atheism and Christianity are 
    # more often confused for one another than with computer graphics.

    
    #########################################################
    print('Parameter Tuning Using Gridsearch')
    # We’ve already encountered some parameters such as use_idf in the TfidfTransformer. Classifiers tend to 
    # have many parameters as well; e.g., MultinomialNB includes a smoothing parameter alpha and SGDClassifier 
    # has a penalty parameter alpha and configurable loss and penalty terms in the objective function (see the 
    # module documentation, or use the Python help function to get a description of these).

    # Instead of tweaking the parameters of the various components of the chain, it is possible to run an 
    # exhaustive search of the best parameters on a grid of possible values. We try out all classifiers on 
    # either words or bigrams, with or without idf, and with a penalty parameter of either 0.01 or 0.001 for 
    # the linear SVM:
    from sklearn.model_selection import GridSearchCV
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    # Obviously, such an exhaustive search can be expensive. If we have multiple CPU cores at our disposal, 
    # we can tell the grid searcher to try these eight parameter combinations in parallel with the n_jobs 
    # parameter. If we give this parameter a value of -1, grid search will detect how many cores are installed 
    # and use them all:

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

    # The grid search instance behaves like a normal scikit-learn model. Let’s perform the search on a smaller 
    # subset of the training data to speed up the computation:
    gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
    
    # The result of calling fit on a GridSearchCV object is a classifier that we can use to predict:
    prediction = twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
    print('Using gridsearch, the preiction for the phrase "God is love" is: ', prediction)

    # The object’s best_score_ and best_params_ attributes store the best mean score and the parameters setting corresponding to that score:
    bestScore = gs_clf.best_score_
    print('Best score: ', bestScore)
    bestParamsList = gs_clf.best_params_
    for param_name in sorted(parameters.keys()):
        print(param_name, 
        ': ', 
        gs_clf.best_params_[param_name])

    # A more detailed summary of the search is available at gs_clf.cv_results_.
    # The cv_results_ parameter can be easily imported into pandas as a DataFrame for further inspection.





if __name__ == "__main__":
    main()