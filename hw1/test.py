
import nltk
import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline 

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def pos_tag(sentence, clf):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            print('append = ', tagged[index][1])
            y.append(tagged[index][1])

    return X, y

def main():
 
    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    
    print(tagged_sentences[0])
    print("Tagged sentences: ", len(tagged_sentences))
    print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))
    
    # [(u'Pierre', u'NNP'), (u'Vinken', u'NNP'), (u',', u','), (u'61', u'CD'), (u'years', u'NNS'), (u'old', u'JJ'), (u',', u','), (u'will', u'MD'), (u'join', u'VB'), (u'the', u'DT'), (u'board', u'NN'), (u'as', u'IN'), (u'a', u'DT'), (u'nonexecutive', u'JJ'), (u'director', u'NN'), (u'Nov.', u'NNP'), (u'29', u'CD'), (u'.', u'.')]
    # Tagged sentences:  3914
    # Tagged words: 100676

    pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))

    # Split the dataset for training and testing
    cutoff = int(.75 * len(tagged_sentences))
    training_sentences = tagged_sentences[:cutoff]
    test_sentences = tagged_sentences[cutoff:]
    
    print(len(training_sentences))   # 2935
    print(len(test_sentences))         # 979
    
    X, y = transform_to_dataset(training_sentences)

    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])
    
    clf.fit(X[:10000], y[:10000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
    
    print('Training completed')
    
    X_test, y_test = transform_to_dataset(test_sentences)
    
    print("Accuracy:", clf.score(X_test, y_test))
    
    # Accuracy: 0.904186083882
    # not bad at all :)
    
    zipObj = pos_tag(nltk.word_tokenize('This is my friend, John.'), clf)
    # [('This', u'DT'), ('is', u'VBZ'), ('my', u'JJ'), ('friend', u'NN'), (',', u','), ('John', u'NNP'), ('.', u'.')
    for item in zipObj:
        print(item)

if __name__ == "__main__":
    main()