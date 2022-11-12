# texts = [ "mycorpus.txt" ]
from gensim import corpora
from gensim.utils import simple_preprocess
# dictionary = corpora.Dictionary(texts)

from smart_open import open  # for transparently opening remote files
from pprint import pprint  # pretty-printer

def parseParagraphsSilmarillion():
    f = open('data/silmarillion.txt', 'r')
    paragraphList = []
    currParagraph = ""
    for line in f:
        # If the current line is an empty newline and the current paragraph is not empty 
        # it's time to save the paragraph and then reset it
        if (line == "\n") and (currParagraph != ""):
            paragraphList.append(currParagraph)
            currParagraph = ""
        elif (line != "\n"):
            formatted = line.rstrip() + ' '
            currParagraph += formatted.rstrip()
        # If the line is an empty newline and the currParagraph is empty we just want to move on
        # to the next line, so do nothing.

    for paragraph in paragraphList:
        print(paragraph)

    with open("../data/silmarillion-formatted.txt", "w") as txt_file:
        for line in paragraphList:
            txt_file.write(line + '\n')


            

class MyCorpus:
    def __init__(self, documentList):
        self.documentList = documentList

    def __iter__(self):
        for document in self.documentList:
            for line in open(document, encoding="ISO-8859-1"):
                # assume there's one document per line, tokens separated by whitespace
                yield simple_preprocess(line, True);

# parseParagraphsSilmarillion()

# Make sure the documents are formatted so that each line is one paragraph
documentList = [
    'data/01 - The Fellowship Of The Ring.txt',
    'data/02 - The Two Towers.txt',
    'data/silmarillion-formatted.txt'
]

corpus_memory_friendly = MyCorpus(documentList)  # doesn't load the corpus into memory!
# print(corpus_memory_friendly)

# collect statistics about all tokens
dictionary = corpora.Dictionary(corpus_memory_friendly)
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)


# for item in dictionary:
#     print(dictionary[item], ' ', dictionary.cfs[item])

paperbackID = dictionary.token2id['paperback']
print('Paperback appears ', dictionary.cfs[paperbackID], ' times')

hobbitID = dictionary.token2id['hobbit']
print('Hobbit appears ', dictionary.cfs[hobbitID], ' times')

print('Source documents contained ', dictionary.num_pos, ' total words')

print('Dictionary contains ', dictionary.num_nnz, ' unique words')

# We can vectorize a new document based on the current dictionary
new_doc = "hobbit hobbit hobbit"
new_vec = dictionary.doc2bow(simple_preprocess(new_doc))
print(new_vec)

print('5 Most common words: ')
print(dictionary[0])
print(dictionary[1])
print(dictionary[2])
print(dictionary[3])
print(dictionary[4])

# We can convert our entire original corpus to a list of vectors:
bow_corpus = [dictionary.doc2bow(text) for text in corpus_memory_friendly]
# for item in bow_corpus:
#     print(item)


# # MODELS
# One simple example of a model is tf-idf. The tf-idf model transforms vectors from the 
# bag-of-words representation to a vector space where the frequency counts are weighted 
# according to the relative rarity of each word in the corpus.

# Here’s a simple example. Let’s initialize the tf-idf model, training it on our corpus 
# and transforming the string “elven kings hobbit packs sheltered nazgul”:


from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "elven kings hobbit packs sheltered nazgul".lower().split()
for item in tfidf[dictionary.doc2bow(words)]:
    print(item)


# test_corpus = MyCorpus(testDocument)  # doesn't load the corpus into memory!
# print(corpus_memory_friendly)

from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

query_document = 'elven hobbit'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
# print(list(enumerate(sims)))
for item in sims:
    print(item)