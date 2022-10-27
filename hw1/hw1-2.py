# Brooke Czerwinski
# Homework 1
# Natural Language Processing - CS 410

# References:
# https://scikit-learn.org/stable/auto_examples/model_selection/gridSearch_text_feature_extraction.html


import os
import itertools

from nltk.tokenize import word_tokenize

def parseJSON(fname, start_idx, stop_idx):
    with open(fname, 'r') as text_file:
        for line in itertools.islice(text_file, start_idx, stop_idx):
            yield eval(line)     
 
    


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

    # Preprocess text

    

if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token