# Brooke Czerwinski
# Homework 1
# Natural Language Processing - CS 410

import nltk

def parseJason(fname):
    for line in open(fname, 'r'):
        yield eval(line)            
        


def main():
    # Read the data in from the file
    filename = "Sarcasm_Small-Set.json"
    data = list(parseJason(filename))

    for item in data:
        print(item)

if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token