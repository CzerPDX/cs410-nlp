import nltk

class Token:
    def __init__(self):
        self.prevTokenList = {}
        self.count = 1
    
    def incrementToken(self):
        self.count += 1

    def addPrev(self, prev):
        if prev not in self.prevTokenList:
            self.prevTokenList[prev] = 1
        else:
            self.prevTokenList[prev] += 1

    def getCount(self):
        return self.count
    
    def getPrev(self):
        return self.prevTokenList
        


class TokenMap:
    def __init__(self, newTokenList):
        self.tokenMap = {}
        self.addTokens(newTokenList)

    def addTokens(self, newTokenList):
        prev = None
        for token in newTokenList:
            if token not in self.tokenMap:
                self.tokenMap[token] = Token()
            else:
                self.tokenMap[token].incrementToken()
            self.tokenMap[token].addPrev(prev)
            prev = token
    
    def printTokenMap(self):
        for token in self.tokenMap:
            print(token, self.tokenMap[token].count, self.tokenMap[token].getPrev())

    def buildNextLevel(self):
        retList = []
        for token in self.tokenMap:
            for prev in self.tokenMap[token].getPrev():
                if (prev != None):
                    retList.append(token + ' ' + prev)
        return retList

            
        


def main():
    file_content = open("green-eggs-and-ham.txt").read()
    unigramList = nltk.word_tokenize(file_content)

    # Create the bigram map and list
    bigramMap = TokenMap(unigramList)
    bigramList = bigramMap.buildNextLevel()

    # Create the trigram map and list
    trigramMap = TokenMap(bigramList)
    trigramList = trigramMap.buildNextLevel()

    
    print(trigramList)

if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token