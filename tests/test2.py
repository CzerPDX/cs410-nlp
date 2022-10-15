import re

def main():
    file_content = open("green-eggs-and-ham.txt").read()
    # https://bobbyhadz.com/blog/python-split-string-on-whitespace-and-punctuation helped below
    # The pattern matches all punctuation listed below: , . ! ? and ". It will also match any
    # alphanumeric word to its end.
    file_content = file_content.lower()


    existingWord = "[\w]+"
    optionalWord = "(\w)+"
    existingPunctuation = "[,.!?\"]"
    pattern = existingWord + '|' + existingPunctuation
    unigramTokens = re.findall(pattern, file_content)
    
    # Map the prevalence of the words without context
    unigramMap = {}
    for token in unigramTokens:
        if token in unigramMap:

            unigramMap[token] += 1
        else:
            unigramMap[token] = 1
    # for token in unigramMap:
    #     print(token, unigramMap[token])
    # print(unigramMap)

    # for token in reversed(unigramMap):
    #     if re.match("^[A-Za-z0-9_-]*$", token):
    #         print(token, unigramMap[token])
    #     else:
    #         print(token + str(unigramMap[token]))
        
    bigramTokenList = []
    prev = None
    ### STILL NEED TO FIX IT SO SPACING WORKS CORRECTLY. MAYBE I SHOULD BE PULLING THIS FROM THE TEXT?
    for token in unigramTokens:
        if (prev != None) & (token != None):
            if (re.fullmatch(existingWord, prev) != None) & (re.fullmatch(existingWord, token) != None) or ((prev != "\"") & (re.fullmatch(existingWord, token) != None)):
                appendStr = prev + " " + token
                bigramTokenList.append(appendStr)
            else:
                appendStr = prev + token
                bigramTokenList.append(appendStr)
        prev = token

    for token in bigramTokenList:
        print(token)


if __name__ == "__main__":
    main()

        
#    wordMap[token].prevList[prev] = WordLink(prev)
    
# wordMap Hash table
# Key: token
#       prevList Hash table
#       Key: prevWord token