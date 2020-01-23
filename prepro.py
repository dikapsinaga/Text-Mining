import nltk 
from nltk.corpus import stopwords
import re



def tokenization(source):
    source = source.lower()
    source = source[:-1]
    
    removeList = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "'", '"', ",", ".", "/", "?", "!", "(", ")", ":", ";", "-", "_", "+", "=", "*", "@", "#", "$", "%", "^", "<", ">", "&", "[", "]", "{", "}", '')
    
    cleanWord =  ""
    for i, item in enumerate(cleanWord):
        print(i, item)

    # print(cleanWord)
    for word in source:
        if(word not in removeList):
            cleanWord += word
            print(cleanWord)
    
    word = cleanWord.split(" ")
    return word


def filtering(source):
    stoplist = set(stopwords.words("english"))
    
    filtered = [word for word in source if word not in stoplist]
    filtered = filtered[:-1]
    return filtered

documents = open("Doc1.txt", "r").read()
print(tokenization(documents))