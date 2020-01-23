import numpy as np
import PreProcessing as pre
import TermWeighting as termW
import KNN as knn
import scipy

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Data:
    def __init__(self, document, classification):
        self.document = document
        self.classification = classification

data = []
data.append(Data(open("Doc1.txt", "r").read(), "A"))
data.append(Data(open("Doc2.txt", "r").read(), "A"))
data.append(Data(open("Doc3.txt", "r").read(), "B"))
data.append(Data(open("Doc4.txt", "r").read(), "B"))
data.append(Data(open("Doc5.txt", "r").read(), "C"))


terms = []
for i in range (len(data)):
    terms.append(pre.termFromDocuments(pre.stemming(pre.filtering(pre.tokenization(data[i].document)))))

terms = pre.termFromDocuments(terms)

# binaryWeight = termW.binaryTermWeighting(data, terms)
# rawWeight = termW.rawTermWeighting(data, terms)
logWeight = termW.logTermWeighting(data, terms)
df = termW.documentFrequency(data, terms)
idf = termW.inverseDocumentFrequency(data, df)
tf_idf = termW.tf_idf(logWeight, idf)


dataTest = []
dataTest.append(Data(open("Doctest.txt", "r").read(), ""))

logWeightTest = termW.logTermWeighting(dataTest, terms)
tf_idfTest = termW.tf_idf(logWeightTest, idf)

cosSim = knn.cosineSimilarity(tf_idf, tf_idfTest[0])
neighbors = knn.getNeighbors(3, cosSim)

print(knn.decision(data, neighbors, cosSim))