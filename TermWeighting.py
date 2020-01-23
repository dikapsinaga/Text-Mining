import math
import PreProcessing as pre
import numpy as np

def binaryTermWeighting(data, terms):
    binaryWeight = []
    documents = [documents.document for documents in data]

    for document in documents:
        documentWeight = []
        document = pre.split(pre.stemming(pre.filtering(pre.tokenization(document))))

        for term in terms:
            if term in document:
                documentWeight.append(1)
            else:
                documentWeight.append(0)

        binaryWeight.append(documentWeight)

    return binaryWeight

def rawTermWeighting(data, terms):
    rawWeight = []
    documents = [documents.document for documents in data]

    for document in documents:
        documentWeight = []
        document = pre.split(pre.stemming(pre.filtering(pre.tokenization(document))))

        for term in terms:
            documentWeight.append(document.count(term))

        rawWeight.append(documentWeight)

    return rawWeight

def logTermWeighting(data, terms):
    logWeight = []

    documents = [documents.document for documents in data]

    for document in documents:
        documentWeight = []
        document = pre.split(pre.stemming(pre.filtering(pre.tokenization(document))))

        for term in terms:
            count = document.count(term)
            if count > 0 :
                documentWeight.append(1 + math.log10(count))
            else :
                documentWeight.append(0)

        logWeight.append(documentWeight)

    return logWeight    

def documentFrequency(data, terms):
    df = []
    documents = [documents.document for documents in data]

    for term in terms:
        dfWeight = 0
        for document in documents:
            document = pre.split(pre.stemming(pre.filtering(pre.tokenization(document))))
            if term in document:
                dfWeight += 1
        df.append(dfWeight)
    return df

def inverseDocumentFrequency(data, dfs):
    return [math.log10(len(data) / df) for df in dfs]


def tf_idf(termFrequencies, inverseDocumentFrequencies):
    tf_idf = []

    for documentTermFrequencies in termFrequencies:
        row_tf_idf = []
        for i in range(0, len(inverseDocumentFrequencies)):
            row_tf_idf.append(documentTermFrequencies[i]*inverseDocumentFrequencies[i])
        tf_idf.append(row_tf_idf)

    return tf_idf