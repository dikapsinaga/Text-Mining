import numpy as np
import operator


def lengths (document) :
    temp = 0
    for word in document :
        temp += word **2
    temp = temp ** 0.5
    return temp

def cosineSimilarity(documents, query):
    cosSim = []

    for i in range(len(documents)):
        temp = 0
        for j in range(len(documents[0])):
            temp += documents[i][j] * query[j]
        temp = temp / (lengths(documents[i]) * lengths(query))
        cosSim.append(temp)
    
    return cosSim

def getNeighbors(k, cosineSimilarity):
    print(cosineSimilarity)
    neighbors = []
    similarity = np.argsort(np.array(cosineSimilarity) * -1 )
    print(similarity)
    for x in range(k):
        neighbors.append(similarity[x])
    return neighbors


def decision(datas, neighbors, cosineSimilarity):
    classification = [data.classification for data in datas]
    
    print(classification)
    print(neighbors)

    classVotes = {}
    for i in range (len(neighbors)):
        response = classification[i]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] = 1

    print(classVotes)
    sorted_class = sorted(classVotes.items(), key=operator.itemgetter(1), reverse = True)
    return sorted_class[0][0] 
