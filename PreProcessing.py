from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def tokenization(source):
    source = source.lower()
    source = source[:-1]
    source = source.replace(",", "")
    source = source.split(". ")

    documents = []
    for document in source:
        documents.append(document.split(" "))

    return documents


def filtering(documents):
    stopwords = open("stopword-list.txt", "r")
    stopwords = stopwords.read()
    stopwords = stopwords.split("\n")

    filtered = []

    for document in documents:
        filtered.append([word for word in document if word not in stopwords])

    return filtered

def stemming(documents):
    stemmed = []
    for document in documents:
        words = []
        for word in document:
            words.append(stemmer.stem(word))
        stemmed.append(words)

    return stemmed

def termFromDocuments(documents):
    terms = []
    for document in documents:
        for word in document:
            if word not in terms:
                terms.append(word)
    return terms

def split(documents):
    docs = []
    for document in documents:
        for word in document:
            docs.append(word)
    return docs