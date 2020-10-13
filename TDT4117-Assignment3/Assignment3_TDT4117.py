import random
random.seed(123)
import codecs
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.corpora import Dictionary
import string

#---------- Loading and preparing data ----------

nltk.download('punkt')

f = codecs.open("pg3300.txt", "r", "utf-8")

stemmer = PorterStemmer()
f = f.read()

paragraphs = f.split('\n\r')
paragraphList = []
listOfParagraphs = []

# Remove signs and stem paragraphs
for paragraph in paragraphs:
    doc = paragraph.translate(str.maketrans('', '', string.punctuation))
    lowercaseDoc = stemmer.stem(doc.lower())
    paragraphList.append(lowercaseDoc)

# If gutenberg appears in paragraph then delete
originialParagraph = []
for par in paragraphList:
    if 'gutenberg' in par:
        paragraphList.remove(par)
    originialPara = paragraphList
    document = word_tokenize(par)
    listOfParagraphs.append(document)

# Stemming the text
for paragraph in range(len(listOfParagraphs)):
        for indexOfWord in range(len(listOfParagraphs[paragraph])):
            listOfParagraphs[paragraph][indexOfWord] = stemmer.stem(listOfParagraphs[paragraph][indexOfWord])

   
# Making dictionary and removing stopwords
dictionary = Dictionary(listOfParagraphs)
f = codecs.open("common-english-words.txt", "r", "utf-8")
stopwords = f.read().split(',')
stopword_ids = []
for word in stopwords:
    try:
        stopwordid = dictionary.token2id[word]
        stopword_ids.append(stopwordid)
    except:
        continue
dictionary.filter_tokens(stopword_ids)
documentToBow = []
for para in listOfParagraphs:
    documentToBow.append(dictionary.doc2bow(para))

# Creating tfidf model, lsi model and matrices

tfidfModel = gensim.models.TfidfModel(documentToBow)
tfidfCorpus = tfidfModel[documentToBow]
tfidfMatrix = gensim.similarities.MatrixSimilarity(tfidfCorpus)
lsiModel = gensim.models.LsiModel(tfidfCorpus, id2word=dictionary, num_topics=100)
lsiCorpus = lsiModel[documentToBow]
lsiMatrix = gensim.similarities.MatrixSimilarity(lsiCorpus)
print("Report and try to interpret first 3 LSI topics: ")
topics = lsiModel.show_topics(3)
print("3 first lsi topics: ", topics)


def preprocessing(query):
    stemmer = PorterStemmer()
    words = query.split(' ')
    lowercasedoc = []
    for word in words:
         lowercasedoc.append(stemmer.stem(word.translate(str.maketrans('', '', string.punctuation+"\n\r\t"))))
    return lowercasedoc


# Queries

query = preprocessing("What is the function of money?")
print(query)
query = dictionary.doc2bow(query)
print("dictionary ", query)
print(dictionary[744])
query = tfidfModel[query]
print("tfidfmodel", query)
print(dictionary[815], ":", dictionary[1142])

#
doc2similarity = enumerate(tfidfMatrix[query])
relevant = sorted(doc2similarity, key=lambda kv: -kv[1])[:3] 
print(sorted(doc2similarity, key=lambda kv: -kv[1])[:3] )

print("The top 3  most relevant paragraphs for the query 'What is the function of money?' according to TF-IDF model: ")
print("Relevant 1: \n"+originialPara[relevant[0][0]] + "\n")
print("Relevant 2: \n"+originialPara[relevant[1][0]] + "\n")
print("Relevant 3: \n"+originialPara[relevant[2][0]])

# LSI-conversion
query = preprocessing("What is the function of money?")
query = dictionary.doc2bow(query)

lsiQuery = lsiModel[query]
topTopics = sorted(lsiQuery, key=lambda kv: - abs(kv[1]))[:3]
print("Top 3 topics with lsi topics weights:")
print(topTopics)
topics = lsiModel.show_topics(100)
for topic in topTopics:
    print("\n LSI topic", topic[0], ":")
    print(topics[topic[0]])


# find the 3 topics most relevant paragraphs according to LSI model and top three with most significant weights:
doc2similarity = enumerate(lsiMatrix.get_similarities(lsiQuery))
print("\n")
sortedParagraphs2 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
print("Top 3 paragraphs: ", sortedParagraphs2)
for par in sortedParagraphs2:
    print("Paragraph", par[0], "\n")



