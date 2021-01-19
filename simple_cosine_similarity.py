#!/usr/bin/python3
#
# Sample parsing, lemmatization, TF analysis, and cosine similarity on small sized review groups
#
# Maintainers: vo.ma@northeastern.edu

import csv
import json
import sys
import math
import string
import nltk
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the Tokenizer
tokenizer = nltk.tokenize.TweetTokenizer()
# Initialize the POS mapper
wordnet = nltk.corpus.wordnet
# Initialize the Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Map POS (Part of Speech) tag to first character lemmatize() accepts
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Tokenize the words of a text with their POS
def get_lem_tokens(tokens):
    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

# Normalize the tokens by removing punctuation and cases
def lem_normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return get_lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Manually compute the IDF of a term n given the document frequency
def idf(n, df):
    result = math.log((n+1.0)/(df+1.0)) + 1
    return result


REVIEW_SET = []

with open('other_reviews/cellphone.csv', mode='r', encoding='utf-8') as input:
    for row in input:
        REVIEW_SET.append(row)

# Similarity calculations 

for i in range(2):
    current_set = []
    if i == 0:
        current_set = REVIEW_SET[:5]
        print('Analyzing ' + str(len(current_set)) + ' reviews')
    else:
        current_set = REVIEW_SET[5:]
        print('Analyzing ' + str(len(current_set)) + ' reviews')
            
    LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')

    print("Transforming tokens into vectors of term frequency (TF)")
    LemVectorizer.fit_transform(current_set)

    print('\nThe indexes of the terms in the vector')
    print(sorted(LemVectorizer.vocabulary_.items()))

    print("\nConverting vectors into TF matrices")
    tf_matrix = LemVectorizer.transform(current_set).toarray()
    print(tf_matrix)

    # Confirm matrix shape (n x m) where n = reviews and m = terms
    # print(tf_matrix_1.shape)
    # print(tf_matrix_2.shape)

    print("\nCalculating inverse document frequency (IDF) matrices")
    # Each vector's component is now the idf for each term
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    print(tfidfTran.idf_)

    # Manually verify that the IDF is correct
    # print("The idf for terms that appear in one document: " + str(idf(2,1)))
    # print("The idf for terms that appear in two documents: " + str(idf(2,2)))

    print("\nCreating the TF-IDF matrices")
    # Transform method here multiples the tf matrix by the diagonal idf matrix
    # The method then divides the tf-idf matrix by the Euclidean norm
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    print(tfidf_matrix.toarray())

    print("\nCreating the cosine similarity matrices")
    # Multiply matrix by transpose to get final result
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    print(cos_similarity_matrix)

    # Write to CSV file
    write_file = numpy.asarray(cos_similarity_matrix)
    if i == 0:
        numpy.savetxt("cellphone_similar.csv", write_file, delimiter=",") 
    else:
        numpy.savetxt("cellphone_different.csv", write_file, delimiter=",")

print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
sys.exit(0)    