#!/usr/bin/python3
#
# Sample parsing, lemmatization, TF analysis, and cosine similarity on 2 review groups (10 reviews total)
#
# Maintainers: vo.ma@northeastern.edu

import csv
import json
import sys
import math
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

REVIEW_SET_1 = []   
REVIEW_SET_2 = []  

# Initialize the Tokenizer
tokenizer = nltk.tokenize.TweetTokenizer()
# Initialize the POS mapper
wordnet = nltk.corpus.wordnet
# Initialize the Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()


# Open and read the sample.txt file reviews in each group (10 total, 5 in each group)
def parse_sample():
    global REVIEW_SET_1
    global REVIEW_SET_2

    file = open('sample.txt')
    all_reviews = file.readlines()
    for review in all_reviews:
        # Skipping new lines ('\n')
        if len(review) > 3:
            if len(REVIEW_SET_1) != 5:
                REVIEW_SET_1.append(review)
            else:
                REVIEW_SET_2.append(review)
        
    # print(len(REVIEW_SET_1))
    # print(len(REVIEW_SET_2))
    # print(REVIEW_SET_1)
    # print(REVIEW_SET_2)
    
    file.close()

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


if __name__ == '__main__':
    print('-----------------------------------')
    print("Starting sample analysis script...")
    print("Parsing sample text file")
    parse_sample()
    
    LemVectorizer_1 = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
    LemVectorizer_2 = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
    print("Transforming tokens into vectors of term frequency (TF)")
    LemVectorizer_1.fit_transform(REVIEW_SET_1)
    LemVectorizer_2.fit_transform(REVIEW_SET_2)

    # The indexes of the terms in the vector
    # print(LemVectorizer.vocabulary_)

    print("Convert vectors into TF matrices")
    tf_matrix_1 = LemVectorizer_1.transform(REVIEW_SET_1).toarray()
    tf_matrix_2 = LemVectorizer_2.transform(REVIEW_SET_2).toarray()
    # print(tf_matrix_1)
    # print(tf_matrix_2)

    # Confirm matrix shape (n x m) where n = reviews and m = terms
    # print(tf_matrix_1.shape)
    # print(tf_matrix_2.shape)

    print("Calculate inverse document frequency (IDF) matrices")
    # Each vector's component is now the idf for each term
    tfidfTran_1 = TfidfTransformer(norm="l2")
    tfidfTran_2 = TfidfTransformer(norm="l2")
    tfidfTran_1.fit(tf_matrix_1)
    tfidfTran_2.fit(tf_matrix_2)
    # print(tfidfTran_1.idf_)
    # print(tfidfTran_2.idf_)

    # Manually verify that the IDF is correct
    # print("The idf for terms that appear in one document: " + str(idf(5,1)))
    # print("The idf for terms that appear in two documents: " + str(idf(5,2)))

    print("Get the TF-IDF matrices")
    # Transform method here multiples the tf matrix by the diagonal idf matrix
    # The method then divides the tf-idf matrix by the Euclidean norm
    tfidf_matrix_1 = tfidfTran_1.transform(tf_matrix_1)
    tfidf_matrix_2 = tfidfTran_2.transform(tf_matrix_2)
    # print(tfidf_matrix_1.toarray())
    # print(tfidf_matrix_1.toarray())

    print("Get the cosine similarity matrices\n")
    # Multiply matrix by transpose to get final result
    cos_similarity_matrix_1 = (tfidf_matrix_1 * tfidf_matrix_1.T).toarray()
    cos_similarity_matrix_2 = (tfidf_matrix_2 * tfidf_matrix_2.T).toarray()
    print("Review group 1\n")
    print(cos_similarity_matrix_1)
    print('\n')
    print("Review group 2\n")
    print(cos_similarity_matrix_2)

    print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
    sys.exit(0)