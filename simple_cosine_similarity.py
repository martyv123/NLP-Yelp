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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

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

# 4 groups: green, yellow, blue, pink

REVIEW_SET = []

focal = "Worth every penny! I splurged on this brand after seeing it online and it’s been my favorite guilty get this year by far. The glass back is so sleek, and I love the new darker color metal. It’s very professional and premium looking. I’ve even received compliments on how high-tech it looks. The display is very large so you can see who’s calling from afar. I am a little afraid about the durability- my friend has the same brand and hers broke within a week. So, maybe invest in a good warranty for extra protection."
focal2 = "Gets the job done. I’ve always used this manufacturer, so I trust them. It’s pretty basic and leaves my hands feeling clean. Harsh enough to gets the dirt and grime off easily, which is what I was looking for. The scent is very fresh, so you feel like you've just held a bunch of flowers - that's pleasant. The color is not very pretty, sort of brownish-greyish, but I can’t complain at such a low price - you get what you pay for. Overall, a good inexpensive option."

with open('certainty_two_way_data.csv', mode='r', encoding='utf-8') as input:
    csv_reader = csv.reader(input, delimiter=',')
    for row in csv_reader:
        REVIEW_SET.append({'group': row[0], 'id': row[1], 'focal': focal2, 'review': row[2]})

# Get the group we want to perform calculations on
group_set = []
text_set = []
for review in REVIEW_SET:
    if review['group'] == "pink":
        group_set.append(review)
        text_set.append(review['review'])

# Similarity calculations 

for i in range(len(text_set)):

    current_set = []
    current_set.append(focal2)
    current_set.append(text_set[i])
            
    LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')

    # print("Transforming tokens into vectors of term frequency (TF)")
    LemVectorizer.fit_transform(current_set)

    # print('\nThe indexes of the terms in the vector')
    # print(sorted(LemVectorizer.vocabulary_.items()))

    # print("\nConverting vectors into TF matrices")
    tf_matrix = LemVectorizer.transform(current_set).toarray()
    # print(tf_matrix)

    # Confirm matrix shape (n x m) where n = reviews and m = terms
    # print(tf_matrix_1.shape)
    # print(tf_matrix_2.shape)

    # print("\nCalculating inverse document frequency (IDF) matrices")
    # Each vector's component is now the idf for each term
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    # print(tfidfTran.idf_)

    # Manually verify that the IDF is correct
    # print("The idf for terms that appear in one document: " + str(idf(2,1)))
    # print("The idf for terms that appear in two documents: " + str(idf(2,2)))

    # print("\nCreating the TF-IDF matrices")
    # Transform method here multiples the tf matrix by the diagonal idf matrix
    # The method then divides the tf-idf matrix by the Euclidean norm
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    # print(tfidf_matrix.toarray())

    # print("\nCreating the cosine similarity matrices")
    # Multiply matrix by transpose to get final result
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    print("#: {}, score: {}".format(i+1, cos_similarity_matrix[0][1])) 

    group_set[i]['similarity'] = cos_similarity_matrix[0][1]

    # Calculating average and appending it to group_set
    # print("\nCalculating averages")
    # num_reviews = len(current_set)
    # for i in range(num_reviews):
    #     # i is the index of the focal review
    #     # cos_similarity_matrix[i] contains the array of similarity scores for the focal review 
    #     curr_scores = cos_similarity_matrix[i]
    #     np.delete(curr_scores, i)
    #     average = sum(curr_scores) / len(curr_scores)
    #     group_set[i]['average'] = average
        

# Write to CSV file
# print("\nWriting to file")
with open('certainty_two_way_data_focal_similarities.csv', mode='a', newline='') as output:
    dict_writer = csv.DictWriter(output, group_set[0].keys())
    dict_writer.writerows(group_set)

print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
sys.exit(0)    