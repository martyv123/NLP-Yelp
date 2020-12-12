#!/usr/bin/python3
#
# Sample parsing, lemmatization, and TF analysis on 2 review groups (10 reviews total)
#
# Copyright: (c) Yakov Bart
#            (c) Marty Vo
#
# Maintainers: y.bart@northeastern.edu
#              vo.ma@northeastern.edu

import csv
import json
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
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


# Substitute contractions for their their full word representations
def remove_contraction(word):
    contractions = { 
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"}

    # If the contraction exists, return the representation - otherwise just return the initial word
    return contractions.get(word, word)

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

# Lemmatize the review sets
def lemmatize(review_set):
    lemmatized_output = []
    for review in review_set:
        # Tokenize the review into individual words
        word_list = tokenizer.tokenize(review)
        # Remove punctuation
        word_list = [word.lower() for word in word_list if word.isalpha()]
        # print(word_list)
        # print('\n')
        for index, word in enumerate(word_list):
            if "'" in word:
                word_list.remove(word)
                contraction_string = remove_contraction(word)
                print('contraction string: ' + contraction_string)
                if "/" in contraction_string:
                    # TODO: Not sure how to handle contractions with multiple meanings yet.
                    #       Will most likely need to consider the sentence that it resides in,
                    #       requiring a more in-depth processing. For now, raise an exception.
                    print(contraction_string)
                    raise Exception("Error: Contraction has multiple derivations")
                else:
                    contraction_words = contraction_string.split()
                    contraction_words.reverse()
                    print('contraction words: ' + str(contraction_words))
                    for contraction_word in contraction_words:
                        word_list.insert(index, contraction_word)
        lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list]
        lemmatized_output.append(lemmatized)
    # print(lemmatized_output)
    # print('\n')
    return lemmatized_output


if __name__ == '__main__':
    print('-----------------------------------')
    print("Starting sample analysis script...")
    print("Parsing sample text file...")
    parse_sample()

    print("Lemmatizing review group 1 and group 2...")
    # Storing the output as a list of strings
    lemmatized_group_1 = lemmatize(REVIEW_SET_1)
    print(lemmatized_group_1)
    lemmatized_group_2 = lemmatize(REVIEW_SET_2)
    print(lemmatized_group_2)

    # Analyzing reviews in set 1
    # TODO: finish!

    # Analyzing reviews in set 2
    # TODO: finish!

    # Performing cosine similarity for reviews in set 1
    # TODO: finish!

    # Performing cosine similarity for reviews in set 2
    # TODO: finish!