#!/usr/bin/python3
#
# Sample parsing, lemmatization, TF analysis, and cosine similarity on multiple review groups.
#
# Maintainers: vo.ma@northeastern.edu

import os
import csv
import json
import sys
import math
import string
import nltk
import numpy
import copy
import time
import pandas as pd
from timeit import default_timer as timer
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

REVIEWED = 0
REVIEW_SET = []
START = True


# Initialize the Tokenizer
tokenizer = nltk.tokenize.TweetTokenizer()
# Initialize the POS mapper
wordnet = nltk.corpus.wordnet
# Initialize the Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Open and read the review data for a Yelp business
def parse_yelp_json_file(file_name, business_id):
    global ALL_REVIEWS
  
    with open(file_name, encoding='utf-8') as to_read:
        for row in to_read:
            review = json.loads(row)
            if review['business_id'] == business_id:
                ALL_REVIEWS.append(review)
    
    # Sort ALL_REVIEWS by date (newest to oldest)
    ALL_REVIEWS.sort(reverse=True, key=lambda x: x['date'])

    # print(ALL_REVIEWS)

    # Write review data to JSON file
    json_file = "williamsburg_smile_design_reviews.json"
    with open(json_file, 'w', newline='') as to_write:
        json_object = json.dumps({'reviews': ALL_REVIEWS}, indent=4)
        to_write.write(json_object)

# Open and read the text file reviews in each group
def parse_txt_file(file_name):
    global REVIEW_SET_1
    global REVIEW_SET_2

    file = open(file_name)
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

# Open and read the csv file 
def parse_csv_file(file_name):
    global REVIEW_SET_1
    global REVIEW_SET_2

    ''' Make sure CSVs are saved in MS-DOS format to avoid weird string conversions from utf-8 '''

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        initial_review = ""
        for row in csv_reader:
            if line_count == 0:
                initial_review = row
                line_count += 1
            else:
                REVIEW_SET_1.append(' '.join(initial_review))
                REVIEW_SET_1.append(' '.join(row))

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

# a. Similarity score with the most recently posted review (relative to the focal review), 
#    2nd most recent posted review,…10th most recent posted review.
#    If less than 10 reviews for the focal review, output similarity score of 1 for each remaining review
def calculate_method_a(review_set, pid):
    print('\n' + str(pid) + ': Now calculating method a...')
    final_output = []

    for index, review in enumerate(review_set):

        # Get 1 focal review + 10 prior reviews
        current_set = review_set[index:(index + 11)]

        # Get just the texts from the current set
        texts = []
        for item in current_set:
            texts.append(item['text'])

        # Output 1 for similarity scores when less than 10 prior reviews
        if len(current_set) != 11:
            for item in current_set:
                # Assign the review IDs compared against
                item['reviewed_against_1a'] = "none"
                item['reviewed_against_2a'] = "none"
                item['reviewed_against_3a'] = "none"
                item['reviewed_against_4a'] = "none"
                item['reviewed_against_5a'] = "none"
                item['reviewed_against_6a'] = "none"
                item['reviewed_against_7a'] = "none"
                item['reviewed_against_8a'] = "none"
                item['reviewed_against_9a'] = "none"
                item['reviewed_against_10a'] = "none"

                # Assign the similarity scores
                item['score_1a'] = 1
                item['score_2a'] = 1
                item['score_3a'] = 1
                item['score_4a'] = 1
                item['score_5a'] = 1
                item['score_6a'] = 1
                item['score_7a'] = 1
                item['score_8a'] = 1
                item['score_9a'] = 1
                item['score_10a'] = 1

                final_output.append(item)

            # print('**Less than 10 prior reviews - finishing at review ' + str(review['review_id']))
            break
        else:
            LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
            # print("Transforming tokens into vectors of term frequency (TF)")
            LemVectorizer.fit_transform(texts)

            # print('\nThe indexes of the terms in the vector')
            # print(sorted(LemVectorizer.vocabulary_.items()))

            # print("\nConverting vectors into TF matrices")
            tf_matrix = LemVectorizer.transform(texts).toarray()
            # print(tf_matrix)

            # Confirm matrix shape (n x m) where n = reviews and m = terms
            # print(tf_matrix_1.shape)
            # print(tf_matrix_2.shape)

            # print("\nCalculating inverse document frequency (IDF) matrices")
            # Each vector's component is now the idf for each term
            tfidfTran = TfidfTransformer(norm="l2")
            tfidfTran.fit(tf_matrix)
            # print(len(tfidfTran.idf_))
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
            # print(cos_similarity_matrix)

            # Getting first column of similarity matrix - should be ten scores after stripping the first score
            first_column = cos_similarity_matrix[0]
            scores = first_column[1:]

            current = current_set[1:]

            # Assign the review IDs compared against
            review['reviewed_against_1a'] = current[0]['review_id']
            review['reviewed_against_2a'] = current[1]['review_id']
            review['reviewed_against_3a'] = current[2]['review_id']
            review['reviewed_against_4a'] = current[3]['review_id']
            review['reviewed_against_5a'] = current[4]['review_id']
            review['reviewed_against_6a'] = current[5]['review_id']
            review['reviewed_against_7a'] = current[6]['review_id']
            review['reviewed_against_8a'] = current[7]['review_id']
            review['reviewed_against_9a'] = current[8]['review_id']
            review['reviewed_against_10a'] = current[9]['review_id']

            # Assign the similarity scores
            review['score_1a'] = scores[0]
            review['score_2a'] = scores[1]
            review['score_3a'] = scores[2]
            review['score_4a'] = scores[3]
            review['score_5a'] = scores[4]
            review['score_6a'] = scores[5]
            review['score_7a'] = scores[6]
            review['score_8a'] = scores[7]
            review['score_9a'] = scores[8]
            review['score_10a'] = scores[9]

            # print('Assigning similarity scores (method a) for review ' + review['review_id'])

            final_output.append(review)

    return final_output

# b. Similarity score with the most recently posted review (relative to the focal review) of the same valence as the focal review, 
#    2nd most recent posted review of the same valence as the focal review,
#    …10th most recent posted review of the same valence as the focal review.
#    If less than 10 reviews for the focal review, output similarity score of 1 for each remaining review
def calculate_method_b(review_set, pid):
    print('\n' + str(pid) + ': Now calculating method b...')
    final_output = []

    for review in review_set:

        current_set = []
        current_set.append(review)

        # Get 1 focal review + 10 prior reviews with same number of stars
        for r in review_set:
            if review['stars'] == r['stars'] and review['review_id'] != r['review_id'] and review['date'] > r['date']:
                current_set.append(r)
            if len(current_set) == 11:
                break

        # Get just the texts from the current set
        texts = []
        for item in current_set:
            texts.append(item['text'])

        # Output 1 for similarity scores when less than 10 prior reviews
        if len(current_set) != 11:
            item = current_set[0]

            # Assign the review IDs compared against
            item['reviewed_against_1b'] = "none"
            item['reviewed_against_2b'] = "none"
            item['reviewed_against_3b'] = "none"
            item['reviewed_against_4b'] = "none"
            item['reviewed_against_5b'] = "none"
            item['reviewed_against_6b'] = "none"
            item['reviewed_against_7b'] = "none"
            item['reviewed_against_8b'] = "none"
            item['reviewed_against_9b'] = "none"
            item['reviewed_against_10b'] = "none"

            # Assign the similarity scores
            item['score_1b'] = 1
            item['score_2b'] = 1
            item['score_3b'] = 1
            item['score_4b'] = 1
            item['score_5b'] = 1
            item['score_6b'] = 1
            item['score_7b'] = 1
            item['score_8b'] = 1
            item['score_9b'] = 1
            item['score_10b'] = 1

            final_output.append(item)

            # print('**Less than 10 prior reviews with same valence for review ' + str(review['review_id']))
        else:
            LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
            # print("Transforming tokens into vectors of term frequency (TF)")
            LemVectorizer.fit_transform(texts)

            # print('\nThe indexes of the terms in the vector')
            # print(sorted(LemVectorizer.vocabulary_.items()))

            # print("\nConverting vectors into TF matrices")
            tf_matrix = LemVectorizer.transform(texts).toarray()
            # print(tf_matrix)

            # Confirm matrix shape (n x m) where n = reviews and m = terms
            # print(tf_matrix_1.shape)
            # print(tf_matrix_2.shape)

            # print("\nCalculating inverse document frequency (IDF) matrices")
            # Each vector's component is now the idf for each term
            tfidfTran = TfidfTransformer(norm="l2")
            tfidfTran.fit(tf_matrix)
            # print(len(tfidfTran.idf_))
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
            # print(cos_similarity_matrix)

            # Getting first column of similarity matrix - should be ten scores after stripping the first score
            first_column = cos_similarity_matrix[0]
            scores = first_column[1:]
            
            # Write to CSV file

            current = current_set[1:]

            # Assign the review IDs compared against
            review['reviewed_against_1b'] = current[0]['review_id']
            review['reviewed_against_2b'] = current[1]['review_id']
            review['reviewed_against_3b'] = current[2]['review_id']
            review['reviewed_against_4b'] = current[3]['review_id']
            review['reviewed_against_5b'] = current[4]['review_id']
            review['reviewed_against_6b'] = current[5]['review_id']
            review['reviewed_against_7b'] = current[6]['review_id']
            review['reviewed_against_8b'] = current[7]['review_id']
            review['reviewed_against_9b'] = current[8]['review_id']
            review['reviewed_against_10b'] = current[9]['review_id']

            # Assign the similarity scores
            review['score_1b'] = scores[0]
            review['score_2b'] = scores[1]
            review['score_3b'] = scores[2]
            review['score_4b'] = scores[3]
            review['score_5b'] = scores[4]
            review['score_6b'] = scores[5]
            review['score_7b'] = scores[6]
            review['score_8b'] = scores[7]
            review['score_9b'] = scores[8]
            review['score_10b'] = scores[9]

            # print('Assigning similarity scores (method b) for review ' + review['review_id'])

            final_output.append(review)
    
    return final_output

# c. For each prior review, calculate the total of “useful”, “funny” and “cool” count, 
#    and select 10 reviews with the highest counts (1, 2, 3…10). For reviews with the same count, use recency to determine the order 
#    (e.g., if two prior reviews have the same max count, the review out of these two that is more recent 
#    (relative to the focal review) should be “1”, and the other is “2”, for setting the order). 
#    Once you’ve ordered all prior reviews by that rule, record similarity scores with review 1, 2, 3…10.
#    If less than 10 reviews for the focal review, output similarity score of 1 for each remaining review
def calculate_method_c(review_set, pid):
    print('\n' + str(pid) + ': Now calculating method c...')
    final_output = []
    review_set_by_ufc = sorted(review_set, key = lambda x: x['total_ufc'], reverse=True)

    for review in review_set:

        current_set = []
        current_set.append(review)

        # Get 1 focal review + 10 prior reviews with highest number of "useful", "funny", and "cool" counts
        for r in review_set_by_ufc:
            if review['date'] > r['date'] and review['review_id'] != r['review_id']:
                current_set.append(r)
            if len(current_set) == 11:
                break

        # Get just the texts from the current set
        texts = []
        for item in current_set:
            texts.append(item['text'])

        # Output 1 for similarity scores when less than 10 prior reviews
        if len(current_set) != 11:
            for item in current_set:
                # Assign the review IDs compared against
                item['reviewed_against_1c'] = "none"
                item['reviewed_against_2c'] = "none"
                item['reviewed_against_3c'] = "none"
                item['reviewed_against_4c'] = "none"
                item['reviewed_against_5c'] = "none"
                item['reviewed_against_6c'] = "none"
                item['reviewed_against_7c'] = "none"
                item['reviewed_against_8c'] = "none"
                item['reviewed_against_9c'] = "none"
                item['reviewed_against_10c'] = "none"

                # Assign the similarity scores
                item['score_1c'] = 1
                item['score_2c'] = 1
                item['score_3c'] = 1
                item['score_4c'] = 1
                item['score_5c'] = 1
                item['score_6c'] = 1
                item['score_7c'] = 1
                item['score_8c'] = 1
                item['score_9c'] = 1
                item['score_10c'] = 1

                final_output.append(item)

            # print('**Less than 10 prior reviews with highest ufc count - finishing at review ' + str(review['review_id']))
            break
        else:
            LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
            # print("Transforming tokens into vectors of term frequency (TF)")
            LemVectorizer.fit_transform(texts)

            # print('\nThe indexes of the terms in the vector')
            # print(sorted(LemVectorizer.vocabulary_.items()))

            # print("\nConverting vectors into TF matrices")
            tf_matrix = LemVectorizer.transform(texts).toarray()
            # print(tf_matrix)

            # Confirm matrix shape (n x m) where n = reviews and m = terms
            # print(tf_matrix_1.shape)
            # print(tf_matrix_2.shape)

            # print("\nCalculating inverse document frequency (IDF) matrices")
            # Each vector's component is now the idf for each term
            tfidfTran = TfidfTransformer(norm="l2")
            tfidfTran.fit(tf_matrix)
            # print(len(tfidfTran.idf_))
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
            # print(cos_similarity_matrix)

            # Getting first column of similarity matrix - should be ten scores after stripping the first score
            first_column = cos_similarity_matrix[0]
            scores = first_column[1:]

            current = current_set[1:]

            # Assign the review IDs compared against
            review['reviewed_against_1c'] = current[0]['review_id']
            review['reviewed_against_2c'] = current[1]['review_id']
            review['reviewed_against_3c'] = current[2]['review_id']
            review['reviewed_against_4c'] = current[3]['review_id']
            review['reviewed_against_5c'] = current[4]['review_id']
            review['reviewed_against_6c'] = current[5]['review_id']
            review['reviewed_against_7c'] = current[6]['review_id']
            review['reviewed_against_8c'] = current[7]['review_id']
            review['reviewed_against_9c'] = current[8]['review_id']
            review['reviewed_against_10c'] = current[9]['review_id']

            # Assign the similarity scores
            review['score_1c'] = scores[0]
            review['score_2c'] = scores[1]
            review['score_3c'] = scores[2]
            review['score_4c'] = scores[3]
            review['score_5c'] = scores[4]
            review['score_6c'] = scores[5]
            review['score_7c'] = scores[6]
            review['score_8c'] = scores[7]
            review['score_9c'] = scores[8]
            review['score_10c'] = scores[9]

            # print('Assigning similarity scores (method c) for review ' + review['review_id'])

            final_output.append(review)

    return final_output

def start_calculations(businesses):
    """Starts the calculation process for a business in Yelp's data set

    Args:
        businesses (string): A single business  # TODO: fix this name, it's misleading

    Returns:
        string: A string saying the process has completed
    """
    global START
    global REVIEWED
    global REVIEW_SET

    current_businesses = []

    pid = os.getpid()

    START = True

    current_businesses.append(businesses)


    # Working on businesses now
    print(str(pid) + ': Process started')
    for i, b in enumerate(current_businesses):
        # print(str(pid) + ': There are ' + str(len(businesses) - i) + ' businesses left to parse through')
        current_business = b
        REVIEW_SET = []

        start = timer()

        # Getting reviews for current business
        # print('\n' + str(pid) +  ': Getting reviews for current business #' + current_business)
        with open('pittsburgh_reviews.csv', mode='r', encoding='utf-8') as input:
                csv_reader = csv.DictReader(input)
                counter = 0
                for row in csv_reader:
                    if counter == 0:
                        counter += 1
                    else:
                        if row['business_id'] == current_business:
                                row['elite'] = False
                                REVIEW_SET.append(row)
                                counter += 1

        num_reviews = len(REVIEW_SET)

        print('\n' + str(pid) +  ': There are ' + str(num_reviews) + ' reviews for current business ' + current_business)

        # Checking for elite status in reviews
        csv.field_size_limit(2147483647) # note: this may or may not cause issues...
        # print(str(pid) + ':Getting Yelp Elite status for users of reviews')
        with open('pittsburgh_users.csv', mode='r', encoding='utf-8') as input:
            counter = 0
            elites = 0
            csv_reader = csv.DictReader(input)
            for row in csv_reader:
                # This is the input file information
                user = row['user_id']
                elite_years = row['elite'].split(',')
                for i, review in enumerate(REVIEW_SET):
                    # This is the review information
                    review_user = review['user_id']
                    review_date = review['date']
                    review_year = int(review_date[0:4])
                    if user == review_user:
                        # print('User in file matches user in review set')
                        # print('These are the years of elite years')
                        # print(elite_years)
                        # print('This is the year of the review')
                        # print(str(review_year))
                        for year in elite_years:
                            if year != '':
                                # print('Year in list of elite years: ' + str(year))
                                if review_year == int(year):
                                    # print('found an elite!')
                                    REVIEW_SET[i]['elite'] = True
                                    elites += 1

        # print('There are ' + (str(elites)) + ' Yelp Elites in this set of reviews')

        # Sort review set by date
        REVIEW_SET = sorted(REVIEW_SET, key=lambda k: k['date'], reverse=True)

        # Number the reviews and get word count
        for index, review in enumerate(REVIEW_SET):
            review['chronological_index'] = len(REVIEW_SET) - index
            review['word_count'] = len(review['text'].split())
            review['total_ufc'] = int(review['useful']) + int(review['funny']) + int(review['cool'])

        # Getting just the text from the review set
        # review_texts = []
        # for review in REVIEW_SET:
        #     review_texts.append(review['text'])

        # print(sorted_review_set)
        # print(review_texts)

        # print('----------')
        # Similarity calculations 

        # Each review will produce 30 similarity scores - 10 from a, b, and c

        ###################### BY CRITERIA a, b, c #######################

        # Getting the criteria outputs

        final_output_a = calculate_method_a(REVIEW_SET, pid)

        final_output_b = calculate_method_b(REVIEW_SET, pid)

        final_output_c = calculate_method_c(REVIEW_SET, pid)
            
        # Combining the criteria outputs

        final_to_write = copy.deepcopy(final_output_a)

        # Append the b criteria scores
        for id, f in enumerate(final_to_write):
            for i in range(10):
                b_score = "score_" + str(i+1) + "b"
                b_review = "reviewed_against_" + str(i+1) + "b"
                    
                f[b_review] = final_output_b[id][b_review]
                f[b_score] = final_output_b[id][b_score]

        # Append the c criteria scores
        for id, f in enumerate(final_to_write):
            for i in range(10):
                c_score = "score_" + str(i+1) + "c"
                c_review = "reviewed_against_" + str(i+1) + "c"

                f[c_review] = final_output_c[id][c_review]
                f[c_score] = final_output_c[id][c_score]

        end = timer()

        total_time = time.strftime('%H:%M:%S', time.gmtime(end - start))

        print('\n' + str(pid) + ': Writing similarity scores to file for business ' + current_business)
        print(str(pid) + ': It took ' + str(total_time) + ' for ' + str(len(REVIEW_SET)) + ' reviews')
        with open ('pittsburgh_businesses_similarities_mp.csv', 'a', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, final_to_write[0].keys())
            if START:
                writer.writeheader()
                START = False
            for row in final_to_write:
                writer.writerow(row)
            print('\n')

            ###################### AVERAGED PRIOR REVIEWS #######################

            # for i in range(len(review_texts)):
            #     # Not comparing very first review, break after writing
            #     if i == len(review_texts) - 1:
            #         full_review_data = REVIEW_SET[i]
            #         full_review_data['average'] = 1
            #         with open ('pittsburgh_businesses_similarities.csv', 'a', encoding='utf-8', newline='') as file:
            #             writer = csv.DictWriter(file, full_review_data.keys())
            #             writer.writerow(full_review_data)
            #         break

            #     current_set = review_texts[i:]
                
            #     LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
            #     # print("Transforming tokens into vectors of term frequency (TF)")
            #     LemVectorizer.fit_transform(current_set)

            #     # print('\nThe indexes of the terms in the vector')
            #     # print(sorted(LemVectorizer.vocabulary_.items()))

            #     # print("\nConverting vectors into TF matrices")
            #     tf_matrix = LemVectorizer.transform(current_set).toarray()
            #     # print(tf_matrix)

            #     # Confirm matrix shape (n x m) where n = reviews and m = terms
            #     # print(tf_matrix_1.shape)
            #     # print(tf_matrix_2.shape)

            #     # print("\nCalculating inverse document frequency (IDF) matrices")
            #     # Each vector's component is now the idf for each term
            #     tfidfTran = TfidfTransformer(norm="l2")
            #     tfidfTran.fit(tf_matrix)
            #     # print(len(tfidfTran.idf_))
            #     # print(tfidfTran.idf_)

            #     # Manually verify that the IDF is correct
            #     # print("The idf for terms that appear in one document: " + str(idf(2,1)))
            #     # print("The idf for terms that appear in two documents: " + str(idf(2,2)))

            #     # print("\nCreating the TF-IDF matrices")
            #     # Transform method here multiples the tf matrix by the diagonal idf matrix
            #     # The method then divides the tf-idf matrix by the Euclidean norm
            #     tfidf_matrix = tfidfTran.transform(tf_matrix)
            #     # print(tfidf_matrix.toarray())

            #     # print("\nCreating the cosine similarity matrices")
            #     # Multiply matrix by transpose to get final result
            #     cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            #     # print(cos_similarity_matrix)

            #     # Getting average of cosine similarity score
            #     first_column = cos_similarity_matrix[0]
            #     first_column = first_column[1:]
            #     similarity_average = sum(first_column) / len(first_column)
            #     # print(first_column)
            #     print('\n' + str(REVIEW_SET[i]['review_id']) + ': ' + str(similarity_average))
            #     # print('Number of reviews ' + str(sorted_review_set[i]['review_id']) + ' compared to: ' + str(len(current_set)))
            #     # print('\n')

            #     # Write to CSV file
            #     # write_file = numpy.asarray(cos_similarity_matrix)
            #     # numpy.savetxt("100_businesses.csv", write_file, delimiter=",")
                
            #     full_review_data = REVIEW_SET[i]
            #     full_review_data['average'] = similarity_average
            #     with open ('pittsburgh_businesses_similarities.csv', 'a', encoding='utf-8', newline='') as file:
            #         writer = csv.DictWriter(file, full_review_data.keys())
            #         if i == 0:
            #             writer.writeheader()
            #         writer.writerow(full_review_data)
            # print('\n')

        REVIEWED += 1
        print('\n' + str(pid) + ' has now reviewed ' + str(REVIEWED) + ' businesses')

        # Add to list of already analyzed businesses
        with open('finished_pittsburgh_businesses.csv', mode='a', encoding='utf-8', newline='') as to_write:
            headers = ['business_id']
            writer = csv.DictWriter(to_write, headers)
            writer.writerow({'business_id': current_business})


    return "\nPool processing complete."


if __name__ == '__main__':
    ####################### DETERMINING YELP ELITE STATUS #######################

    # elite_users = {'2005': [], '2006': [], '2007': [], '2008': [], '2009': [], 
    #                '2010': [], '2011': [], '2012': [], '2013': [], '2014': [],
    #                '2015': [], '2016': [], '2017': [], '2018': [], '2019': []}

    # with open('yelp_data/yelp_academic_dataset_user.json', encoding='utf-8') as to_read:
    #     for row in to_read:
    #         user = json.loads(row)
    #         if len(user['elite']) > 0:
    #             years = user['elite'].split(',')
    #             for year in years:
    #                 elite_users[str(year)].append(user['user_id'])

    # user_file = "yelp_user_status.json"
    # with open(user_file, 'w', newline='') as to_write:
    #     json_object = json.dumps(elite_users, indent=4)
    #     to_write.write(json_object)

    # for year in elite_users:
    #     print(year + ' ' + str(len(elite_users[year])))

    ####################### Filtering dataset #######################

    # 1) Select all businesses reviewed by folks who had Elite status in 2018 (2018 Elite Yelpers)

    # 2) Out of these, drop all businesses that have less than 10 reviews

    # 3) Out of remaining ones, drop all businesses that do NOT have any reviews posted by 2018 Elite Yelpers during 2018.

    # elites_2018 = []
    # businesses_with_more_than_10_reviews = []
    # reviews_dict = {}

    # print('getting 2018 elite users')
    # # Get the users who were Elite in 2018
    # with open('yelp_data/yelp_user_status.json', encoding='utf-8') as elites:
    #     data = json.load(elites)
    #     elites_2018 = data['2018']

    # print('getting businesses with more than 10 reviews')
    # # Drop all businesses with less than 10 reviews
    # with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8') as businesses:
    #     counter = 0
    #     for row in businesses:
    #         business = json.loads(row)
    #         if business['review_count'] >= 10:
    #             businesses_with_more_than_10_reviews.append(business['business_id'])
    #             reviews_dict[business['business_id']] = []
    #         print('parsed ' + str(counter) + ' businesses look for more than 10 reviews')
    #         counter += 1

    # print('getting reviews for businesses')
    # # Get all reviews for businessess in filtered list
    # with open('yelp_data/yelp_academic_dataset_review.json', encoding='utf-8') as reviews:
    #     counter = 0
    #     for row in reviews:
    #         review = json.loads(row)
    #         business_id = review['business_id']
    #         if business_id in businesses_with_more_than_10_reviews:
    #             reviews_dict[business_id].append(review)
    #         print('parsed ' + str(counter) + ' reviews to add to reviews_dict')
    #         counter += 1

    # businesses_2018 = []
    
    # print('checking for business reviews from at least one 2018 elite')
    # # If the business has had at least one review from a 2018 Elite Yelper, keep the business
    # for business in businesses_with_more_than_10_reviews:
    #     reviews = reviews_dict.get(business)
    #     for review in reviews:
    #         user = review['user_id']
    #         date = review['date']
    #         if user in elites_2018 and '2018' in date:
    #             businesses_2018.append(business)
    #             print('adding ' + str(business) + ' to final list of businesses')
    #             break

    # locations = {}

    # # Get city for each business in final list
    # print('getting city count for final list of businesses')
    # with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8') as businesses:
    #     for row in businesses:
    #         business = json.loads(row)
    #         business_id = business['business_id']
    #         if business_id in businesses_2018:
    #             locations[business_id] = business['city'] 

    # # Write dict to final file
    # print('writing to output file')
    # with open('2018_elite_businesses.csv', 'w') as to_write:
    #     writer = csv.writer(to_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for key in locations:
    #         value = locations.get(key)
    #         writer.writerow([key, value])  

    # cities = []

    # print('\nPutting total number of cities into list')
    # with open('2018_elite_businesses.csv', 'r') as businesses:
    #     reader = csv.reader(businesses, delimiter=',')
    #     for row in reader:
    #         cities.append(row[1])

    # city_count = Counter(cities)

    # print('\nWriting total number of cities into CSV')
    # with open('2018_elite_businesses_city_count.csv', 'w') as cities:
    #     writer = csv.writer(cities, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for key in city_count:
    #         value = city_count.get(key)
    #         writer.writerow([key, value])

    # print('\nTotal number of reviews parsed: 8021122')
    # print('Yelp Elites in 2018: ' + str(len(elites_2018)))
    # print('Businesses with more than 10 reviews: ' + str(len(businesses_with_more_than_10_reviews)))  
    # print('Businesses with reviews from at least one 2018 Yelp Elite in 2018: ' + str(len(businesses_2018)))  


    ####################### PARSING AND SET-UP #######################
    print('-----------------------------------')
    print("Starting analysis script...")
    # file = sys.argv[1]
    # business_id = sys.argv[2]
    # print("Parsing sample file " + str(file))
    # parse_txt_file(file)
    # parse_csv_file(file)

    # All reviews pertaining to the business ID will be stored in ALL_REVIEWS
    # Reviews are sorted chronologically
    # parse_yelp_json_file(file, business_id) 
    # print (ALL_REVIEWS)
    # print('\n')

    

    ###################### SIMILARITY CALCULATIONS #######################

    REVIEW_SET = []
    START = True
    businesses = []

    # Getting businesses to work on
    print('\nGetting all businesses to work on')
    with open('pittsburgh_businesses.csv', mode='r', encoding='utf-8', newline='') as input:
        counter = 0
        for row in input:
            # skipping header row
            if counter == 0:
                counter +=1
            else:
                businesses.append(row.split(',')[0].rstrip())
                counter += 1

    print('\nTrimming businesses that we have already reviewed')
    analyzed = []
    with open('finished_pittsburgh_businesses.csv', mode='r', encoding='utf-8') as input:
        counter = 0
        for row in input:
            # skipping header row
            if counter == 0:
                counter += 1
            else:
                analyzed.append(row.split(',')[0].rstrip())
                counter += 1

    businesses = [business for business in businesses if business not in analyzed]

    print('\nThere are ' + str(len(businesses)) + ' in total')

    time.sleep(3)

    ###################### MAIN LOOP #######################

    # start_calculations(businesses)

    ###################### MULTIPROCESSING #######################

    print('\nStarting pool processes...')
    with Pool(processes=4) as pool:
        results = pool.map(start_calculations, businesses) 
        pool.close()
        pool.join()







    print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
    sys.exit(0)    