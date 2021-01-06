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
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

ALL_REVIEWS = []
REVIEW_SET = [] 

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
    # filtered_businesses = []
    # filtered_reviews = {}

    # print('getting 2018 elite users')
    # # Get the users who were Elite in 2018
    # with open('yelp_data/yelp_user_status.json', encoding='utf-8') as elites:
    #     data = json.load(elites)
    #     elites_2018 = data['2018']

    # print('getting businesses with more than 10 reviews')
    # # Drop all businesses with less than 10 reviews
    # with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8') as businesses:
    #     for row in businesses:
    #         business = json.loads(row)
    #         if business['review_count'] >= 10:
    #             filtered_businesses.append(business['business_id'])

    # print('getting reviews for businesses')

    # for business in filtered_businesses:
    #     filtered_reviews[business] = []

    # # Get all reviews for businessess in filtered list
    # with open('yelp_data/yelp_academic_dataset_review.json', encoding='utf-8') as reviews:
    #     for row in reviews:
    #         review = json.loads(row)
    #         business_id = review['business_id']
    #         if business_id in filtered_businesses:
    #             filtered_reviews[business_id].append(review)
    #             print('review id: ' + review['review_id'])

    # filtered_reviews_2018 = {}
    
    # print('checking for business reviews from at least one 2018 elite')
    # # If the business has had at least one review from a 2018 Elite Yelper, keep the business
    # for business in filtered_reviews:
    #     reviews = filtered_reviews.get(business)
    #     for review in reviews:
    #         user = review['user_id']
    #         if user in elites_2018:
    #             filtered_reviews_2018[business] = reviews
    #             print('review id: ' + review['review_id'])
    #             break

    # final_reviews = {}

    # print('removing business reviews without any 2018 Yelp Elite posts')
    # # If the business has not had a review in 2018 from a Yelp Elite, remove the business
    # for business in filtered_reviews_2018:
    #     reviews = filtered_reviews.get(business)
    #     for review in reviews:
    #         date = review['date']
    #         user = review['user_id']
    #         if '2018' in date:
    #             if user in elites_2018:
    #                 final_reviews[business] = reviews
    #                 print('review id: ' + review['review_id'])
    #                 break

    # locations = {}

    # # Get city for each business in final list
    # print('getting city count for final list of businesses')
    # with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8') as businesses:
    #     for row in businesses:
    #         business = json.loads(row)
    #         business_id = business['business_id']
    #         if business_id in final_reviews:
    #             locations[business_id] = business['city'] 

    # # Write dict to final file
    # print('writing to output file')
    # with open('2018_elite_businesses.csv', 'w') as to_write:
    #     writer = csv.writer(to_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for key in locations:
    #         value = locations.get(key)
    #         writer.writerow([key, value])
            

    # print('\nTotal number of reviews parsed: 8021122')
    # print('Yelp Elites in 2018: ' + str(len(elites_2018)))
    # print('Businesses with more than 10 reviews: ' + str(len(filtered_businesses)))
    # print('Businesses with reviews from at least one 2018 Yelp Elite: ' + str(len(filtered_reviews_2018)))  
    # print('Businesses with reviews from at least one 2018 Yelp Elite in 2018: ' + str(len(final_reviews)))     


    # user_file = "yelp_user_status.json"
    # with open(user_file, 'w', newline='') as to_write:
    #     json_object = json.dumps(elite_users, indent=4)
    #     to_write.write(json_object)

    # cities = []

    # print('\nPutting total number of cities into list')
    # with open('2018_elite_businesses.csv', 'r') as businesses:
    #     reader = csv.reader(businesses, delimiter=',')
    #     for row in reader:
    #         cities.append(row[1])

    # city_count = Counter(cities)

    # print('\nWriting total number of cities into CSV')
    # with open('2018_city_count.csv', 'w') as cities:
    #     writer = csv.writer(cities, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for key in city_count:
    #         value = city_count.get(key)
    #         writer.writerow([key, value])

    ####################### CONVERTING JSON TO CSV #######################       

    # print('Writing business CSVs')
    # with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8', mode='r') as fin:
    #     first = fin.readlines(1)
    #     business_keys = json.loads(first[0]).keys()
    #     with open('yelp_business.csv', encoding='utf-8', mode='w') as fout:
    #         writer = csv.DictWriter(fout, business_keys)
    #         writer.writeheader()
    #         for line in fin:
    #             serialized_line = json.loads(line)
    #             writer.writerow(serialized_line)

    # print('Writing users CSVs')
    # with open('yelp_data/yelp_academic_dataset_user.json', encoding='utf-8', mode='r') as fin:
    #     first = fin.readlines(1)
    #     user_keys = json.loads(first[0]).keys()
    #     with open('yelp_user.csv', encoding='utf-8', mode='w') as fout:
    #         writer = csv.DictWriter(fout, user_keys)
    #         writer.writeheader()
    #         for line in fin:
    #             serialized_line = json.loads(line)
    #             writer.writerow(serialized_line)

    # print('Writing reviews CSVs')
    # with open('yelp_data/yelp_academic_dataset_review.json', encoding='utf-8', mode='r') as fin:
    #     first = fin.readlines(1)
    #     review_keys = json.loads(first[0]).keys()
    #     with open('yelp_review.csv', encoding='utf-8', mode='w') as fout:
    #         writer = csv.DictWriter(fout, review_keys)
    #         writer.writeheader()
    #         for line in fin:
    #             serialized_line = json.loads(line)
    #             writer.writerow(serialized_line)

    # print('Writing checkin CSVs')
    # with open('yelp_data/yelp_academic_dataset_checkin.json', encoding='utf-8', mode='r') as fin:
    #     first = fin.readlines(1)
    #     checkin_keys = json.loads(first[0]).keys()
    #     with open('yelp_checkin.csv', encoding='utf-8', mode='w') as fout:
    #         writer = csv.DictWriter(fout, checkin_keys)
    #         writer.writeheader()
    #         for line in fin:
    #             serialized_line = json.loads(line)
    #             writer.writerow(serialized_line)

    # print('Writing tip CSVs')
    # with open('yelp_data/yelp_academic_dataset_tip.json', encoding='utf-8', mode='r') as fin:
    #     first = fin.readlines(1)
    #     tip_keys = json.loads(first[0]).keys()
    #     with open('yelp_tip.csv', encoding='utf-8', mode='w') as fout:
    #         writer = csv.DictWriter(fout, tip_keys)
    #         writer.writeheader()
    #         for line in fin:
    #             serialized_line = json.loads(line)
    #             writer.writerow(serialized_line)


    ####################### PARSING AND SET-UP #######################
    print('-----------------------------------')
    print("Starting sample analysis script...")
    file = sys.argv[1]
    business_id = sys.argv[2]
    print("Parsing sample file " + str(file))
    # parse_txt_file(file)
    # parse_csv_file(file)

    # All reviews pertaining to the business ID will be stored in ALL_REVIEWS
    # Reviews are sorted chronologically
    # parse_yelp_json_file(file, business_id) 
    # print (ALL_REVIEWS)
    # print('\n')

    

    ###################### SIMILARITY CALCULATIONS #######################

    print(REVIEW_SET)
    print('\n')

    # Retrieving just the text from reviews
    text_reviews = []
    for review in ALL_REVIEWS:
        text = review['text']
        text_reviews.append(text)

    id = 0
    # print(text_reviews)
    for count in range(len(text_reviews)):
        if id + 5 > len(text_reviews):
            REVIEW_SET = text_reviews[id:]
        else:
            REVIEW_SET= text_reviews[id:id + 5]

        # print('starting index ' + str(id))
        # print(REVIEW_SET)
        # print(len(REVIEW_SET))
        # print ('\n')



        REVIEW_SET = ['Elise ate a salad for lunch on Friday.', 'John skipped lunch and went for a walk instead.']



    
        LemVectorizer = CountVectorizer(tokenizer=lem_normalize, stop_words='english')
        print("Transforming tokens into vectors of term frequency (TF)")
        LemVectorizer.fit_transform(REVIEW_SET)

        # The indexes of the terms in the vector
        # print(LemVectorizer.vocabulary_)

        print("\nConvert vectors into TF matrices")
        tf_matrix = LemVectorizer.transform(REVIEW_SET).toarray()
        # print(tf_matrix)

        # Confirm matrix shape (n x m) where n = reviews and m = terms
        # print(tf_matrix_1.shape)
        # print(tf_matrix_2.shape)

        print("\nCalculate inverse document frequency (IDF) matrices")
        # Each vector's component is now the idf for each term
        tfidfTran = TfidfTransformer(norm="l2")
        tfidfTran.fit(tf_matrix)
        # print(tfidfTran.idf_)

        # Manually verify that the IDF is correct
        # print("The idf for terms that appear in one document: " + str(idf(2,1)))
        # print("The idf for terms that appear in two documents: " + str(idf(2,2)))

        print("Get the TF-IDF matrices")
        # Transform method here multiples the tf matrix by the diagonal idf matrix
        # The method then divides the tf-idf matrix by the Euclidean norm
        tfidf_matrix = tfidfTran.transform(tf_matrix)
        # print(tfidf_matrix.toarray())

        print("Get the cosine similarity matrices\n")
        # Multiply matrix by transpose to get final result
        cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        print(cos_similarity_matrix)

        # Write to CSV file
        # group = numpy.asarray(cos_similarity_matrix)

        # write_file = group
        # numpy.savetxt("yelp_sample.csv", write_file, delimiter=",")     


    print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
    sys.exit(0)    