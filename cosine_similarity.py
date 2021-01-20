#!/usr/bin/python3
#
# Sample parsing, lemmatization, TF analysis, and cosine similarity on multiple review groups.
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
    businesses = []
    businesses_review_count = []

    # Getting businesses to work on
    print('\nGetting businesses to work on')
    with open('pittsburgh_businesses.csv', mode='r', encoding='utf-8') as input:
        counter = 0
        for row in input:
            # skipping header row
            if counter == 0:
                counter +=1
            else:
                businesses.append(row.split(',')[0])
                businesses_review_count.append({'business_id': row.split(',')[0], 'reviews': 0})
                counter += 1

    # Working on businesses now
    print('Starting calculations')
    for i in range(len(businesses)):
        print('There are ' + str(len(businesses) - i) + ' businesses left to parse through')
        current_business = businesses[i]
        REVIEW_SET = []

        # Getting reviews for current business
        print('\nGetting reviews for current business #' + str(i))
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

        # Get actual review count for businesses
        print('Getting actual review count for this businesses')
        for business in businesses_review_count:
            if business['business_id'] == current_business:
                business['reviews'] = num_reviews


        # Write review count to file
        print('Writing review counts to file')
        with open('pittsburgh_businesses_review_count.csv', mode='a', encoding='utf-8', newline='') as to_write:
            headers = ['business_id', 'reviews']
            writer = csv.DictWriter(to_write, headers)
            for business in businesses_review_count:
                if business['business_id'] == current_business:
                    writer.writerow(business)

        # Comparing review counts
        print('\nComparing review counts')
        with open('yelp_data/yelp_academic_dataset_business.json', mode='r', encoding='utf-8') as input:
            for row in input:
                data = json.loads(row)
                if data['business_id'] == current_business:
                    print('yelp says ' + str(data['review_count']) + ' reviews ')
                    print('I found ' + str(len(REVIEW_SET)) + ' reviews')

        # Checking for elite status in reviews
        csv.field_size_limit(2147483647) # note: this may or may not cause issues...
        print('\nGetting Yelp Elite status for users of reviews')
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

        print('There are ' + (str(elites)) + ' Yelp Elites in this set of reviews')

        # Sort review set by date
        REVIEW_SET = sorted(REVIEW_SET, key=lambda k: k['date'], reverse=True)

        # Number the reviews and get word count
        for index, review in enumerate(REVIEW_SET):
            review['chronological_index'] = len(REVIEW_SET) - index
            review['word_count'] = len(review['text'].split())

        # Getting just the text from the review set
        review_texts = []
        for review in REVIEW_SET:
            review_texts.append(review['text'])

        # print(sorted_review_set)
        # print(review_texts)

        print('\n-----')
        # Similarity calculations 
        for i in range(len(review_texts)):
            # Not comparing very first review, break after writing
            if i == len(review_texts) - 1:
                full_review_data = REVIEW_SET[i]
                full_review_data['average'] = 1
                with open ('pittsburgh_businesses_similarities.csv', 'a', encoding='utf-8', newline='') as file:
                    writer = csv.DictWriter(file, full_review_data.keys())
                    writer.writerow(full_review_data)
                break

            current_set = review_texts[i:]
            
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

            # Getting average of cosine similarity score
            first_column = cos_similarity_matrix[0]
            first_column = first_column[1:]
            similarity_average = sum(first_column) / len(first_column)
            # print(first_column)
            print('\n' + str(REVIEW_SET[i]['review_id']) + ': ' + str(similarity_average))
            # print('Number of reviews ' + str(sorted_review_set[i]['review_id']) + ' compared to: ' + str(len(current_set)))
            # print('\n')

            # Write to CSV file
            # write_file = numpy.asarray(cos_similarity_matrix)
            # numpy.savetxt("100_businesses.csv", write_file, delimiter=",")
             
            full_review_data = REVIEW_SET[i]
            full_review_data['average'] = similarity_average
            with open ('pittsburgh_businesses_similarities.csv', 'a', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, full_review_data.keys())
                if i == 0:
                    writer.writeheader()
                writer.writerow(full_review_data)

    print('\nTF-IDF analysis and Cosine Similarity calculation complete.')
    sys.exit(0)    