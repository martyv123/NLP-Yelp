#!/usr/bin/python3
#
# Ultilities script for various functions related to NLP processing of Yelp dataset.
#
# Maintainers: vo.ma@northeastern.edu

import csv
import json
import sys
import math

# Getting length of reviews for already calculated cosine similarity averages
def get_review_length():
    print('Getting length of reviews in file')
    text_lengths = []
    with open('subset_calculations/120_businesses_50_reviews_similarities.csv', mode='r', encoding='utf-8') as input:
        reader = csv.DictReader(input)
        for row in reader:
            review_id = row['review_id']
            review_length = len(row['text'].split())
            text_lengths.append({'review_id': review_id, 'review_length': review_length})
    print('Writing lengths of review to output file')
    with open('temp_review_lengths.csv', mode='w', encoding='utf-8') as output:
        headers = ['review_id', 'review_length']
        writer = csv.DictWriter(output, headers)
        writer.writerows(text_lengths)

# Get businesses with x reviews
def get_businesses_with_x_reviews(reviews):
    print('Getting businesses with ' + str(reviews) + ' reviews')
    with open('2018_elite_businesses.csv', mode='r') as input:
        reader = csv.DictReader(input)
        for row in reader:
            business = row['business_id']
            with open('yelp_data/yelp_academic_dataset_business.json', mode='r', encoding='utf-8') as businesses:
                for item in businesses:
                    data = json.loads(item)
                    keys = data.keys()
                    if business == data['business_id'] and data['review_count'] <= reviews:
                        with open('2018_elite_businesses_50_reviews.csv', mode='a', encoding='utf-8') as output:
                            writer = csv.DictWriter(output, keys)
                            writer.writerow(data)



################################### MAIN ###################################
if __name__ == '__main__':
    get_businesses_with_x_reviews(50)
    print('Utilities functions complete')