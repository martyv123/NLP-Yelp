#!/usr/bin/python3
#
# Ultilities script for various functions related to NLP processing of Yelp dataset.
#
# Maintainers: vo.ma@northeastern.edu

import csv
import json
import sys
import math
import sqlite3
import pandas as pd
from sqlite3 import Error

    ####################### CONVERTING JSON TO CSV #######################       

# print('Adding business to businesses list')
# businesses = []
# with open('yelp_data/yelp_academic_dataset_business.json', encoding='utf-8', mode='r') as input:
#     counter = 0
#     for line in input:
#         if counter in [1000, 20000, 30000, 50000, 70000, 100000, 150000, 200000]:
#             print('writing record number ' + str(counter))
#         data = json.loads(line)
#         businesses.append(data)
#         counter += 1

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

################################### UTILITY FUNCTIONS ###################################

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

# Put CSV file into DB file
def convert_all_businesses_to_db():
        file = 'yelp_data/yelp_academic_dataset_business.json'
        print('Converting JSON to DB file')
        connection = sqlite3.connect('yelp_business.db')
        cursor = connection.cursor()

        with open(file, mode='r', encoding='utf-8') as input:
            no_records = 0
            cursor.execute("DROP TABLE IF EXISTS all_businesses;")
            cursor.execute("CREATE TABLE IF NOT EXISTS all_businesses (business_id TEXT PRIMARY KEY NOT NULL, " +
                                                                      "name TEXT, " +
                                                                      "address TEXT, " +
                                                                      "city TEXT, " + 
                                                                      "state TEXT, " +
                                                                      "postal_code TEXT, " +
                                                                      "latitude REAL, " +
                                                                      "longitude REAL, " +
                                                                      "stars REAL, " +
                                                                      "review_count INTEGER, " +
                                                                      "is_open INTEGER, " +
                                                                      "attributes TEXT, " +
                                                                      "categories TEXT, " +
                                                                      "hours TEXT);")

            for row in input:
                data = json.loads(row)
                items = []
                items.append(data['business_id'])
                items.append(data['name'])
                items.append(data['address'])
                items.append(data['city'])
                items.append(data['state'])
                items.append(data['postal_code'])
                items.append(data['latitude'])
                items.append(data['longitude'])
                items.append(data['stars'])
                items.append(data['review_count'])
                items.append(data['is_open'])
                items.append(str(data['attributes']))
                items.append(str(data['categories']))
                items.append(str(data['hours']))
                cursor.execute("INSERT INTO all_businesses(business_id, name, address, city, state, \
                                                postal_code, latitude, longitude, stars, review_count, \
                                                is_open, attributes, categories, hours) \
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(items))
                connection.commit()
                if no_records in [5000, 10000, 40000, 50000, 100000, 150000, 200000]:
                    print('currently at record ' + str(no_records))
                no_records += 1

        connection.close()
        print('Conversion done')

# TODO: put reviews into db
def convert_reviews_to_db():
    

# Put CSV file into DB file
def convert_businesses_2018_to_db():
        file = '2018_elite_businesses.csv'
        print('Converting CSV to DB file')
        connection = sqlite3.connect('yelp_business.db')
        cursor = connection.cursor()

        with open(file, mode='r') as input:
            no_records = 0
            cursor.execute("DROP TABLE IF EXISTS businesses_2018;")
            cursor.execute("CREATE TABLE IF NOT EXISTS businesses_2018 (business_id TEXT PRIMARY KEY NOT NULL);")

            for row in input:
                if no_records > 0:
                    items = row.split(",")[0]
                    item = (items,)
                    cursor.execute("INSERT INTO businesses_2018(business_id) VALUES (?)", item)
                    connection.commit()
                    if no_records in [5000, 10000, 40000, 50000, 100000, 150000, 200000]:
                        print('currently at record ' + str(no_records))
                no_records += 1

        connection.close()
        print('Conversion done')



################################### CONNECTING TO DB ###################################

# Creating connection
conn = None
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)

# Selecting tasks
def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")

    rows = cur.fetchall()

    for row in rows:
        print(row)



################################### MAIN ###################################
if __name__ == '__main__':
    # get_businesses_with_x_reviews(50)
    # convert_all_businesses_to_db()
    # convert_businesses_2018_to_db()
    print('Utilities functions complete')