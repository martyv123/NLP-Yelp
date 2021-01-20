import csv
import sys

csv.field_size_limit(2147483647)

with open('pittsburgh_users.csv', mode='r', encoding='utf-8') as input:
    csv_reader = csv.DictReader(input)
    for row in csv_reader:
        print(row['elite'])