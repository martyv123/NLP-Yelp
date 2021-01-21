import csv
import sys

# csv.field_size_limit(2147483647)

# with open('pittsburgh_users.csv', mode='r', encoding='utf-8') as input:
#     csv_reader = csv.DictReader(input)
#     for row in csv_reader:
#         print(type(row['elite']))


# example = ['a', 'b', 'c', 'd', 'e']

# print(example[0:5])

# print([1 for i in range(10)])

example = []

example.append({'date': 2010, 'total_ufc': 0})
example.append({'date': 2011, 'total_ufc': 15})
example.append({'date': 2012, 'total_ufc': 18})
example.append({'date': 2013, 'total_ufc': 18})
example.append({'date': 2014, 'total_ufc': 3})
example.append({'date': 2015, 'total_ufc': 1})
example.append({'date': 2015, 'total_ufc': 45})

example = sorted(example, key = lambda x: (x['date'], x['total_ufc']), reverse=True)

print(example)

