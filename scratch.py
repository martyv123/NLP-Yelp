import csv


businesses = set()

with open('open_table/las_vegas.csv', mode='r') as input:
    csv_reader = csv.reader(input, delimiter=',')
    for row in csv_reader:
        businesses.add(row[2])
        
with open('open_table/las_vegas_businesses.csv', mode='a', newline='') as output:
    csv_writer = csv.writer(output, delimiter=',')
    csv_writer.writerow(['business'])
    for b in businesses:
        csv_writer.writerow([b])
