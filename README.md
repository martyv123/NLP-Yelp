# Text Similarity Analysis


### Dr. Ann Kronrod
### ann_kronrod@uml.edu

### Dr. Bart Yakov
### y.bart@northeastern.edu

#### Marty Vo
#### vo.ma@northeastern.edu

#### Last updated: 12/12/2020

<br>

# Implementation Details 

This is a script that works with the dataset provided by Yelp to calcualte the textual similarity between reviews.
The script will parse, lemmatize, and perform a term frequency (TF) analysis on the reviews
provided in the sample.txt file. More specifically, it performs a TF-IDF analysis, which stands
for Term Frequency (TF) - Inverse Document Frequency (IDF). We then use the weighted values of
the terms in the reviews to compare the similarity between reviews by using cosine similarity. As we 
continue to analyze larger data sets, modifications will be added to script as necessary.

This script parses the sample review data and lemmatizes the individual reviews by creating tokens
for each term. Stopwords, such as "to", "the", and others were removed from the lemmatized list of
reviews. Punctuation was also removed from the lemmatized lists as it is not useful in determining
the relevance of a term in a document. Parsing and lemmatization was handled by the nltk library
which can be found [here](https://www.nltk.org/).

The term frequency, inverse document frequency, cosine similarity, and all intermediate matrix 
transformations were calculated using scikit-learn's library. You can find more information about the 
calculations and methods [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text).

This script will return the ouputted cosine similarity values to an output file. There were several criteria used to decide
which reviews to compare against each other. One method analyzed a focal review against the ten previous chronological reviews.
The second method analyzed a focal review against the the ten previous chronological reviews with the same number of stars.
The third method analyzed a focal review against the ten previous chronological reviews with the same number of accolades.

Support for OpenTable reviews can be found in the `open_table.py` script. This analyzes reviews from OpenTable in a similar fashion.
T

<br>

# Research

Research regarding methods to determine text-similarity can be found [here](https://github.com/martyv123/NLP-Yelp/blob/main/RESEARCH.MD)

<br>

# Citation

The basis for this script was taken from the article below:
* https://sites.temple.edu/tudsc/2017/03/30/measuring-similarity-between-texts-in-python/
  

Other resources:

* https://medium.com/@adriensieg/text-similarities-da019229c894

* https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677

* https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

* https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

* https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/

