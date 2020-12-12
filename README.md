# Text Frequency Analysis


#### Marty Vo
#### vo.ma@northeastern.edu
#### Last updated: 12/12/2020

<br>

This is a minimal script that reads in the ten text review examples provided by Prof. Bart. The
script will parse, lemmatize, and perform a term frequency (TF) analysis on the reviews
provided in the sample.txt file. More specifically, it performs a TF-IDF analysis, which stands
for Term Frequency (TF) - Inverse Document Frequency (IDF). We will then use the weighted values of
the terms in the reviews to compare the similarity between reviews by using cosine similarity. As we 
continue to analyze larger data sets, modifications will be added to script as necessary.

The cosine similarity is the cosine of the angle between two vectors. In text analysis, each vector can
represent a document (in our case a review). The greater the value of θ, the less the value of cos θ,
thus the less the similarity between two document.


# Current Implementation Details 

Contractions were parsed into their full-length counterparts using a dictionary. For example, 
"don't" would be parsed as "do not". These contractions are evaluated using a map of their
definitions from Wikipedia. For contractions where there can be several meanings, such as "he'd"
which could translate into "he had" or "he would", the script does not yet handle such conversions. 
An error is raised which will stop the analysis. 

In future verions of the script, I will look at more in-depth processing to determine how these
contractions should be converted. Likely, it will depend on the context of the surrounding words
of the contraction. It may also be possible that we decide to just use contractions in the 
lemmatized lists as to determine word frequency afterall.

Stopwords, like 'the' or 'of' were not removed from the reviews during parsing. In the future,
we may find it more useful to remove the stopwords before processing reviews. It is possible that 
stopwords add a needed context to the review that may help in determining their differences.
There are text processing implementations that both remove or keep stopwords. It is largely project
specific if there is a need to do so. 

<br>

# Research

Research regarding methods to determine text-similarity can be found [here](https://github.com/martyv123/NLP-Yelp/blob/main/RESEARCH.MD)

<br>

# Current & Other resources

https://medium.com/@adriensieg/text-similarities-da019229c894

https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677

https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/

https://sites.temple.edu/tudsc/2017/03/30/measuring-similarity-between-texts-in-python/