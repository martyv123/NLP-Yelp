# Sample Text Frequency Analysis


#### Marty Vo
#### vo.ma@northeastern.edu
#### 12/12/2020

<br>

This is a minimal script that reads in the ten text review examples provided by Prof. Bart. The
script will parse, lemmatize, and perform a term frequency (TF) analysis on the reviews
provided in the sample.txt file. More specifically, it performs a TF-IDF analysis, which stands
for Term Frequency (TF) - Inverse Document Frequency (IDF). As we continue to analyze larger
data sets, I will make modifications to the script as necessary.



Note: Contractions were parsed into their full-length counterparts. For example, "don't" would
be parsed as "do not". These contractions are evaluated using a map of their definitions from 
Wikipedia. For contractions where there can be several meanings, such as "he'd" which could
translate into "he had" or "he would", the script does not yet handle such conversions. An error
is raised which will stop the analysis. 

In future verions of the script, I will look at more in-depth processing to determine how these
contractions should be converted. Likely, it will depend on the context of the surrounding words
of the contraction. It may also be possible that we decide to just use contractions in the 
lemmatized lists as to determine word frequency afterall.

<br>

# Research
## Text Similarities: Estimate the degree of similarity between two texts

https://medium.com/@adriensieg/text-similarities-da019229c894

* What is text similarity?
  * Determines how 'close' two pieces of text are in lexical similarity and semantic similarity
    * Example: "the cat ate the mouse" and "the mouse ate the cat food"
    * Lexically, the two phrases are very similar due to the words used
    * Semantically, the two phrases are very different
  * There is a dependency structure in any sentence:
    * Differences in word order create a difference in meaning
* What is our winning strategy?
  * 

<br>

## Semantic Similarity in Sentences and BERT
https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677

<br>

# Current resources

https://medium.com/@adriensieg/text-similarities-da019229c894

https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677

https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

