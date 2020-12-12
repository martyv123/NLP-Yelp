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
  * We can represent documents as vectors of features
  * We can compare documents by measuring the distance between these features
* What are word embeddings?
  * Embeddings are capable of capturing the context of a word in a document
    * This is good for identifying contextual content
    * On the other hand, TF-IDF is good for classifying documents as a whole
  * What processes are available to us?
    * _Jaccard Similarity_ - focuses on the the size of intersection divided by size of union of two sets
      * Not very useful for large documents
    * _K-means and Hierarchical Clustering Dendrogram_ - convert sentences into vectors
      * Does not offer a visible improvement in comparison in the Silhouette Coefficient
        * This is due to the high dimensional nature that text data has
        * https://en.wikipedia.org/wiki/Silhouette_(clustering) 
    * _Cosine Similarity_ - measures the cosine of the angle between two vectors
      * Offers a measure of similarity between two non-zero vectors: i.e., orientation and not magnitude
      * Advantageous because two documents that are far apart in Euclidean distance may still be oriented closer together
      * Still not the best, as it cannot determine semantic meaning very well
      * https://en.wikipedia.org/wiki/Cosine_similarity 
* It appears that the best processes rely on a combination of using word embeddings and some form of distance calculation
  * What works best?
    * The author rates the usage of several, most notably the use of embeddings compounded with the Siamese Manhattan LTSM
    * Runner-ups:
      * Embeddings + Universal sentencer encoder
        * https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15
      * Embeddings + Variational Auto Encoder (VAE)
        * https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
      * Embeddings + Word Mover Distance
        * https://towardsdatascience.com/word-distance-between-word-embeddings-cc3e9cf1d632
* There is a much more in-depth statisical analysis provided on the author's blog.

<br>
<br>
<br>

## Semantic Similarity in Sentences and BERT
https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677
