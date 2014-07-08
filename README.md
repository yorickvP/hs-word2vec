About
-----
This is the code that I wrote for the task at http://www.haskellers.com/jobs/61 , where the goal
is to use the methods from [a recent paper by Mikolov et al.](http://arxiv.org/abs/1309.4168),
to generate a graph describing word similarity, based on context, by turning the words into
vector representations, and running PCA on the result.

Instructions
------------
```sh
cabal install --dependencies
cabal build
cabal run -- scrape http://tinyurl.com/mczfmz9
cabal run makeCorpus
# optional, improve quality and better progress reports
mv corpus.txt corpus_unshuffled.txt
sort --random-sort corpus_unshuffled.txt > corpus.txt

cabal run train
cabal run -- plot outwords.txt --limit 110 --filter 3.5
# feh pca.png
# gnuplot # and enter:
#  plot "plot1.dat" using 2:3:1 with labels
```


Output files
------------------
 * `pca.png`: the main output file, a 1024x1024 graph with the amount of words in the --limit
 * `plot1.dat`: the file used as input for gnuplot, see the instructions for a way to run gnuplot manually
 * `error.png`: a graph of the error function over the number of iterations
 * `outwords.txt`: the vector representations of all the words (sorted descending by frequency)
 * `search.txt`: a cache of the search results from the arxiv search
 * `corpus.txt`: outputted by makeCorpus, each line contains a sentence
 * `pdfs/*.pdf`: the downloaded pdfs by the scraping.

Notes
-----
Sadly, it seems that the amount of data isn't enough to generate a word representation that is quite as nice as the one that can be obtained from running `plot --binary` on the (gunzipped) [sample vector file from the paper authors.](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
For some more design rationale, please see observations.txt
