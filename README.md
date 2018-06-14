# WordEmbeddings-Fasttext, FastText (Gensim) and Word2Vec

This implementation gives the flexibility of choosing word embeddings on your corpus. One has the option of choosing word Embeddings from FAIR fastext (https://arxiv.org/pdf/1712.09405.pdf) published in LREC from Thomas Mikolov and team. Fasttext embeddings outperformed the Glove and Word2Vec on an average by 2~2.5% on a simple Imdb sentiment classification task (Keras Dataset). 

USAGE:
To run it on the Imdb dataset, run:python main.py
To run it on your data: comment out line 32-40 and uncomment 41-53


FILES:
word_embeddings.py – contains all the functions for embedding and choosing which word embedding model you want to choose.
config.json – you can mention all your parameters here (embedding dimension, maxlen for padding, etc)
model_params.json - you can mention all your model parameters here (epochs, batch size etc.)
main.py – This is the main file. Just use this file to run in terminal.
 
You have the option of :
Choosing the word vector model - In config.json specify “option” as  0 – Word2vec, 1 – Gensim FastText, 2- Fasttext (FAIR)
