# WordEmbeddings-ELMo, Fasttext, FastText (Gensim) and Word2Vec

This implementation gives the flexibility of choosing word embeddings on your corpus. One has the option of choosing word Embeddings from ELMo (https://arxiv.org/pdf/1802.05365.pdf) - recently introduced by Allennlp and these word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. Also fastext embeddings (https://arxiv.org/pdf/1712.09405.pdf) published in LREC from Thomas Mikolov and team is available. 
ELMo embeddings outperformed the Fastext, Glove and Word2Vec on an average by 2~2.5% on a simple Imdb sentiment classification task (Keras Dataset). 

### USAGE:

To run it on the Imdb dataset, 

	run: python main.py

To run it on your data: comment out line 32-40 and uncomment 41-53


### FILES:
* word_embeddings.py – contains all the functions for embedding and choosing which word embedding model you want to choose.
* config.json – you can mention all your parameters here (embedding dimension, maxlen for padding, etc)
* model_params.json - you can mention all your model parameters here (epochs, batch size etc.)
* main.py – This is the main file. Just use this file to run in terminal.
 
You have the option of choosing the word vector model

In **config.json** specify “option” as  0 – Word2vec, 1 – Gensim FastText, 2- Fasttext (FAIR), 3- ELMo


The model is very generic. You can change your model as per your requirements. 

Feel free to reach out in case you need any help.

Special thanks to Jacob Zweig for the write up: https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440. Its a good 2 min read.
