import json
import numpy as np
from sklearn.model_selection import train_test_split
from word_embeddings import load_data,prepare_data_for_word_vectors,building_word_vector_model,\
classification_model,padding_input,prepare_data_for_word_vectors_imdb,ELMoEmbedding,data_prep_ELMo,Classification_model_with_ELMo


def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set

with open("config.json","r") as f:
    params_set = json.load(f)
params_set = json_to_dict(params_set)


with open("model_params.json", "r") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)

'''
    load_data function works on imdb data. In order to load your data, comment line 27 and pass your data in the form of X,y
    X = text data column
    y = label column(0,1 etc)

'''
# for imdb data
if params_set["option"]in [0,1,2]:
    x_train,x_test,y_train,y_test = load_data(params_set["vocab_size"],params_set["max_len"])
    sentences,word_ix = prepare_data_for_word_vectors_imdb(x_train)
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y_train)

    # for other data:
    # put your data in the form of X,y
    '''
    X = ["this is a sentence","this is another sentence by me","yet another sentence for training","one more again"]
    y=np.array([0,1,1,0])

    sentences_as_words,sentences,word_ix = prepare_data_for_word_vectors(X)
    print("sentences loaded")
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)


    print("word vector model built")
    x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=params_set["split_ratio"], random_state=42)
    print("Data split done")
    '''
    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])


    model = classification_model(params_set["embed_dim"],x_train_pad,x_test_pad,y_train,y_test,
                         params_set["vocab_size"],word_ix,model_wv,params_set["trainable_param"])
    print(model.summary())

else:
    x_train,x_test,y_train,y_test = load_data(params_set["vocab_size"],params_set["max_len"])

    train_text,train_label,test_text,test_label = data_prep_ELMo(x_train,y_train,x_test,y_test,params_set["max_len"])

    model = Classification_model_with_ELMo(train_text,train_label,test_text,test_label)
    print(model.summary())
