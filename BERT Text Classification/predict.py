import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.utils import to_categorical, plot_model
from keras.layers import Dense, Input

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.initializers import TruncatedNormal
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import Model


import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, TFBertModel

bert = TFBertModel.from_pretrained('./bert-model')
loaded_tokenizer = AutoTokenizer.from_pretrained('./tokenizer_directory')



max_len = 100


input_ids = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_ids')
input_mask = Input(shape = (max_len, ), dtype = tf.int32, name = 'attention_mask')

embeddings = bert(input_ids, attention_mask = input_mask)[0]    # 0 is last hidden state. 1 is pooler_output
out = keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation = 'relu')(out)
out = keras.layers.Dropout(0.1)(out)
out = Dense(32, activation = 'relu')(out)

y = Dense(6, activation = 'softmax')(out)

model = Model(inputs = [input_ids, input_mask], outputs = y)


# loading model weights
model.load_weights('./text_classification_bert_weights.h5')


encoded_dict = {'anger': 0, 'fear': 1, 'happy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}


def input_text(text):
    
    test_input_ids = loaded_tokenizer.encode(text, add_special_tokens=True, 
                                         max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')
    
    text = model.predict([test_input_ids, tf.ones_like(test_input_ids)])
    
    if np.argmax(text) == 0:
        return list(encoded_dict.keys())[0] 
    elif np.argmax(text) == 1:
        return list(encoded_dict.keys())[1]
    
    elif np.argmax(text) == 2:
        return list(encoded_dict.keys())[2]
    
    elif np.argmax(text) == 3:
        return list(encoded_dict.keys())[3]
    
    elif np.argmax(text) == 4:
        return list(encoded_dict.keys())[4]
    
    else:
        return list(encoded_dict.keys())[5]
    
    
# print(input_text('i am happy'))