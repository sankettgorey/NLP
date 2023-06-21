import json
from requests import request
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import Input
import numpy as np
from keras.utils import pad_sequences
from loguru import logger
# from pred import decode_sequence
from loguru import logger

# model = keras.models.load_model('nmt_10epochs.h5')


# Load the vocabulary from the file
with open('source_idx2word.json', 'r') as f:
    source_idx2word = json.load(f)

with open('target_idx2word.json', 'r') as f:
    target_idx2word = json.load(f)

source_word2idx = dict([(word, index) for index, word in source_idx2word.items()])

target_word2idx = dict([(word, index) for index, word in target_idx2word.items()])


# integers = list(target_idx2word.keys())
integers = [int(i) for i in target_idx2word.keys()]


target_idx2word = dict(zip(integers, target_idx2word.values()))


