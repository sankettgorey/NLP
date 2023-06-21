import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
import numpy as np
from keras.layers import Input, Dense, GRU
import pickle
import keras


max_seq_length = 61

# Load the tokenizer


def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


tokenizer = load_tokenizer()

input_vocab_size = len(tokenizer.word_index) + 1
output_vocab_size = len(tokenizer.word_index) + 1
latent_dim = 256


# Define the model architecture
# Encoder
def encoder(max_seq_length, input_vocab_size, latent_dim):
    encoder_inputs = Input(shape=(max_seq_length,))
    encoder_embedding = Embedding(input_vocab_size, latent_dim)(encoder_inputs)
    encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h = encoder_gru(encoder_embedding)
    encoder_states = [state_h]
    return encoder_states, encoder_inputs


# Decoder
def decoder(max_seq_length, output_vocab_size, latent_dim, encoder_states):
    decoder_inputs = Input(shape=(max_seq_length,))
    decoder_embedding = Embedding(
        output_vocab_size, latent_dim)(decoder_inputs)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(
        decoder_embedding, initial_state=encoder_states)
    # return decoder_outputs

    # Dense layer for prediction
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    output = decoder_dense(decoder_outputs)
    return decoder_inputs, output


encoder_states, encoder_inputs = encoder(
    max_seq_length, input_vocab_size, latent_dim)
decoder_inputs, output = decoder(
    max_seq_length, output_vocab_size, latent_dim, encoder_states)


model = Model([encoder_inputs, decoder_inputs], output)

model = keras.models.load_model(
    './model')
