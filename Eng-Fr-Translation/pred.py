from vocab import source_idx2word, target_idx2word, source_word2idx, target_word2idx
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense

latent_dim = 256

model = keras.models.load_model('nmt_10epochs.h5')

# model = load_model('/content/drive/MyDrive/Machine Translation Keras/nmt_eng_fr')

# constructing the model layers
encoder_inputs = model.input[0]
encoder_output, state_h_enc, state_c_enc = model.layers[4].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)


decoder_inputs = model.input[1]
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_layer = model.layers[3]
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = model.layers[5]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]

decoder_dense = model.layers[6]

decoder_out = decoder_dense(decoder_outputs)
decoder_model= Model([decoder_inputs] + decoder_states_inputs, [decoder_out] + decoder_states)



# Encode the input sequence to get the "thought vectors"
# encoder_inputs = model.input[0]
# encoder_model = Model(encoder_inputs, encoder_states)

# # Decoder setup
# # Below tensors will hold the states of the previous time step
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# # To predict the next word in the sequence, set the initial states to the states from the previous time step
# decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
# decoder_states2 = [state_h2, state_c2]
# decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# # Final decoder model
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs2] + decoder_states2)



def decode_sequence(input_seq):
    # Encode the input as state vectors.
    print(input_seq)
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of 
    #target sequence with the start character.
    target_seq[0, 0] = target_word2idx['<sos>']
# Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
# Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print(sampled_token_index)
        
        sampled_word =target_idx2word[sampled_token_index]
        decoded_sentence += ' '+ sampled_word
# Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '<eos>' or
           len(decoded_sentence) > 50):
            stop_condition = True
# Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
# Update states
        states_value = [h, c]
    return decoded_sentence

max_source_length = 6


def generate_batch(text, batch_size=1):

    encoder_input_data = np.zeros((batch_size, max_source_length), 
                                  dtype='float32')
    print(encoder_input_data)

    print(text.split())
    length = len(text.split())
    # logger.info(length)
    for i in range(batch_size):
        # logger.info(i)
        for j, word in enumerate(text.split()):
            encoder_input_data[i, j] = source_word2idx[word]
        print(encoder_input_data)
    return encoder_input_data


word = input('enter word: ')

test_gen = generate_batch(word, batch_size = 1)

# input_seq = next(test_gen)
decoded_sentence = decode_sequence(test_gen)
# print('Input Source sentence:', X_test[k:k+1].values[0])
# print('Actual Target Translation:', y_test[k:k+1].values[0][6:-4])
print('Predicted Target Translation:', decoded_sentence[:-4])