import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from predict import model
import numpy as np
import streamlit as st


# # Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_seq_length = 61


def transliterate(input_sentence):
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    output_seq = model.predict([input_seq, input_seq])[0]

    output_indexes = [np.argmax(pred) for pred in output_seq]
    output_sentence = tokenizer.sequences_to_texts([output_indexes])[0]

    # Remove spaces between characters
    output_sentence = output_sentence.replace(' ', '')

    # string_accuracy(input_sentence, output_sentence)

    return output_sentence

# streamlit UI is disabled due to some error
# def main():
#     # Set the page title
#     st.title("Text Input and Output")

#     # Text input
#     user_input = st.text_input("Enter some text:")

#     # Process the input and display the output
#     if user_input:
#         # output = process_text(user_input)
#         output = transliterate(user_input)
#         st.write("Output:", output)

# Test the model with input sentences


input_sentence = input('Enter word: ')
translated_sentence = transliterate(input_sentence)
print(f"Input: {input_sentence}")
print(f"Transliteration: {''.join(translated_sentence)}")

# if __name__ == '__main__':
#     main()
