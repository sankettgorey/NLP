import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


model = load_model('./sentimen_analysis.h5')

tokenizer_file = 'tokenizer.pkl'

with open('./tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
# print(tokenizer.word_index['disaster'])

def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    print(sequence)
    
    padded = pad_sequences(sequence, padding = 'post', truncating='post', maxlen = 25)
    # print(padded)
    
    preds = model.predict(padded)
    
    return 'disaster' if preds >= 0.5 else 'not disaster'


# print(predict_text('people are not dead'))


