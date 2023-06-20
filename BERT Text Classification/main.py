import streamlit as st
from predict import input_text

def main():
    # Set the page title
    st.title("Text Input and Output")

    # Text input
    user_input = st.text_input("Enter some text:")

    # Process the input and display the output
    if user_input:
        # output = process_text(user_input)
        output = input_text(user_input)
        st.write("Output:", output)

def process_text(text):
    # Perform some processing on the input text
    # Here, we'll just return the input text as it is
    return input_text

if __name__ == '__main__':
    main()