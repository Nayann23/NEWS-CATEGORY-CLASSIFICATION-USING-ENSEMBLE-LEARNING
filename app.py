# import streamlit as st
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# model = pickle.load(open('stacking_model.pkl', 'rb'))
# vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# category_dict = {
#     0: "World",
#     1: "Sports",
#     2: "Business",
#     3: "Sci/Tech"
# }

# st.title("News Classification App")

# user_input = st.text_area("Enter news text here:")

# if st.button("Predict"):
#     if user_input.strip():
#         input_tfidf = vectorizer.transform([user_input])
#         prediction = model.predict(input_tfidf)[0]
#         category = category_dict.get(prediction, "Unknown Category")
#         st.write(f"Predicted Category: {category}")
#     else:
#         st.write("Please enter some text for prediction.")



# _____________________________________________________________


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

model = pickle.load(open('stacking_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

category_dict = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    .css-1v3fvcr h1 {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown(
    "<h1 style='color: #DC143C;'>News Classification App</h1>",
    unsafe_allow_html=True
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip():
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        decision_scores = model.decision_function(input_tfidf)[0]
        probs = softmax(decision_scores)
        confidence = np.max(probs) * 100
        category = category_dict.get(prediction, "Unknown Category")
        st.write(f"Predicted Category: {category} ({confidence:.2f}%)")
    else:
        st.write("Please enter some text for prediction.")
