import streamlit as st
from src.planner import search_whiskies_with_planner
from src.sommelier import generate_explanation

st.title("🥃 The AI Whisky Sommelier")
user_query = st.text_input("What type of whisky are you looking for?")
if st.button("Find Recommendations"):
    recs = search_whiskies_with_planner(user_query)
    response = generate_explanation(user_query, recs)
    st.markdown(response)