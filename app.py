import streamlit as st

a = st.number_input("enter a num")

clicked = st.button("submit")
if clicked:
    st.write(f"submited: {a}")
