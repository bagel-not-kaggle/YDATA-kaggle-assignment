"""
import streamlit as st
st.title("Streamlit Update Cycle - Potential Pitfall Example")
# Counter example that resets unintentionally due to update cycle
counter = 0  # This is not using session state, causing issues
st.write(f"Current Counter Value: **{counter}**")
# When the button is clicked, it triggers an update
if st.button("Increment Counter"):
   counter += 1
   st.write("Counter incremented!")

st.write("---")
st.warning(
   "👆 Notice that every time you click the button, "
   "the counter resets to 0 before being incremented.\n\n"
   "This happens because Streamlit reruns the entire script on each update, "
   "and without proper state management, variables are reset to their initial values."
)
st.subheader("The Fix")
st.write(
   "To fix this issue, you can use Streamlit's `st.session_state` "
   "to preserve the counter value across updates."
)
if st.button("Try the fixed version (with session state)"):
   st.session_state.counter = st.session_state.get("counter", 0) + 1
   st.write(f"Fixed Counter Value: **{st.session_state.counter}**")
"""
import streamlit as st
from catboost import CatBoostClassifier

MODEL_PATH = "models/catboost_model.cbm"

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

# from bokeh.plotting import figure

# Create a Bokeh figure
# p = figure(title="Bokeh in Streamlit", x_axis_label="X", y_axis_label="Y")
# p.line([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], line_width=2)

# Show it in Streamlit
# st.bokeh_chart(p)
