import streamlit as st
import pickle
import json

import numpy as np

locations = None
data_columns = None
model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)


print("loading saved artifacts...start")

with open('./columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]

if model is None:
    with open('./banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
print("loading saved artifacts...done")

st.header("bengluru Price Predictor")

selected_location = st.selectbox(
    "Type or select a movie from the dropdown",
    locations
)
selected_bhk = st.number_input(
    "select the no. of bedrooms you want for the apartment", value=3, step=1)
selected_bath = st.number_input(
    "select the no. of baths you want for the apartment", value=3, step=1)
selected_area = st.number_input(
    "select the no. of total area in sq.ft you want for the apartment", value=1500, step=50)

get_estimated_price = get_estimated_price(
    selected_location, selected_area, selected_bhk, selected_bath)

if st.button("get Price"):
    st.text("the estimated price is: ")
    st.text("{} Lakhs".format(get_estimated_price))


# if __name__ == "__main__":
# dummy()
