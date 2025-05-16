import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime

LOG_FILE = "workout_log.csv"

if not os.path.exists(LOG_FILE):
    define_init = pd.DataFrame(columns = ["date", "exercise", "weight", "reps"])
    define_init.to_csv(LOG_FILE, index = False)
    
def estimate_1rm(weight, reps):
    if reps == 1:
        return weight
    return weight * (1 + reps / 30)

st.title("PR Estimator")
with st.form("log_form"):
    date = st.date_input("Date", value = datetime.today())
    exercise = st.text_input("Exercise", placeholder = "e.g., deadlift")
    weight = st.number_input("Weight (kg)", min_value = 0.00, step = 0.25)
    reps = st.number_input("Reps", min_value = 1, step = 1)
    submit = st.form_submit_button("Log Set")
    
if submit:
    new_log = pd.DataFrame([{
        "date": date.strftime("%Y-%m-%d"),
        "exercise": exercise.lower(),
        "weight": weight,
        "reps": reps,
    }])
    new_log.to_csv(LOG_FILE, mode = "a", header = False, index = False)
    st.success("Set successfully logged!")

data_file = pd.read_csv(LOG_FILE)

if not data_file.empty and "weight" in data_file.columns and "reps" in data_file.columns:
    data_file["estimated_1rm"] = data_file.apply(lambda row: estimate_1rm(row.weight, row.reps), axis = 1)
    st.subheader("Your workout history")
    st.dataframe(data_file.sort_values("date", ascending = False))
    
    if len(data_file) >= 5:
        x = data_file[["weight", "reps"]]
        y = data_file["estimated_1rm"]
        
        if len(x) > 0 and len(y) > 0:
            model = LinearRegression()
            model.fit(x, y)
            
            latest_input = [[weight, reps]]
            prediction = model.predict(latest_input)[0]
            st.subheader("1RM Prediction")
            st.metric(label = "Prediction", value = f"{prediction:.2f}kg")
        else:
            st.warning("Not enough data to train the AI model.")
    else:
        estimation = estimate_1rm(weight, reps)
        st.subheader("1RM Estimation")
        st.metric(label = "Estimation", value = f"{estimation:.2f}kg")
else:
    st.warning("No workout data available. Please log a new set using the form above.")