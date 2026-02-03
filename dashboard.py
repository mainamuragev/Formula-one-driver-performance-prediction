import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from main import load_data, train_model

st.title("Formula One Driver Performance Prediction 🏎️")

data = load_data()
st.subheader("Driver Stats")
st.dataframe(data)

model, mse = train_model(data)
st.write(f"Model trained with Mean Squared Error: {mse:.2f}")

st.subheader("Points Distribution")
fig, ax = plt.subplots()
sns.barplot(x="Driver", y="Points", data=data, ax=ax)
st.pyplot(fig)

