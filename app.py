import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit App Title
st.title("Data Science Project: Weather Data Analysis")

# Load dataset
df = pd.read_csv("weather_data.csv")

# Display basic info
st.header("Dataset Overview")
st.write(df.head())
st.write(df.describe())

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Data visualization
st.header("Temperature Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df["Temperature"], bins=30, kde=True)
st.pyplot(plt)

# Correlation heatmap
st.header("Correlation Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)

# User Input for Custom Analysis
st.sidebar.header("Custom Analysis")
temp_range = st.sidebar.slider("Select Temperature Range", float(df["Temperature"].min()), float(df["Temperature"].max()), (float(df["Temperature"].min()), float(df["Temperature"].max())))
filtered_df = df[(df["Temperature"] >= temp_range[0]) & (df["Temperature"] <= temp_range[1])]
st.write(f"Filtered Data: {len(filtered_df)} records")
st.write(filtered_df)

# Train-test split
from sklearn.model_selection import train_test_split
X = df.drop("Temperature", axis=1)
y = df["Temperature"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
st.header("Model Training")
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
from sklearn.metrics import mean_squared_error
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
st.write(f"### Mean Squared Error: {mse:.2f}")

# Feature Importance
st.header("Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Conclusion
st.write("This tool provides an interactive data science analysis on weather data using Streamlit.")
