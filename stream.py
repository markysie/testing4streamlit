import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

 st.title("제주도 관광지추천")

# Load data
df = pd.read_csv("merged_data.csv")

# Data normalization
scaler = MinMaxScaler()
df[["Y_COORD", "X_COORD"]] = scaler.fit_transform(df[["Y_COORD", "X_COORD"]])

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocessor and model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ["Y_COORD", "X_COORD", "AGE_GRP"])
    ])

knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Grid search
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(train_df[["Y_COORD", "X_COORD", "AGE_GRP"]], train_df["VISIT_AREA_NM"])

# Streamlit app
st.title("Jeju Island Tourist Attraction Recommendation")

# User input
age_range = st.radio("Select Age Range:", [20, 30, 40, 50, 60])

user_location = st.text_input("Enter Your Location (comma-separated coordinates):", "33.450722, 126.512222")
user_location = list(map(float, user_location.split(',')))

# Recommendation button
if st.button("Search"):
    # Transform user input
    user_input = pd.DataFrame(np.array([user_location[0], user_location[1], age_range]).reshape(1, -1),
                              columns=["Y_COORD", "X_COORD", "AGE_GRP"])
    user_input_transformed = pd.DataFrame(preprocessor.transform(user_input),
                                          columns=["Y_COORD", "X_COORD", "AGE_GRP"])

    # Use the tuned model for prediction
    user_recommendation_tuned = grid_search.best_estimator_.predict(user_input_transformed)

    # Display recommendation
    st.success(f"Recommended Tourist Attraction: {user_recommendation_tuned[0]}")

# Display best hyperparameters and tuned model accuracy
st.subheader("Tuned Model Information:")
st.text(f"Best Hyperparameters: {grid_search.best_params_}")

# Test the tuned model
y_pred_tuned = grid_search.predict(test_df[["Y_COORD", "X_COORD", "AGE_GRP"]])

# Display tuned model accuracy
accuracy_tuned = accuracy_score(test_df["VISIT_AREA_NM"], y_pred_tuned)
st.text(f"Tuned Model Accuracy: {accuracy_tuned:.2%}")
