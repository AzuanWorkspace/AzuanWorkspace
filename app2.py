# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load book data
@st.cache
def load_data():
    # Assuming the book data is in CSV format
    data = pd.read_csv('Best_Books_Ever.csv')  # Replace with actual data path
    return data

book_data = load_data()

# Title of the dashboard
st.title("Library Dashboard: Predictions and Sentiment Analysis")

# Dropdown for selecting prediction type
option = st.selectbox(
    "Select a prediction type:",
    ("Most Borrowed", "Most Popular", "Sentiment Analysis")
)

# Preprocess the data: Label Encoding
label_encoder = LabelEncoder()

# Fallback for missing 'genres' column
if 'genres' not in book_data.columns:
    book_data['genres'] = 'Unknown'

book_data['title_encoded'] = label_encoder.fit_transform(book_data['title'])
book_data['author_encoded'] = label_encoder.fit_transform(book_data['author'])
book_data['genres_encoded'] = label_encoder.fit_transform(book_data['genres'])

# Define a RandomForest model to use for predictions
def train_borrowing_model():
    X = book_data[['likedPercent', 'bbeVotes', 'title_encoded', 'author_encoded', 'genres_encoded']]
    y = book_data['rating']  # Assuming rating correlates with borrow trends
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    return model, X.columns

# Section for Most Borrowed Prediction
if option == "Most Borrowed":
    st.header("Most Borrowed Prediction")
    
    # Train model and get feature columns
    model, feature_columns = train_borrowing_model()
    
    # Show feature importance for borrow prediction
    feature_importance = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_columns, ax=ax)
    st.pyplot(fig)

# Section for Most Popular Books
elif option == "Most Popular":
    st.header("Most Popular Books")
    top_books = book_data.nlargest(10, 'rating')[['title', 'author', 'rating']]
    st.table(top_books)

# Section for Sentiment Analysis
elif option == "Sentiment Analysis":
    st.header("Sentiment Analysis of Book Descriptions")
    analyzer = SentimentIntensityAnalyzer()

    # Handle missing values in the description column
    book_data['description'] = book_data['description'].fillna('')

    # Apply sentiment analysis
    book_data['sentiment'] = book_data['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if x else 0)

    # Show top 5 books by positive sentiment
    positive_books = book_data.nlargest(5, 'sentiment')[['title', 'author', 'sentiment']]
    st.table(positive_books)

    # Show top 5 books by negative sentiment
    negative_books = book_data.nsmallest(5, 'sentiment')[['title', 'author', 'sentiment']]
    st.table(negative_books)
