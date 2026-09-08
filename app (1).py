
import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Website title
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it is Real or Fake.")

# News input
news_text = st.text_area(
    "Enter News Article:",
    height=250,
    placeholder="Paste the news article here..."
)

# Prediction
if st.button("🔍 Check News"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        # Convert news into TF-IDF features
        news_vector = tfidf.transform([news_text])

        # Predict
        prediction = model.predict(news_vector)[0]

        # Display result
        if prediction == 1 or str(prediction).lower() == "real":
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
