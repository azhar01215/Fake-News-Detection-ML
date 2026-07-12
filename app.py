import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.sidebar.title("📊 Model Information")

st.sidebar.success("Accuracy: 99%")

st.sidebar.write("### 🤖 Algorithm")
st.sidebar.write("Logistic Regression")

st.sidebar.write("### 📚 Vectorizer")
st.sidebar.write("TF-IDF")

st.sidebar.write("### 📂 Dataset")
st.sidebar.write("44,898 News Articles")

st.sidebar.write("### 👨‍💻 Developer")
st.sidebar.write("MD Azhar Mehemud Molla")

st.title("📰 Fake News Detection App")

st.markdown("""
### About this Project

This Fake News Detection System uses Machine Learning to classify whether a news article is **Fake** or **Real**.

### Features
- ✅ Machine Learning Model
- ✅ TF-IDF Text Vectorization
- ✅ Logistic Regression
- ✅ Confidence Score
- ✅ Interactive Web Application
""")
st.divider()

st.write("Enter a news article below and click Predict.")

news = st.text_area("Enter News")

if st.button("Load Example News"):
    news = """The Government of India announced a new education policy to improve digital learning across the country."""

if st.button("Clear"):
    st.rerun()

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        vector = vectorizer.transform([news])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.info(f"Confidence Score: {confidence:.2f}%")

        st.progress(int(confidence))

        st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit & Scikit-learn</center>",
    unsafe_allow_html=True
)