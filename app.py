import streamlit as st
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 Fake News Detection")

st.sidebar.info("""
### 🤖 Machine Learning Project

**Algorithm:** Logistic Regression

**Vectorizer:** TF-IDF

**Accuracy:** 99%

**Dataset:** 44,898 News Articles

**Developer:** MD Azhar Mehemud Molla
""")

# ----------------------------
# Main Title
# ----------------------------
st.title("📰 Fake News Detection System")

st.markdown("""
Detect whether a news article is **Fake** or **Real** using Machine Learning.
""")

st.divider()

# ----------------------------
# News Input
# ----------------------------
news = st.text_area(
    "📝 Enter News Article",
    height=220,
    placeholder="Paste the complete news article here..."
)

# ----------------------------
# Example Button
# ----------------------------
if st.button("📄 Load Example News"):
    news = """The Government of India announced a new education policy to improve digital learning across the country."""

# ----------------------------
# Clear Button
# ----------------------------
if st.button("🗑️ Clear"):
    st.rerun()

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠️ Please enter a news article.")

    else:

        vector = vectorizer.transform([news])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News Detected")
            st.balloons()

        st.info(f"📊 Confidence Score: {confidence:.2f}%")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption("👨‍💻 Developed by MD Azhar Mehemud Molla")

st.caption("📰 Fake News Detection using Machine Learning")

st.caption("🚀 Powered by Streamlit & Scikit-learn")