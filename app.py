import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

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

st.image("logo.png", width=120)

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
    st.rerun()

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
            result = "Fake"
            st.error("🚨 Fake News Detected")
        else:
            result = "Real"
            st.success("✅ Real News Detected")
            st.balloons()

        st.info(f"📊 Confidence Score: {confidence:.2f}%")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

        # ----------------------------
        # Probability Chart
        # ----------------------------

        fake_probability = probability[0][0] * 100
        real_probability = probability[0][1] * 100

        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots(figsize=(6, 2.5))

        labels = ["Fake", "Real"]
        values = [fake_probability, real_probability]

        ax.barh(labels, values)

        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Model Confidence")

        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.2f}%", va="center")

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------
        # Save Prediction History
        # ----------------------------

        history = {
            "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Prediction": result,
            "Confidence": f"{confidence:.2f}%",
            "News": news
        }

        history_file = "prediction_history.csv"

        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df = pd.concat([df, pd.DataFrame([history])], ignore_index=True)
        else:
            df = pd.DataFrame([history])

        df.to_csv(history_file, index=False)

# ----------------------------
# Prediction History
# ----------------------------

st.divider()

st.subheader("📜 Prediction History")

history_file = "prediction_history.csv"

if os.path.exists(history_file):

    history_df = pd.read_csv(history_file)

    st.dataframe(history_df, use_container_width=True)

    with open(history_file, "rb") as file:

        st.download_button(
            label="📥 Download Prediction History",
            data=file,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

else:

    st.info("No prediction history available.")

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.caption("👨‍💻 Developed by MD Azhar Mehemud Molla")

st.caption("📰 Fake News Detection using Machine Learning")

st.caption("🚀 Powered by Streamlit & Scikit-learn")