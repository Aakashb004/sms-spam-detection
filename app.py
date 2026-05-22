import streamlit as st
import joblib
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Load model and vectorizer
model = joblib.load('models/spam_model.pkl')
tfidf = joblib.load('models/vectorizer.pkl')

# Spam keywords
spam_keywords = [
    "free", "win", "winner", "offer", "reward",
    "urgent", "money", "prize", "claim", "cash"
]

# Text preprocessing
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum()]
    cleaned_words = [word for word in filtered_words if word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join([ps.stem(word) for word in cleaned_words])

# Page config
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# Initialize Session States
if "example" not in st.session_state:
    st.session_state.example = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Theme toggle (Streamlit UI dark mode relies on configuration, 
# but we use this toggle to dynamically change the Matplotlib chart style!)
dark_mode = st.toggle("🌙 Dark Mode", value=True)

# Main title
st.title("📩 SMS Spam Detection System")
st.write("Detect whether an SMS message is Spam or Not Spam.")

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Try Spam Example"):
        st.session_state.example = "URGENT! You won a FREE iPhone. Claim now!"
with col2:
    if st.button("Try Normal Example"):
        st.session_state.example = "Hey bro, let's meet at 5 PM today."

# File upload handler (Updates text box text automatically)
uploaded_file = st.file_uploader("📁 Upload a text file", type=["txt"])
if uploaded_file is not None:
    st.session_state.example = uploaded_file.read().decode("utf-8")

# Single Unified Text Area
input_sms = st.text_area(
    "Enter the SMS message",
    value=st.session_state.example,
    height=150
)

# Predict button
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Stats
        word_count = len(input_sms.split())
        char_count = len(input_sms)

        # Spam keyword detection
        detected_keywords = [kw for kw in spam_keywords if kw.lower() in input_sms.lower()]

        # Preprocess, Vectorize & Predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        ham_probability = probability[0] * 100
        spam_probability = probability[1] * 100

        # Statistics Layout
        st.subheader("📊 Message Statistics")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Words", word_count)
        with stat_col2:
            st.metric("Characters", char_count)

        # Spam keyword display
        st.subheader("🚨 Detected Spam Keywords")
        if detected_keywords:
            st.error(", ".join(detected_keywords))
        else:
            st.success("No suspicious keywords detected")

        # Two Column Layout for Prediction Result vs Pie Chart
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            st.subheader("🔍 Prediction Result")
            if result == 1:
                st.error("🚨 Spam Message")
                st.progress(int(spam_probability))
                st.write(f"Spam Confidence: {spam_probability:.2f}%")
            else:
                st.success("✅ Not Spam")
                st.progress(int(ham_probability))
                st.write(f"Not Spam Confidence: {ham_probability:.2f}%")

        with res_col2:
            st.subheader("📈 Confidence Visualization")
            
            # Match Matplotlib style with Dark Mode choice
            if dark_mode:
                plt.style.use('dark_background')
                text_color = 'white'
            else:
                plt.style.use('default')
                text_color = 'black'

            labels = ["Not Spam", "Spam"]
            values = [ham_probability, spam_probability]

            # Re-designed clean pie chart
            fig, ax = plt.subplots(figsize=(3, 3))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10, 'color': text_color},
                pctdistance=0.65
            )
            
            # Ensure text colors match selection perfectly
            for t in texts + autotexts:
                t.set_color(text_color)

            ax.axis('equal')  
            plt.tight_layout()
            st.pyplot(fig)

        # Save history
        prediction_label = "Spam" if result == 1 else "Not Spam"
        st.session_state.history.append({
            "Message": input_sms[:60] + "..." if len(input_sms) > 60 else input_sms,
            "Prediction": prediction_label
        })

# Prediction history
st.subheader("📜 Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions yet.")

# Footer
st.markdown("---")
st.caption("Built using Python, NLP, Scikit-learn, and Streamlit")