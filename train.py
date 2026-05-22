import pandas as pd
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def transform_text(text):
    text = text.lower()

    words = nltk.word_tokenize(text)

    filtered_words = []

    for word in words:
        if word.isalnum():
            filtered_words.append(word)

    cleaned_words = []

    for word in filtered_words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            cleaned_words.append(word)

    final_words = []

    for word in cleaned_words:
        final_words.append(ps.stem(word))

    return " ".join(final_words)

# Apply preprocessing
df['transformed_message'] = df['message'].apply(transform_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_message']).toarray()

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model and vectorizer
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')

print("Model saved successfully!")