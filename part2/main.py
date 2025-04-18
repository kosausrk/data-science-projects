# main.py

# 1) Imports
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2) Ensure NLTK stopwords are available
nltk.download('stopwords', quiet=True)

# 3) Load your data and drop any empty/malformed rows
df = pd.read_csv('./movie-data1.csv', encoding='utf-8')
df.dropna(subset=['review', 'sentiment'], inplace=True)
df = df[df['review'].str.strip() != '']

# 4) Cleaning function
def clean_text(txt: str) -> str:
    # strip HTML tags
    txt = re.sub(r'<[^>]+>', ' ', txt)
    # remove punctuation and digits
    txt = re.sub(r'[^a-zA-Z\s]', ' ', txt)
    # lowercase + collapse repeated whitespace
    return re.sub(r'\s+', ' ', txt).strip().lower()

# 5) Build stopword list
nltk_sw  = set(stopwords.words('english'))
custom_sw = {'br', 'movie', 'film'}
all_stopwords = list(nltk_sw.union(custom_sw))

# 6) Apply cleaning
df['clean_review'] = df['review'].apply(clean_text)

# 7) Prepare features/labels
X = df['clean_review']
y = df['sentiment'].map(lambda s: s.strip().lower())

# 8) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9) Vectorize
vectorizer = CountVectorizer(
    stop_words=all_stopwords,
    max_features=1000,
    min_df=5,
    max_df=0.8
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 10) Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 11) Evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 12) Visualizations

# 12.1 Sentiment Distribution
plt.figure(figsize=(8,6))
sns.countplot(x=y, palette='Set2')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# 12.2 Word Clouds for each class
for label in ['positive', 'negative']:
    text = ' '.join(df[df['sentiment'].str.lower() == label]['clean_review'])
    if not text:
        continue
    wc = WordCloud(
        stopwords=all_stopwords,
        background_color='white',
        max_words=200
    ).generate(text)
    plt.figure(figsize=(6,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"Word Cloud for {label.title()} Reviews")
    plt.axis('off')
    plt.show()

# 12.3 Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['negative','positive'])
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negative','Positive'],
    yticklabels=['Negative','Positive']
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 13) Prediction helper
def predict_sentiment(review: str) -> str:
    clean = clean_text(review)
    vec   = vectorizer.transform([clean])
    return model.predict(vec)[0]

# 14) Sample run
if __name__ == '__main__':
    sample = "The movie was absolutely fantastic and thrilling!"
    print("Sample review prediction:", predict_sentiment(sample))
