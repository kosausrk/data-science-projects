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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2) Ensure NLTK stopwords are available
nltk.download('stopwords', quiet=True)

# 3) Load your data and drop any empty/malformed rows

#using data 2 ~1.5k rows 
df = pd.read_csv('./movie-data2.csv', encoding='utf-8')
df.dropna(subset=['review', 'sentiment'], inplace=True)
df = df[df['review'].str.strip() != '']

# 4) Cleaning function
def clean_text(txt: str) -> str:
    txt = re.sub(r'<[^>]+>', ' ', txt)  # strip HTML
    txt = re.sub(r'[^a-zA-Z\s]', ' ', txt)  # remove punctuation/digits
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

# 9) TF-IDF Vectorize with n-grams
vectorizer = TfidfVectorizer(
    stop_words=all_stopwords,
    max_features=2000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)  # include unigrams and bigrams
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

sns.countplot(x=y, hue=y, palette='Set2', dodge=False)


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

# 12.4 Top N-Grams per Sentiment
import numpy as np
feature_names = vectorizer.get_feature_names_out()
class_labels = model.classes_
log_probs = model.feature_log_prob_

for i, label in enumerate(class_labels):
    top_indices = np.argsort(log_probs[i])[-10:]
    top_features = [feature_names[j] for j in reversed(top_indices)]
    print(f"\nTop predictive n-grams for '{label}' sentiment:")
    for phrase in top_features:
        print(f"  - {phrase}")


# # 12.5 Most Common Mid-Length Review Phrases (Practical Insight)
# from collections import Counter
# import nltk

# nltk.download('punkt', quiet=True)


# from nltk.tokenize import sent_tokenize

# # Collect mid-length phrases (5–15 words) from cleaned reviews
# all_phrases = []
# for review in df['clean_review']:
#     sentences = sent_tokenize(review)
#     for sentence in sentences:
#         sentence = sentence.strip()
#         word_count = len(sentence.split())
#         if 5 < word_count < 15:
#             all_phrases.append(sentence)

# # Count and get top phrases
# common_phrases = Counter(all_phrases).most_common(10)

# # Plot if available
# if common_phrases:
#     phrases, freqs = zip(*common_phrases)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=list(freqs), y=list(phrases), palette='viridis')
#     plt.title("Top Mid-Length Review Phrases", fontsize=14)
#     plt.xlabel("Frequency")
#     plt.ylabel("Phrase")
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No common mid-length phrases found.")


# 12.5 Most Common Mid-Length Review Phrases (No NLTK Required)
from collections import Counter

# Simple sentence split using punctuation (fallback method)
def basic_sentence_split(text):
    return re.split(r'[.!?]', text)

# Collect mid-length phrases (5–15 words) from cleaned reviews
all_phrases = []
for review in df['clean_review']:
    sentences = basic_sentence_split(review)
    for sentence in sentences:
        sentence = sentence.strip()
        word_count = len(sentence.split())
        if 5 < word_count < 15:
            all_phrases.append(sentence)

# Count and get top phrases
common_phrases = Counter(all_phrases).most_common(10)

# Plot if available
if common_phrases:
    phrases, freqs = zip(*common_phrases)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(freqs), y=list(phrases), palette='viridis')
    plt.title("Top Mid-Length Review Phrases (No NLTK)", fontsize=14)
    plt.xlabel("Frequency")
    plt.ylabel("Phrase")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("No common mid-length phrases found.")




# 13) Prediction helper
def predict_sentiment(review: str) -> str:
    clean = clean_text(review)
    vec   = vectorizer.transform([clean])
    return model.predict(vec)[0]

# 14) Sample run
if __name__ == '__main__':
    sample = "The movie was absolutely fantastic and thrilling!"
    print("Sample review prediction:", predict_sentiment(sample))
