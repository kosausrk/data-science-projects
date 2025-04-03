# Import necessary libraries for visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# Load Data
data = pd.read_csv('./movie-data1.csv')

# Display first few rows to verify the structure
print(data.head())

# Preprocessing the Data
X = data['review']
y = data['sentiment']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the accuracy and print a classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualizations
## 1. Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=data, palette='Set2')
plt.title("Sentiment Distribution")
plt.show()

## 2. Word Cloud for Positive Reviews
positive_reviews = data[data['sentiment'] == 'positive']['review']
positive_text = ' '.join(positive_reviews)
positive_wordcloud = WordCloud(stopwords='english', background_color='white').generate(positive_text)

plt.figure(figsize=(8, 8))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Positive Reviews")
plt.axis('off')
plt.show()

## 3. Word Cloud for Negative Reviews
negative_reviews = data[data['sentiment'] == 'negative']['review']
negative_text = ' '.join(negative_reviews)
negative_wordcloud = WordCloud(stopwords='english', background_color='white').generate(negative_text)

plt.figure(figsize=(8, 8))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Negative Reviews")
plt.axis('off')
plt.show()

## 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Make Predictions
# Function to predict sentiment of new reviews
def predict_sentiment(review):
    review_vec = vectorizer.transform([review])
    sentiment = model.predict(review_vec)
    return sentiment[0]

# Test with a sample review
sample_review = "The movie was absolutely fantastic and thrilling!"
print(f"Sentiment: {predict_sentiment(sample_review)}")
