import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Step 1: Data Load
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Encode Labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Step 4: Text Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

# Step 6: Accuracy
accuracy = metrics.accuracy_score(y_test, nb_pred)
print("Accuracy:", accuracy)

# Step 7: Sample Test
sample = ["Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121"]
sample_vec = vectorizer.transform(sample)
prediction = nb.predict(sample_vec)
print("Prediction (0=Ham, 1=Spam):", prediction[0])
