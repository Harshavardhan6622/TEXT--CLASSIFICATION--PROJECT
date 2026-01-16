import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from preprocessing import clean_text
import pickle

df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
