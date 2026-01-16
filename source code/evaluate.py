import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import clean_text

df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label','text']
df['text'] = df['text'].apply(clean_text)

tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

X = tfidf.transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
print("Recall:", recall_score(y_test, y_pred, pos_label="spam"))
print("F1-Score:", f1_score(y_test, y_pred, pos_label="spam"))