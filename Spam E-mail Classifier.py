# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset
# Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Preprocess Data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

# Step 5: Vectorize Text (Bag of Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6A: Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# Step 6B: Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

# Step 7: Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Step 8: Evaluate Both Models
evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("SVM", y_test, svm_preds)

# Step 9: Visualize Confusion Matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_conf_matrix(y_test, nb_preds, "Naive Bayes")
plot_conf_matrix(y_test, svm_preds, "SVM")

# Step 10: Predict New Email
def predict_email(text):
    vec = vectorizer.transform([text])
    nb_result = nb_model.predict(vec)[0]
    svm_result = svm_model.predict(vec)[0]
    print(f"Naive Bayes Prediction: {'Spam' if nb_result else 'Ham'}")
    print(f"SVM Prediction: {'Spam' if svm_result else 'Ham'}")

# Test Example
predict_email("Congratulations! You've won a free ticket. Call now!")