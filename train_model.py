import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
import numpy as np
from scipy.sparse import hstack


# Load train.jsonl file
train_file = "train.jsonl"
train_data = []

with open(train_file) as file:
    for line in file:
        train_data.append(json.loads(line))

# turn dictionary into dataframe
df_train = pd.DataFrame(train_data)

# Convert True/False to 1/0
df_train["label"] = df_train["is_generated"].astype(int)

# Declare new function that counts "if", "for", "while" in the code
def count_conds(code: str) -> list[int]:
    if_count = len(re.findall(r"\bif\b", code))
    for_count = len(re.findall(r"\bfor\b", code))
    while_count = len(re.findall(r"\bwhile\b", code))

    return [if_count, for_count, while_count]


counts = []
for code in df_train["code"]:
    counts.append(count_conds(code))

word_counts = np.array(counts)

# Split data into train/validation (2 : 10)
x_train, x_test, y_train, y_test = train_test_split(
    df_train["code"],
    df_train["label"],
    test_size=0.2,
    stratify=df_train["label"],
    random_state=42
)

# Vectorize the code (character n-grams style pattern)
# fit: learning vocabs from the training set
# transform: evaluating each vocab in terms of rareness (how important the word is) = TF-IDF matrix
# Only transform for validation as we only want the model to use learned vocab from train
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# **Combine features
train_indices = x_train.index
test_indices = x_test.index

x_train_combined = hstack([x_train_vec, word_counts[train_indices]])
x_test_combined = hstack([x_test_vec, word_counts[test_indices]])


# Train the model
model = LogisticRegression()
model.fit(x_train_combined, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save thefitted vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Test it on new data
predictions = model.predict(x_test_combined)

print("predictions: ", predictions)
print("Correct answers: ")
print(y_test)

# Calculate the accuracy of this model using accuracy_score function
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy: .2f}")
