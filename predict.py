import pickle
import json
import re
import numpy as np
from scipy.sparse import hstack
from train_model import count_conds, word_counts
from tensorboard.compat.tensorflow_stub.tensor_shape import vector

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the data from test.jsonl
test_data = []
word_feature = []
with open('test.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        code = data["code"]
        test_data.append(data['code'])
        word_feature.append(count_conds(code))

# Convert word_feature to Numpy array
word_feature_array = np.array(word_feature)

# Transform the data using the same vectorizer
x_vec = vectorizer.transform(test_data)

# ** Combine theTF-IDF feature with theword feature
x_combined = hstack([x_vec, word_feature_array])

# Make prediction
predictions = model.predict(x_combined)

# Print results
for prediction in predictions:
    print(prediction)
