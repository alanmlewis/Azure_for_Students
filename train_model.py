# Important the function which will load the dataset
import numpy as np
from sklearn.datasets import fetch_openml
# Load the dataset
mushrooms = fetch_openml(name='mushroom', version=1)
# Extract the features from the dataset
features = mushrooms.data[['cap-shape','cap-surface','cap-color','stalk-shape','stalk-root']]
# Extract the class labels from the dataset
classes = mushrooms.target
# Create a dictionary which maps the label e(dible) to 0 and p(oisonous) to 1
map_dict = {'e': 0, 'p': 1}
# Map the class labels to numerical values
classes = classes.map(map_dict).to_numpy()

#Create a list of the row indices which are not missing values of 'stalk-root'
idx = np.where(features['stalk-root'].notna())

# Remove samples with missing values
x = features.dropna()
# Retain only class labels corresponding to samples with complete features
y = classes[idx]

# Import Scikit-learn's one-hot encoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Split the data into a training and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 1
)

# Define the machine learning pipeline
pipeline = Pipeline([
        ("encoder", OneHotEncoder(sparse_output=False)),
        ("model"  , LogisticRegression())
        ])

# Fit the model
pipeline.fit(x_train, y_train)

# Predict the accuracy of the tree over the test subset.
y_predict_test = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_predict_test)

print(f'Congratulations! You scored an accuracy of {int(accuracy*100)}%!')

import joblib
joblib.dump(pipeline, 'model.joblib')
