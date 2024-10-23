import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Replace NaN values with empty strings
data = df.where(pd.notnull(df), "")

# Print dataset info
data.info()
print(data.shape)

# Convert 'Konu' column to numerical labels
data.loc[data["label"] == "spam", "label"] = 0
data.loc[data["label"] == "ham", "label"] = 1

# Filter out rows with invalid labels
data = data[data["label"].isin([0, 1])]

# Extract features and labels
X = data["text"]
Y = data["label"].astype("int")

# Print features and labels
print("Features (X):", X.head())
print("Labels (Y):", Y.head())

# Split the dataset into training and testing sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_Train_features = feature_extraction.fit_transform(X_Train)
X_Test_features = feature_extraction.transform(X_Test)

# Print shapes of the feature matrices
print("Shape of X_Train_features:", X_Train_features.shape)
print("Shape of X_Test_features:", X_Test_features.shape)

# Model training
model = LogisticRegression()
model.fit(X_Train_features, Y_Train)

# Model evaluation on training data
prediction_on_training_data = model.predict(X_Train_features)
accuracy_on_training_data = accuracy_score(Y_Train, prediction_on_training_data)
print("Accuracy on training data: ", accuracy_on_training_data)

# Model evaluation on test data
prediction_on_test_data = model.predict(X_Test_features)
accuracy_on_test_data = accuracy_score(Y_Test, prediction_on_test_data)
print("Accuracy on test data: ", accuracy_on_test_data)

# Predicting a new mail
input_your_mail = ["10000"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction.shape)
print(prediction)
if prediction[0] == 0:
    print("This is a spam mail")
else:
    print("This is a ham mail")
