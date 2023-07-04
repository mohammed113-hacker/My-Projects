from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from transformation import X,y


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb1 = GaussianNB()
nb1.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred1 = nb1.predict(X_test)

# Print the classification report and confusion matrix
print("Naive Bayes Report before applying SMOTE :\n")
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))


# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

priors = [0.8, 0.2]
# Train a Naive Bayes classifier on the resampled data
nb = GaussianNB(priors=priors,var_smoothing=1e-3)
nb.fit(X_train_resampled, y_train_resampled)

# Predict the labels for the testing data
y_pred = nb.predict(X_test)

# Print the classification report and confusion matrix
print("Naive Bayes Report after applying SMOTE :\n")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))