
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from transformation import X,y
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm1=SVC()
svm1.fit(X_train,y_train)
y_pred1=svm1.predict(X_test)
print("SVM Report before applying SMOTE :\n")
print(classification_report(y_test,y_pred1,zero_division=0))
print(confusion_matrix(y_test,y_pred1))


# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train an SVM model on the resampled data
svm = SVC()
svm.fit(X_train_resampled, y_train_resampled)
y_pred=svm.predict(X_test)
# Evaluate the performance of the SVM model on the testing data
print("SVM Report after applying SMOTE : \n")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))