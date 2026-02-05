# import
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data load, split
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Scaler 
scaler = StandardScaler()

# ✅ train에만 fit
X_train_scaled = scaler.fit_transform(X_train)

# ✅ test는 transform만
X_test_scaled = scaler.transform(X_test)

print("Scaled sample:\n", X_train_scaled[:3])

clf = LogisticRegression(
    max_iter=500,
    random_state=42
)

clf.fit(X_train_scaled, y_train)
print("Training done.")

y_pred = clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print("Report:\n", report)

