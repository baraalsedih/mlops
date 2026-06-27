"""
BEFORE: Training without any experiment tracking.

Problems trainees should notice:
- Where did this model come from? Which hyperparameters?
- How does it compare to last week's run?
- Can we reproduce it?
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=data.target_names))

# Save model — but where? which version? trained on what data?
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
