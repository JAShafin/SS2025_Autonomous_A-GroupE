import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 1. 加载特征和标签
X = np.load("X_features.npy")   # shape (N, D)
y = np.load("y_labels.npy")     # shape (N,)

print(f"Loaded X: {X.shape}, y: {y.shape}")

# 2. divide testing and training dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# 3. hyper parameter search
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(class_weight='balanced'),
                    param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# 4. use best para to train model
best = grid.best_estimator_

# 5. evaluation
y_pred = best.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
labels = ["straight", "left", "right"]

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
# 6. save the model
joblib.dump(best, "svm_lane_model.pkl")
print("Model saved to svm_lane_model.pkl")
