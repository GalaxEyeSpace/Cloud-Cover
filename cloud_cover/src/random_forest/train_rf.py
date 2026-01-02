import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load RFCD training data
X = np.load("/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_pixel_training_data/25k_samples/X_TOA_25k.npy")
y = np.load("/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_pixel_training_data/25k_samples/y_TOA_25k.npy")

print("Dataset:", X.shape, y.shape)

# Train / validation split
# Paper uses OOB error
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Initialize RandomForest Classifier model
rf = RandomForestClassifier(
    n_estimators=100,       # Ntree = 100
    max_features=6,        # Mtry = 6
    max_depth=20,
    min_samples_leaf=50,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

# Train RF model
from time import time
start_time = time()
print("Training RFCD model...")
rf.fit(X_train, y_train)
end_time = time()
print("Training time: %.2f seconds" % (end_time - start_time))

print("\nOOB accuracy:", rf.oob_score_)
print("OOB error:", 1.0 - rf.oob_score_)

y_pred = rf.predict(X_val)

# Validation metrics
print("\nConfusion matrix (validation set)")
print(confusion_matrix(y_val, y_pred))

print("\nClassification report (validation set)")
print(classification_report(y_val, y_pred, digits=4))

# Save trained RFCD model
out = "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_model_25k.joblib"
joblib.dump(rf, out)

print("\nSaved RFCD model:", out)
