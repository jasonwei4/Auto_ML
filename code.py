# Install required packages if not already installed
!pip install autogluon pandas scikit-learn matplotlib

# Import necessary libraries
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Verify the environment
import sys
print("Python executable:", sys.executable)  # Should output /Users/jason/Desktop/vscode/.venv/bin/python

# Load the Titanic dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Split training data into train and validation sets
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['Survived'])
print(f"\nTraining set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

# Initialize AutoGluon TabularPredictor
predictor = TabularPredictor(
    label='Survived',  # Target column
    path='autogluon_model',  # Path to save the model
    eval_metric='f1',  # Metric to optimize (F1 score for binary classification)
    verbosity=2  # Show detailed logs
)

# Train the model
predictor.fit(
    train_data=train_set,
    time_limit=300,  # Limit training time to 300 seconds
    presets='medium_quality'  # Use medium quality preset for faster training
)

# Display the leaderboard of models
print("\nModel leaderboard:")
print(predictor.leaderboard())

# Predict probabilities on the validation set
y_val = val_set['Survived']
y_pred_proba = predictor.predict_proba(val_set, as_multiclass=False)[:, 1]  # Get probabilities for the positive class (Survived=1)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
average_precision = average_precision_score(y_val, y_pred_proba)

# Plot the precision-recall curve
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {average_precision:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve for Titanic Survival Prediction')
# plt.legend()
# plt.grid(True)
# plt.show()

# Print the average precision score
print(f"\nAverage Precision: {average_precision:.3f}")

# Select a threshold based on the F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
print(f"\nBest threshold (based on F1 score): {best_threshold:.3f}")
print(f"Corresponding precision: {precision[best_threshold_idx]:.3f}")
print(f"Corresponding recall: {recall[best_threshold_idx]:.3f}")

# Apply the threshold to the validation set predictions
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Evaluate classification performance on the validation set
print("\nClassification performance on validation set (using best threshold):")
print(f"Precision: {precision_score(y_val, y_pred):.3f}")
print(f"Recall: {recall_score(y_val, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.3f}")
print("\nDetailed classification report:")
print(classification_report(y_val, y_pred, target_names=['Not Survived', 'Survived']))

# Predict on the test set using the best threshold
test_pred_proba = predictor.predict_proba(test_data, as_multiclass=False)[:, 1]
test_pred = (test_pred_proba >= best_threshold).astype(int)

# Save the predictions to a submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to submission.csv")
print(submission.head())