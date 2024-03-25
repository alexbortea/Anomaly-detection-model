import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import numpy as np

# Load the train and test datasets
train = pd.read_csv('train.csv')  # Update the path to your train.csv file
test = pd.read_csv('test.csv')    # Update the path to your test.csv file

print(test.head())

print(train.head())
# Extract features and the target from the training data
X_train = train[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
y_train = train['is_anomaly']
X_test = test[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Splitting the training data into training and validation sets
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Computing class weights due to imbalance in the data
class_weights_rf = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_rf), y=y_train_rf)
class_weights_dict_rf = {i: class_weights_rf[i] for i in range(len(class_weights_rf))}

# Initializing and training the Random Forest Classifier with class weight balancing
rf_model = RandomForestClassifier(class_weight=class_weights_dict_rf, random_state=42, n_estimators=100)
rf_model.fit(X_train_rf, y_train_rf)

# Making predictions on the validation set
y_val_pred_rf = rf_model.predict(X_val_rf)

# Evaluating accuracy on the validation set
accuracy_val_rf = accuracy_score(y_val_rf, y_val_pred_rf)

# Adjusting the decision threshold for Random Forest to detect more anomalies
rf_threshold = 0.3
test_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of being an anomaly
y_pred_rf = (test_prob_rf >= rf_threshold).astype(int)

test['is_anomaly'] = y_pred_rf

# 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

normal_points = test[test['is_anomaly'] == 0]
anomaly_points = test[test['is_anomaly'] == 1]

ax.scatter(normal_points['feature_0'], normal_points['feature_1'], normal_points['feature_2'], c='blue', label='Normal')
ax.scatter(anomaly_points['feature_0'], anomaly_points['feature_1'], anomaly_points['feature_2'], c='red', label='Anomaly')

ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 2')
ax.set_title('3D Scatter Plot of Features')

ax.legend()
plt.show()

# Preparing the final output
final_output_rf = pd.DataFrame({'id': test['id'], 'is_anomaly': y_pred_rf})

# Exporting the results to a CSV file
final_output_rf.to_csv('predictions15.csv', index=False)

print(f"Accuracy on Validation Set: {accuracy_val_rf}")
print(f"Number of anomalies detected with Random Forest: {sum(y_pred_rf)}")
