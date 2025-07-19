import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from preprocess import preprocess

# Load data
train_df = pd.read_csv('data/train.csv')
X = preprocess(train_df)
y = train_df['Survived']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {acc:.4f}')

# Save model
joblib.dump(model, 'src/model.pkl')
