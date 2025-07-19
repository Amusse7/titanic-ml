import pandas as pd
import joblib
from preprocess import preprocess

# Load model and test data
model = joblib.load('src/model.pkl')
test_df = pd.read_csv('data/test.csv')
X_test = preprocess(test_df, is_train=False)

# Predict
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('output/submission.csv', index=False)

print("Submission file created at output/submission.csv")
