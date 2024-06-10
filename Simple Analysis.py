import pandas as pd
# Load the train.csv file
train_df = pd.read_csv('train.csv')
# Display the first few rows of the train dataset
train_df.head()
# Check for missing values in the dataset
missing_values = train_df.isnull().sum()
missing_values
# Fill missing values in Age with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Drop the Cabin column
train_df.drop(columns=['Cabin'], inplace=True)

# Fill missing values in Embarked with the most common port
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Verify that there are no more missing values
missing_values_after = train_df.isnull().sum()
missing_values_after
# Convert 'Sex' into numerical form
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' column
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

# Create a new feature 'FamilySize'
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']

# Drop the 'Name' and 'Ticket' columns as they are not useful for prediction
train_df.drop(columns=['Name', 'Ticket'], inplace=True)

# Display the first few rows of the preprocessed dataset
train_df.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into features (X) and target (y)
X = train_df.drop(columns=['PassengerId', 'Survived'])
y = train_df['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = rf_model.predict(X_val)

# Calculate the accuracy and classification report
accuracy = accuracy_score(y_val, y_val_pred)
classification_report_output = classification_report(y_val, y_val_pred)

accuracy, classification_report_output
# Load the test.csv file
test_df = pd.read_csv('test.csv')

# Display the first few rows of the test dataset
test_df.head()
# Check for missing values in the test dataset
test_missing_values = test_df.isnull().sum()
test_missing_values

# Convert 'Sex' into numerical form
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' column
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Create a new feature 'FamilySize'
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# Drop the 'Name' and 'Ticket' columns as they are not useful for prediction
test_df.drop(columns=['Name', 'Ticket'], inplace=True)

# Ensure all necessary columns are present in the test dataset
# Columns in train dataset excluding 'PassengerId' and 'Survived'
expected_columns = X.columns.tolist()  
test_df = test_df.reindex(columns=expected_columns, fill_value=0)

# Verify that the test dataset has the correct columns
test_df.head()
# Make predictions on the preprocessed test dataset
test_predictions = rf_model.predict(test_df)

# Create the submission dataframe
submission_df = pd.DataFrame({
    'PassengerId': test_df.index + 892,  # Adjusting index to match PassengerId in test.csv
    'Survived': test_predictions
})

# Save the submission dataframe to a CSV file
submission_file_path = 'titanic_submission.csv'
submission_df.to_csv(submission_file_path, index=False)

import ace_tools as tools; # type: ignore #This step is especially useful when working in environments where immediate data visualization is needed, such as in Jupyter notebooks or interactive Python environments. It helps in ensuring that the data looks correct before any further actions, like saving or submitting, are taken.
tools.display_dataframe_to_user(name="Titanic Submission DataFrame", dataframe=submission_df)

submission_file_path
