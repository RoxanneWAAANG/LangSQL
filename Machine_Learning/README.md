# Text2SQL -- Machine Learning Approach

## Overview
The machine learning model is trained to extract features from text input (questions) and a database schema, then generate corresponding SQL queries using classification techniques. This tool is built using scikit-learn's RandomForestClassifier and other machine learning methods to process the input data and generate SQL commands such as `SELECT`, `COUNT`, `MAX`, `MIN`, etc.

## Features
- **Text-to-SQL Conversion**: Converts natural language questions to SQL queries.
- **Operations and Tables**: Supports various SQL operations like `SELECT`, `COUNT`, `AVG`, and more.
- **Join, Where, Order**: Identifies JOIN conditions, WHERE clauses, and ORDER BY clauses.
- **Multi-Table Handling**: Supports handling multiple tables in SQL queries.
- **Training**: Trainable on a custom dataset for improving the accuracy of predictions.
- **Model Serialization**: Trained models can be saved and loaded for later use.

## Usage

### 1. Training the Model
To train the model on a custom dataset, use the following command:

```python
from ml_text2sql import ML_Text2SQL

data_path = "/path/to/your/training_data.json"
model_save_path = "ml_text2sql_model.pkl"

# Load data
with open(data_path, "r") as f:
    data = json.load(f)

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Initialize and train the model
model = ML_Text2SQL()
model.fit(train_data)

# Save the trained model
model.save_model(model_save_path)
```

### 2. Predicting SQL Queries
Once the model is trained, you can use it to predict SQL queries from natural language questions:

```python
# Load the trained model
model = ML_Text2SQL.load_model("ml_text2sql_model.pkl")

# Define a question and schema
question = "How many employees are in the company?"
schema = {
    "schema_items": [
        {"table_name": "employees"},
        {"table_name": "departments"}
    ]
}

# Predict the SQL query
predicted_sql = model.predict(question, schema)
print(predicted_sql)
```

### 3. Evaluating the Model
To evaluate the model's performance on a test set:

```python
# Evaluate the model on the test set
correct = 0
total = len(test_data)
for example in test_data:
    question, schema, true_sql = example["question"], example["schema"], example["sql"]
    predicted_sql = model.predict(question, schema)
    
    if predicted_sql.lower() == true_sql.lower():
        correct += 1

accuracy = correct / total
print(f"Final accuracy: {accuracy:.4f}")
```

### 4. Saving and Loading the Model
To save and load the trained model, you can use the following commands:

```python
# Save the model
model.save_model("ml_text2sql_model.pkl")

# Load the model
loaded_model = ML_Text2SQL.load_model("ml_text2sql_model.pkl")
```

## Files
- **ml_text2sql.py**: Main code for training, predicting, and saving the model.
- **evaluation_results.json**: Stores the evaluation results of the model on the test dataset.

