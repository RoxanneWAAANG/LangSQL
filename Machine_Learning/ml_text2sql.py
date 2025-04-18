import json
import numpy as np
import re
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import pickle

class ML_Text2SQL:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
        self.table_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.operation_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.join_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.condition_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.order_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.operation_encoder = LabelEncoder()
        self.table_encoder = MultiLabelBinarizer()  # Added to handle multiple table labels
        self.operations = ["SELECT", "COUNT", "MAX", "MIN", "AVG", "SUM"]
        self.is_fitted = False
        
    def extract_features(self, question, schema):
        """Extract features from the question and schema."""
        question_features = self.vectorizer.transform([question]).toarray().flatten() if self.is_fitted else np.zeros(self.vectorizer.max_features)
        schema_features = np.array([int(table["table_name"].lower() in question.lower()) for table in schema["schema_items"]])
        
        pattern_features = np.array([
            int(any(keyword in question.lower() for keyword in ['count', 'how many'])),
            int(any(keyword in question.lower() for keyword in ['maximum', 'highest', 'most'])),
            int(any(keyword in question.lower() for keyword in ['minimum', 'lowest', 'least'])),
            int(any(keyword in question.lower() for keyword in ['average', 'mean'])),
            int(any(keyword in question.lower() for keyword in ['total', 'sum'])),
            int(any(keyword in question.lower() for keyword in ['order', 'sort'])),
            int(any(keyword in question.lower() for keyword in ['limit', 'top', 'first'])),
            int(any(keyword in question.lower() for keyword in ['group', 'each'])),
            int(any(keyword in question.lower() for keyword in ['greater', 'more than', 'larger'])),
            int(any(keyword in question.lower() for keyword in ['less', 'smaller', 'lower', 'fewer'])),
            int(any(keyword in question.lower() for keyword in ['equal', 'same as', 'is'])),
            int(any(keyword in question.lower() for keyword in ['join', 'and', 'with']))
        ])
        
        return np.concatenate([question_features, schema_features, pattern_features])

    def fit(self, training_data):
        """Train the model."""
        if not training_data:
            raise ValueError("No training data provided.")
        
        print("Training model...")
        questions = [example["question"] for example in training_data]
        self.vectorizer.fit(questions)
        self.is_fitted = True
        
        # Extract features and ensure consistent shape
        features = []
        max_feature_length = None
        
        for example in training_data:
            feature_vector = self.extract_features(example["question"], example["schema"])
            
            # Store the maximum feature length
            if max_feature_length is None:
                max_feature_length = len(feature_vector)
            elif len(feature_vector) != max_feature_length:
                # Ensure features are consistent in length
                if len(feature_vector) < max_feature_length:
                    feature_vector = np.pad(feature_vector, (0, max_feature_length - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:max_feature_length]
            
            features.append(feature_vector)
        
        X = np.array(features)  # Convert list of consistent-sized features to numpy array
        
        # Prepare labels for training
        y_operations = []
        
        # Extract operations from SQL queries
        for example in training_data:
            y_operations.append(self._extract_operation(example["sql"]))
        
        y_operations_encoded = self.operation_encoder.fit_transform(y_operations)
        
        # Use MultiLabelBinarizer for table labels
        y_tables = [example["table_labels"] for example in training_data]
        y_tables_encoded = self.table_encoder.fit_transform(y_tables)
        
        y_join = [int("JOIN" in example["sql"].upper()) for example in training_data]
        y_condition = [int("WHERE" in example["sql"].upper()) for example in training_data]
        y_order = [int("ORDER BY" in example["sql"].upper()) for example in training_data]
        
        print(f"Training classifiers with feature matrix shape: {X.shape}")
        print(f"Table labels shape after encoding: {y_tables_encoded.shape}")
        
        # Fit the classifiers with properly encoded table labels
        self.table_classifier.fit(X, y_tables_encoded)
        self.operation_classifier.fit(X, y_operations_encoded)
        self.join_classifier.fit(X, y_join)
        self.condition_classifier.fit(X, y_condition)
        self.order_classifier.fit(X, y_order)
        
        print("Model training complete.")

    def predict(self, question, schema):
        """Predict SQL query from the question."""
        features = self.extract_features(question, schema)
        
        # Ensure consistent feature length
        expected_length = self.table_classifier.n_features_in_
        features = np.pad(features, (0, expected_length - len(features))) if len(features) < expected_length else features[:expected_length]
        
        operation_idx = self.operation_classifier.predict([features])[0]
        predicted_operation = self.operation_encoder.inverse_transform([operation_idx])[0]
        
        has_join = self.join_classifier.predict([features])[0]
        has_condition = self.condition_classifier.predict([features])[0]
        has_order = self.order_classifier.predict([features])[0]
        
        # Get binary prediction and convert back to class labels
        pred_tables_encoded = self.table_classifier.predict([features])[0]
        table_indices = np.where(pred_tables_encoded == 1)[0]
        
        # If no tables were predicted, fall back to schema-based selection
        if len(table_indices) == 0:
            schema_features = np.array([int(table["table_name"].lower() in question.lower()) for table in schema["schema_items"]])
            table_indices = np.where(schema_features == 1)[0]
            
            # If still no tables found, use the first table
            if len(table_indices) == 0 and len(schema["schema_items"]) > 0:
                table_indices = [0]
                
        # Convert indices to actual table names
        tables = [schema["schema_items"][i]["table_name"] for i in table_indices if i < len(schema["schema_items"])]
        
        # Fallback if no tables were found
        tables = tables or ["unknown_table"]
        
        return self._generate_sql(predicted_operation, tables, has_join, has_condition, has_order, question)

    def _extract_operation(self, sql):
        """Extract operation (SELECT, COUNT, etc.) from SQL."""
        sql_upper = sql.upper()
        for op in self.operations:
            if op in sql_upper:
                return op
        return "SELECT"
    
    def _generate_sql(self, operation, tables, has_join, has_condition, has_order, question):
        """Generate SQL query from predicted components."""
        sql = f"SELECT * FROM {tables[0]}" if operation == "SELECT" else f"SELECT {operation}(*) FROM {tables[0]}"
        
        if has_join and len(tables) > 1:
            sql += f" JOIN {tables[1]} ON {tables[0]}.{tables[0]}_id = {tables[1]}.{tables[0]}_id"
        
        if has_condition:
            numbers = re.findall(r'\d+', question)
            sql += f" WHERE {tables[0]}_id > {numbers[0]}" if numbers else f" WHERE {tables[0]}_id IS NOT NULL"
        
        if has_order:
            sql += f" ORDER BY {operation}(*) DESC" if "highest" in question.lower() else f" ORDER BY {operation}(*) ASC"
            if "top" in question.lower():
                sql += " LIMIT 1"
        
        return sql
    
    def save_model(self, model_path):
        """Save the trained model to disk."""
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model from disk."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def main():
    """Main function to train and evaluate the model."""
    data_path = "/home/jack/Projects/yixin-llm/yixin-llm-data/Text2SQL/data/sft_spider_dev_text2sql.json"
    model_save_path = "ml_text2sql_model.pkl"
    
    # Load data
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Initialize and train the model
    model = ML_Text2SQL()
    model.fit(train_data)
    model.save_model(model_save_path)
    
    # Evaluate model on test set
    correct = 0
    total = len(test_data)
    results = []
    
    for i, example in enumerate(test_data):
        question, schema, true_sql = example["question"], example["schema"], example["sql"]
        predicted_sql = model.predict(question, schema)
        
        is_correct = predicted_sql.lower() == true_sql.lower()
        correct += is_correct
        results.append({"id": i, "question": question, "true_sql": true_sql, "predicted_sql": predicted_sql, "is_correct": is_correct})
    
    accuracy = correct / total
    print(f"Final accuracy: {accuracy:.4f}")
    
    # Save evaluation results
    with open("evaluation_results.json", 'w') as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()