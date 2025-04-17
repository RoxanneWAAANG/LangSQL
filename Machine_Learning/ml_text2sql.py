import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ML_Text2SQL:
    def __init__(self):
        # Vectorizer for question text
        self.question_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        
        # Classifiers for SQL components
        self.table_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.operation_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Operation labels
        self.operations = ["SELECT", "COUNT", "MAX", "MIN", "AVG"]
        self.operation_encoder = LabelEncoder()
    
    def extract_features(self, question, schema):
        """Extract features from question and schema."""
        # Text features from question
        if not hasattr(self.question_vectorizer, 'vocabulary_'):
            question_features = np.zeros(100)
        else:
            question_features = self.question_vectorizer.transform([question]).toarray()[0]
        
        # Schema features - check if tables/columns are mentioned
        schema_features = []
        question = question.lower()
        
        for item in schema["schema_items"]:
            table_name = item["table_name"].lower()
            table_mentioned = int(table_name in question or f"{table_name}s" in question)
            schema_features.append(table_mentioned)
        
        # Pattern features
        pattern_features = [
            int(any(p in question for p in ["how many", "number", "count", "total"])),
            int(any(p in question for p in ["maximum", "highest", "largest"])),
            int(any(p in question for p in ["minimum", "lowest", "smallest"]))
        ]
        
        return np.concatenate([question_features, np.array(schema_features), np.array(pattern_features)])
    
    def _extract_operation(self, sql):
        """Extract main SQL operation."""
        sql_upper = sql.upper()
        for op in self.operations:
            if op in sql_upper:
                return op
        return "SELECT"
    
    def fit(self, training_data):
        """Train ML model on training data."""
        if not training_data:
            print("Error: No training data provided.")
            return
            
        # Prepare training data
        X_questions = []
        y_tables = []
        y_operations = []
        
        for example in training_data:
            X_questions.append(example["question"])
            y_tables.append(example["table_labels"])
            y_operations.append(self._extract_operation(example["sql"]))
        
        # Fit question vectorizer
        self.question_vectorizer.fit(X_questions)
        
        # Extract full feature vectors
        X = [self.extract_features(ex["question"], ex["schema"]) for ex in training_data]
        X = np.array(X)
        y_tables = np.array(y_tables)
        
        # Fit operation encoder and encode labels
        self.operation_encoder.fit(y_operations)
        y_operations_encoded = self.operation_encoder.transform(y_operations)
        
        # Train classifiers
        self.table_classifier.fit(X, y_tables)
        self.operation_classifier.fit(X, y_operations_encoded)
    
    def predict(self, question, schema):
        """Predict SQL query for a question."""
        # Extract features
        features = self.extract_features(question, schema)
        
        # Predict tables and operation
        predicted_tables = self.table_classifier.predict([features])[0]
        operation_idx = self.operation_classifier.predict([features])[0]
        predicted_operation = self.operation_encoder.inverse_transform([operation_idx])[0]
        
        # Build SQL query based on predictions
        used_tables = []
        for i, used in enumerate(predicted_tables):
            if used == 1 and i < len(schema["schema_items"]):
                used_tables.append(schema["schema_items"][i]["table_name"])
        
        # If no tables predicted, use first table
        if not used_tables and schema["schema_items"]:
            used_tables = [schema["schema_items"][0]["table_name"]]
        
        # Generate SQL based on operation
        if predicted_operation == "COUNT" and used_tables:
            return f"SELECT COUNT(*) FROM {used_tables[0]}"
        elif used_tables:
            return f"SELECT * FROM {used_tables[0]}"
        
        # Fallback query
        return "SELECT * FROM singer LIMIT 5"

# Example usage
if __name__ == "__main__":
    try:
        with open("/home/jack/Projects/yixin-llm/Text2SQL/dataset/sample_spider_data.json", "r") as f:
            data = json.load(f)
        print(f"Loaded dataset with {len(data)} examples")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Split data for training and testing
    train_size = max(1, int(len(data) * 0.8))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create and train ML model
    ml_model = ML_Text2SQL()
    ml_model.fit(train_data)
    
    # Test on examples
    correct = 0
    for i, example in enumerate(test_data):
        question = example["question"]
        schema = example["schema"]
        true_sql = example["sql"]
        
        predicted_sql = ml_model.predict(question, schema)
        
        print(f"Example {i+1}:")
        print(f"Question: {question}")
        print(f"True SQL: {true_sql}")
        print(f"Predicted SQL: {predicted_sql}")
        
        if predicted_sql.lower() == true_sql.lower():
            correct += 1
            print("Correct: Yes")
        else:
            print("Correct: No")
        print("---")
    
    if test_data:
        print(f"Accuracy: {correct / len(test_data):.2f}")