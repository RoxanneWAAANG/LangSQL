# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

"""
Module: ml_text2sql
Implements a classic ML-based Text-to-SQL system using TF-IDF and Random Forests.
Provides training, prediction, and model persistence utilities.
"""
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
    """
    A machine-learning-based Text-to-SQL translator.

    Uses TF-IDF features combined with schema pattern features to train
    classifiers for tables, operations (SELECT, COUNT, etc.), joins, conditions, and ordering.
    """
    def __init__(self):
        """
        Initialize vectorizers, classifiers, and label encoders.
        """
        self.vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
        self.table_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.operation_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.join_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.condition_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.order_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.operation_encoder = LabelEncoder()
        self.table_encoder = MultiLabelBinarizer()
        self.operations = ["SELECT", "COUNT", "MAX", "MIN", "AVG", "SUM"]
        self.is_fitted = False

    def extract_features(self, question, schema):
        """
        Extract numeric feature vector from question text and schema.

        Args:
            question (str): Natural-language question.
            schema (dict): Schema dictionary with 'schema_items'.

        Returns:
            np.ndarray: Concatenated TF-IDF, schema, and pattern features.
        """
        # TF-IDF features or zeros if not yet fitted
        question_features = (
            self.vectorizer.transform([question]).toarray().flatten()
            if self.is_fitted else np.zeros(self.vectorizer.max_features)
        )
        # Binary features for presence of each table name
        schema_features = np.array([
            int(table["table_name"].lower() in question.lower())
            for table in schema["schema_items"]
        ])
        # Pattern-based lexical features
        pattern_features = np.array([
            int(any(k in question.lower() for k in kws))
            for kws in [
                ['count', 'how many'], ['maximum', 'highest', 'most'],
                ['minimum', 'lowest', 'least'], ['average', 'mean'],
                ['total', 'sum'], ['order', 'sort'], ['limit', 'top', 'first'],
                ['group', 'each'], ['greater', 'more than', 'larger'],
                ['less', 'smaller', 'lower', 'fewer'], ['equal', 'same as', 'is'],
                ['join', 'and', 'with']
            ]
        ])
        return np.concatenate([question_features, schema_features, pattern_features])

    def fit(self, training_data):
        """
        Train the Text-to-SQL classifiers on the provided dataset.

        Args:
            training_data (list): Each item must have 'question', 'schema', 'sql', and 'table_labels'.

        Raises:
            ValueError: If training_data is empty.
        """
        if not training_data:
            raise ValueError("No training data provided.")
        print("Training model...")
        # Fit TF-IDF
        questions = [ex["question"] for ex in training_data]
        self.vectorizer.fit(questions)
        self.is_fitted = True
        # Build feature matrix
        features = []
        max_len = None
        for ex in training_data:
            fv = self.extract_features(ex["question"], ex["schema"])
            if max_len is None:
                max_len = len(fv)
            elif len(fv) != max_len:
                # Pad or truncate to ensure consistent length
                fv = np.pad(fv, (0, max_len - len(fv))) if len(fv) < max_len else fv[:max_len]
            features.append(fv)
        X = np.array(features)
        # Encode operations
        ops = [self._extract_operation(ex["sql"]) for ex in training_data]
        y_ops = self.operation_encoder.fit_transform(ops)
        # Encode tables
        y_tables = [ex["table_labels"] for ex in training_data]
        y_tbls = self.table_encoder.fit_transform(y_tables)
        # Other labels
        y_join = [int("JOIN" in ex["sql"].upper()) for ex in training_data]
        y_cond = [int("WHERE" in ex["sql"].upper()) for ex in training_data]
        y_ord = [int("ORDER BY" in ex["sql"].upper()) for ex in training_data]
        print(f"Training classifiers with X.shape={X.shape}, y_tables.shape={y_tbls.shape}")
        # Fit classifiers
        self.table_classifier.fit(X, y_tbls)
        self.operation_classifier.fit(X, y_ops)
        self.join_classifier.fit(X, y_join)
        self.condition_classifier.fit(X, y_cond)
        self.order_classifier.fit(X, y_ord)
        print("Model training complete.")

    def predict(self, question, schema):
        """
        Predict an SQL query given a question and schema.

        Args:
            question (str): Natural-language question.
            schema (dict): Schema dictionary used for feature extraction.

        Returns:
            str: Generated SQL query.
        """
        features = self.extract_features(question, schema)
        # Align feature vector length
        exp_len = self.table_classifier.n_features_in_
        features = (
            np.pad(features, (0, exp_len - len(features)))
            if len(features) < exp_len else features[:exp_len]
        )
        # Operation
        op_idx = self.operation_classifier.predict([features])[0]
        op = self.operation_encoder.inverse_transform([op_idx])[0]
        # Flags
        has_join = self.join_classifier.predict([features])[0]
        has_cond = self.condition_classifier.predict([features])[0]
        has_ord = self.order_classifier.predict([features])[0]
        # Table selection
        tbl_enc = self.table_classifier.predict([features])[0]
        tbl_idxs = np.where(tbl_enc == 1)[0]
        if len(tbl_idxs) == 0:
            # fallback by schema keywords
            kw = np.array([int(t["table_name"].lower() in question.lower()) for t in schema["schema_items"]])
            tbl_idxs = np.where(kw == 1)[0] or [0]
        tables = [schema["schema_items"][i]["table_name"] for i in tbl_idxs if i < len(schema["schema_items"])]
        tables = tables or ["unknown_table"]
        return self._generate_sql(op, tables, has_join, has_cond, has_ord, question)

    def _extract_operation(self, sql):
        """
        Identify the SQL operation keyword in a query.
        """
        up = sql.upper()
        for op in self.operations:
            if op in up:
                return op
        return "SELECT"

    def _generate_sql(self, operation, tables, has_join, has_condition, has_order, question):
        """
        Construct an SQL string from predicted components.
        """
        # Base select or aggregate
        sql = (
            f"SELECT * FROM {tables[0]}"
            if operation == "SELECT" else f"SELECT {operation}(*) FROM {tables[0]}"
        )
        # JOIN
        if has_join and len(tables) > 1:
            sql += f" JOIN {tables[1]} ON {tables[0]}.{tables[0]}_id = {tables[1]}.{tables[0]}_id"
        # WHERE
        if has_condition:
            nums = re.findall(r'\d+', question)
            cond = nums[0] if nums else "IS NOT NULL"
            sql += f" WHERE {tables[0]}_id > {cond}" if nums else f" WHERE {tables[0]}_id IS NOT NULL"
        # ORDER BY
        if has_order:
            direction = "DESC" if "highest" in question.lower() else "ASC"
            sql += f" ORDER BY {operation}(*) {direction}"
            if "top" in question.lower():
                sql += " LIMIT 1"
        return sql

    def save_model(self, model_path):
        """
        Serialize and save the trained ML_Text2SQL instance.

        Args:
            model_path (str): File path to write the pickle.
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path):
        """
        Load a serialized ML_Text2SQL instance from disk.

        Args:
            model_path (str): Path to the pickle file.

        Returns:
            ML_Text2SQL: The loaded model object.
        """
        with open(model_path, 'rb') as f:
            return pickle.load(f)


def main():
    """
    Entry point for training and evaluating the ML Text-to-SQL model.
    """
    data_path = "dataset/sample_spider_data.json"
    model_path = "ml_text2sql_model.pkl"
    # Load dataset
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    # Split train/test
    split = int(len(data) * 0.8)
    train, test = data[:split], data[split:]
    # Train
    model = ML_Text2SQL()
    model.fit(train)
    model.save_model(model_path)
    # Evaluate
    correct = 0
    results = []
    for i, ex in enumerate(test):
        pred = model.predict(ex['question'], ex['schema'])
        is_corr = pred.lower() == ex['sql'].lower()
        correct += is_corr
        results.append({
            'id': i, 'question': ex['question'],
            'true_sql': ex['sql'], 'predicted_sql': pred,
            'is_correct': is_corr
        })
    acc = correct / len(test) if test else 0
    print(f"Final accuracy: {acc:.4f}")
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({'accuracy': acc, 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
