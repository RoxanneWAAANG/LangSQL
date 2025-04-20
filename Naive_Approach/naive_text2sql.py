# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

import json
import os
import re
from typing import Dict, List, Any

class NaiveText2SQL:
    """
    A naive approach to text-to-SQL conversion using pattern matching and templates.
    This serves as a baseline model that doesn't require training.
    """
    
    def __init__(self):
        # Define common SQL patterns and templates
        self.count_patterns = ["how many", "number of", "count", "total"]
        self.select_patterns = ["list", "show", "what", "find", "give", "tell"]
        self.max_patterns = ["maximum", "highest", "largest", "most"]
        self.min_patterns = ["minimum", "lowest", "smallest", "least"]
        self.avg_patterns = ["average", "mean", "typical"]
        
    def identify_tables(self, question: str, schema: Dict[str, Any]) -> List[str]:
        """
        Identify which tables are mentioned in the question.
        
        Args:
            question: The natural language question
            schema: The database schema information
            
        Returns:
            A list of table names that appear to be relevant to the question
        """
        mentioned_tables = []
        for item in schema["schema_items"]:
            table_name = item["table_name"]
            # Check singular and plural forms
            if (table_name in question.lower() or 
                f"{table_name}s" in question.lower()):
                mentioned_tables.append(table_name)
        
        return mentioned_tables
    
    def identify_columns(self, question: str, schema: Dict[str, Any], tables: List[str]) -> Dict[str, List[str]]:
        """
        Identify which columns are mentioned in the question.
        
        Args:
            question: The natural language question
            schema: The database schema information
            tables: List of relevant tables
            
        Returns:
            A dictionary mapping table names to lists of column names
        """
        mentioned_columns = {table: [] for table in tables}
        
        for item in schema["schema_items"]:
            if item["table_name"] not in tables:
                continue
                
            for col_name in item["column_names"]:
                # Check if column name appears in question
                if col_name in question.lower():
                    mentioned_columns[item["table_name"]].append(col_name)
        
        return mentioned_columns
    
    def predict(self, question: str, schema: Dict[str, Any]) -> str:
        """
        Generate a SQL query based on pattern matching.
        
        Args:
            question: The natural language question
            schema: The database schema information
            
        Returns:
            A SQL query string
        """
        question = question.lower()
        
        # Identify tables and columns
        tables = self.identify_tables(question, schema)
        columns = self.identify_columns(question, schema, tables)
        
        # If no tables were found, try to infer from the question
        if not tables:
            for item in schema["schema_items"]:
                # Check if any column names are mentioned
                for col_name in item["column_names"]:
                    if col_name in question:
                        tables.append(item["table_name"])
                        columns[item["table_name"]] = [col_name]
                        break
        
        # If still no tables were found, use the first table as default
        if not tables and schema["schema_items"]:
            tables = [schema["schema_items"][0]["table_name"]]
            columns = {tables[0]: []}
        
        # Determine the SQL operation
        if any(pattern in question for pattern in self.count_patterns):
            # COUNT operation
            if tables:
                return f"SELECT COUNT(*) FROM {tables[0]}"
        
        elif any(pattern in question for pattern in self.max_patterns):
            # MAX operation
            if tables and columns[tables[0]]:
                return f"SELECT MAX({columns[tables[0]][0]}) FROM {tables[0]}"
        
        elif any(pattern in question for pattern in self.min_patterns):
            # MIN operation
            if tables and columns[tables[0]]:
                return f"SELECT MIN({columns[tables[0]][0]}) FROM {tables[0]}"
        
        elif any(pattern in question for pattern in self.avg_patterns):
            # AVG operation
            if tables and columns[tables[0]]:
                return f"SELECT AVG({columns[tables[0]][0]}) FROM {tables[0]}"
                
        else:
            # Default SELECT operation
            if tables:
                if columns[tables[0]]:
                    cols = ", ".join(columns[tables[0]])
                    return f"SELECT {cols} FROM {tables[0]}"
                else:
                    return f"SELECT * FROM {tables[0]}"
        
        # Fallback query
        return "SELECT * FROM singer LIMIT 5"

# Example usage
if __name__ == "__main__":
    # Load sample data
    with open("dataset/sample_spider_data.json", "r") as f:
        data = json.load(f)
    
    # Create the naive model
    naive_model = NaiveText2SQL()
    
    # Test on sample questions
    sample_questions = [
        "How many singers do we have?",
        "List all concerts",
        "What is the highest capacity of stadiums?",
        "Show all singers from United States"
    ]
    
    for question in sample_questions:
        sql = naive_model.predict(question, data[0]["schema"])
        print(f"Question: {question}")
        print(f"Predicted SQL: {sql}")
        print("---")