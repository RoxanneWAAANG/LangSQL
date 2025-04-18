# Text2SQL -- Naive Model

## Overview
The naive model is a rule-based approach to converting natural language questions into SQL queries. This model uses pattern matching and predefined templates to generate SQL queries without the need for training data. It serves as a baseline model for text-to-SQL tasks, focusing on basic query types such as `COUNT`, `MAX`, `MIN`, `AVG`, and `SELECT`.

## Features
- **Pattern Matching**: Identifies questions that correspond to specific SQL operations like `COUNT`, `MAX`, `MIN`, `AVG`, and `SELECT`.
- **Table and Column Identification**: Extracts relevant tables and columns from the schema based on the question.
- **Fallback Behavior**: If no relevant tables or columns are found, it defaults to a basic `SELECT *` query.

## Usage

### 1. Understanding the Predictions
The model will generate SQL queries based on the patterns found in the question. For example:
- **Question:** "How many singers do we have?"
  - **Predicted SQL:** "SELECT COUNT(*) FROM singer"
- **Question:** "What is the highest capacity of stadiums?"
  - **Predicted SQL:** "SELECT MAX(capacity) FROM concert"

### 2. Example Questions
- **"How many singers do we have?"** – The model will recognize the "how many" pattern and generate a `COUNT` SQL query.
- **"List all concerts"** – Recognizes the "list" pattern and generates a `SELECT` query with all columns.
- **"What is the highest capacity of stadiums?"** – Recognizes the "highest" pattern and generates a `MAX` SQL query.
- **"Show all singers from United States"** – Identifies the relevant table and column and generates a `SELECT` query.

## Limitations
- **Limited Query Types**: The model only supports basic SQL queries such as `COUNT`, `MAX`, `MIN`, `AVG`, and `SELECT`.
- **No Advanced Operations**: The model does not support complex SQL operations such as `JOIN`, `GROUP BY`, or `WHERE` clauses.
- **Pattern-Based**: The model relies on specific question patterns and may not handle more complex or ambiguous queries well.



