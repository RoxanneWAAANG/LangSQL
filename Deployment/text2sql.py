# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

"""
Module: refactored_chatbot
This module provides utilities for loading database schemas, extracting DDL,
indexing content, and a ChatBot class to generate SQL queries from natural language.
"""
import os
import json
import re
import sqlite3
import copy
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from whoosh import index

from utils.db_utils import (
    get_db_schema,
    check_sql_executability,
    get_matched_contents,
    get_db_schema_sequence,
    get_matched_content_sequence
)
from schema_item_filter import SchemaItemClassifierInference, filter_schema


class DatabaseUtils:
    """
    Utilities for loading database comments, schemas, and DDL statements.
    """

    @staticmethod
    def _remove_similar_comments(names, comments):
        """
        Remove comments identical to table/column names (ignoring underscores and spaces).
        """
        filtered = []
        for name, comment in zip(names, comments):
            normalized_name = name.replace("_", "").replace(" ", "").lower()
            normalized_comment = comment.replace("_", "").replace(" ", "").lower()
            filtered.append("") if normalized_name == normalized_comment else filtered.append(comment)
        return filtered

    @staticmethod
    def load_db_comments(table_json_path):
        """
        Load additional comments for tables and columns from a JSON file.

        Args:
            table_json_path (str): Path to JSON file containing table and column comments.

        Returns:
            dict: Mapping from database ID to comments structure.
        """
        additional_info = json.load(open(table_json_path))
        db_comments = {}

        for db_info in additional_info:
            db_id = db_info["db_id"]
            comment_dict = {}

            # Process column comments
            original_cols = db_info["column_names_original"]
            col_names = [col.lower() for _, col in original_cols]
            col_comments = [c.lower() for _, c in db_info["column_names"]]
            col_comments = DatabaseUtils._remove_similar_comments(col_names, col_comments)
            col_table_idxs = [t_idx for t_idx, _ in original_cols]

            # Process table comments
            original_tables = db_info["table_names_original"]
            tbl_names = [tbl.lower() for tbl in original_tables]
            tbl_comments = [c.lower() for c in db_info["table_names"]]
            tbl_comments = DatabaseUtils._remove_similar_comments(tbl_names, tbl_comments)

            for idx, name in enumerate(tbl_names):
                comment_dict[name] = {
                    "table_comment": tbl_comments[idx],
                    "column_comments": {}
                }
                # Associate columns
                for t_idx, col_name, col_comment in zip(col_table_idxs, col_names, col_comments):
                    if t_idx == idx:
                        comment_dict[name]["column_comments"][col_name] = col_comment

            db_comments[db_id] = comment_dict

        return db_comments

    @staticmethod
    def get_db_schemas(db_path, tables_json):
        """
        Build a mapping from database ID to its schema representation.

        Args:
            db_path (str): Directory containing database subdirectories.
            tables_json (str): Path to JSON with table comments.

        Returns:
            dict: Mapping from db_id to schema object.
        """
        comments = DatabaseUtils.load_db_comments(tables_json)
        schemas = {}
        for db_id in tqdm(os.listdir(db_path), desc="Loading schemas"):
            sqlite_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
            schemas[db_id] = get_db_schema(sqlite_path, comments, db_id)
        return schemas

    @staticmethod
    def get_db_ddls(db_path):
        """
        Extract formatted DDL statements for all tables in each database.

        Args:
            db_path (str): Directory containing database subdirectories.

        Returns:
            dict: Mapping from db_id to its DDL string.
        """
        ddls = {}
        for db_id in os.listdir(db_path):
            conn = sqlite3.connect(os.path.join(db_path, db_id, f"{db_id}.sqlite"))
            cursor = conn.cursor()
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
            ddl_statements = []

            for name, raw_sql in cursor.fetchall():
                sql = raw_sql or ""
                sql = re.sub(r'--.*', '', sql).replace("\t", " ")
                sql = re.sub(r" +", " ", sql)
                formatted = sqlparse.format(
                    sql,
                    keyword_case="upper",
                    identifier_case="lower",
                    reindent_aligned=True
                )
                # Adjust spacing for readability
                formatted = formatted.replace(", ", ",\n    ")
                if formatted.rstrip().endswith(";"):
                    formatted = formatted.rstrip()[:-1] + "\n);"
                formatted = re.sub(r"(CREATE TABLE.*?)\(", r"\1(\n    ", formatted)
                ddl_statements.append(formatted)

            ddls[db_id] = "\n\n".join(ddl_statements)
        return ddls


class ChatBot:
    """
    ChatBot for generating and executing SQL queries using a causal language model.
    """

    def __init__(self, model_name: str = "seeklhy/codes-1b", device: str = "cuda:0") -> None:
        """
        Initialize the ChatBot with model and tokenizer.

        Args:
            model_name (str): HuggingFace model identifier.
            device (str): CUDA device string or 'cpu'.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.max_length = 4096
        self.max_new_tokens = 256
        self.max_prefix_length = self.max_length - self.max_new_tokens

        # Schema item classifier
        self.schema_classifier = SchemaItemClassifierInference("Roxanne-WANG/LangSQL")

        # Initialize content searchers
        self.content_searchers = {}
        index_dir = "db_contents_index"
        for db_id in os.listdir(index_dir):
            path = os.path.join(index_dir, db_id)
            if index.exists_in(path):
                self.content_searchers[db_id] = index.open_dir(path).searcher()
            else:
                raise FileNotFoundError(f"Whoosh index not found for '{db_id}' at '{path}'")

        # Load schemas and DDLs
        self.db_ids = sorted(os.listdir("databases"))
        self.schemas = DatabaseUtils.get_db_schemas("databases", "data/tables.json")
        self.ddls = DatabaseUtils.get_db_ddls("databases")

    def get_response(self, question: str, db_id: str) -> str:
        """
        Generate an executable SQL query for a natural language question.

        Args:
            question (str): User question in natural language.
            db_id (str): Identifier of the target database.

        Returns:
            str: Executable SQL query or an error message.
        """
        # Prepare data
        schema = copy.deepcopy(self.schemas[db_id])
        contents = get_matched_contents(question, self.content_searchers[db_id])
        data = {
            "text": question,
            "schema": schema,
            "matched_contents": contents
        }
        data = filter_schema(data, self.schema_classifier, top_k=6, top_m=10)
        data["schema_sequence"] = get_db_schema_sequence(data["schema"])
        data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        prefix = (
            f"{data['schema_sequence']}\n"
            f"{data['content_sequence']}\n"
            f"{question}\n"
        )

        # Tokenize and ensure length limits
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(prefix)["input_ids"]
        if len(input_ids) > self.max_prefix_length:
            input_ids = [self.tokenizer.bos_token_id] + input_ids[-(self.max_prefix_length - 1):]
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64).to(self.model.device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.int64).to(self.model.device)
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                num_return_sequences=4
            )

        # Decode and choose executable SQL
        decoded = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        final_sql = None
        for sql in decoded:
            if check_sql_executability(sql, os.path.join("databases", db_id, f"{db_id}.sqlite")) is None:
                final_sql = sql.strip()
                break
        if not final_sql:
            final_sql = decoded[0].strip() or "Sorry, I cannot generate a suitable SQL query."

        return final_sql
