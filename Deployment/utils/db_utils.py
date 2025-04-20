# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Module: db_utils
Provides utilities for history logging, SQL execution with timeouts, content matching, and schema extraction for Text-to-SQL.
"""
import os
import json
import sqlite3
from typing import List, Tuple, Dict, Any

from func_timeout import func_set_timeout, FunctionTimedOut
from nltk.tokenize import word_tokenize
from nltk import ngrams
from whoosh.qparser import QueryParser

from utils.bridge_content_encoder import get_matched_entries


class HistoryLogger:
    """
    Log user queries into a SQLite history database.
    """

    DB_PATH = 'data/history/history.sqlite'

    @staticmethod
    def add_record(question: str, db_id: str) -> None:
        """
        Insert a new query record into the history database.

        Args:
            question: The user's natural-language question.
            db_id: Identifier of the target database.
        """
        conn = sqlite3.connect(HistoryLogger.DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO record (question, db_id) VALUES (?, ?)",
                (question, db_id)
            )
            conn.commit()
        finally:
            conn.close()


class SQLExecutor:
    """
    Execute SQL queries with optional time limits and error handling.
    """

    @staticmethod
    def _get_cursor(sqlite_path: str) -> sqlite3.Cursor:
        """
        Create or retrieve a SQLite cursor for the given path.

        Args:
            sqlite_path: Filesystem path to the .sqlite file.

        Returns:
            sqlite3.Cursor with text_factory to ignore decode errors.
        """
        if not os.path.exists(sqlite_path):
            print(f"Opening new connection: {sqlite_path}")
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        return conn.cursor()

    @staticmethod
    @func_set_timeout(15)
    def execute(cursor: sqlite3.Cursor, sql: str) -> List[Tuple]:
        """
        Execute a SQL statement with a short timeout.

        Args:
            cursor: SQLite cursor.
            sql: SQL string to execute.

        Returns:
            List of result tuples.

        Raises:
            FunctionTimedOut if execution exceeds 15 seconds.
        """
        cursor.execute(sql)
        return cursor.fetchall()

    @staticmethod
    @func_set_timeout(2000)
    def execute_long(cursor: sqlite3.Cursor, sql: str) -> List[Tuple]:
        """
        Execute a SQL statement with a longer timeout (for indexing).
        """
        cursor.execute(sql)
        return cursor.fetchall()

    @staticmethod
    def check_executability(generated_sql: str, db_path: str) -> Any:
        """
        Check if a SQL query runs without error or timeout.

        Args:
            generated_sql: The SQL query string.
            db_path: Path to the SQLite database.

        Returns:
            None if successful, or an error message string.
        """
        sql = generated_sql.strip()
        if not sql:
            return "Error: empty query"
        try:
            cursor = SQLExecutor._get_cursor(db_path)
            SQLExecutor.execute(cursor, sql)
            return None
        except FunctionTimedOut:
            return "Error: execution timed out"
        except Exception as e:
            return str(e)


class ContentMatcher:
    """
    Extract and match content snippets from database columns using n-grams and Whoosh.
    """

    @staticmethod
    def obtain_n_grams(text: str, max_n: int) -> List[str]:
        """
        Generate all n-grams up to max_n from the input text.

        Args:
            text: Input string to tokenize.
            max_n: Maximum n-gram length.

        Returns:
            List of n-gram strings.
        """
        # Ensure tokenizers are downloaded
        import nltk
        nltk.download('punkt', quiet=True)

        tokens = word_tokenize(text)
        grams = []
        for n in range(1, max_n + 1):
            grams += [' '.join(gram) for gram in ngrams(tokens, n)]
        return grams

    @staticmethod
    def get_matched_contents(
        question: str,
        searcher
    ) -> Dict[str, List[str]]:
        """
        Perform coarse and fine-grained matching of question against indexed content.

        Args:
            question: User query.
            searcher: Whoosh Index searcher instance.

        Returns:
            Mapping from 'table.column' to matched values.
        """
        # Coarse matching via n-grams
        grams = ContentMatcher.obtain_n_grams(question, 4)
        coarse: Dict[str, List[str]] = {}

        for gram in grams:
            qp = QueryParser('content', schema=searcher.schema)
            q = qp.parse(gram)
            for hit in searcher.search(q, limit=10):
                entry = json.loads(hit.raw)
                tc = '.'.join(entry['id'].split('-')[:2])
                coarse.setdefault(tc, []).append(entry['contents'])

        # Fine-grained filtering
        fine: Dict[str, List[str]] = {}
        for tc, vals in coarse.items():
            fm = get_matched_entries(question, vals)
            if not fm:
                continue
            for match_str, (value, _, score, _, _) in fm:
                if score >= 0.9:
                    fine.setdefault(tc, []).append(value.strip())
                    if len(fine[tc]) >= 25:
                        break
        return fine


class SchemaBuilder:
    """
    Extract schema details and format sequences for model prompting.
    """

    @staticmethod
    def detect_special_char(name: str) -> bool:
        """
        Check if a name contains schema-special characters.
        """
        return any(c in name for c in ['(', ')', '-', ' ', '/'])

    @staticmethod
    def quote(name: str) -> str:
        """
        Wrap a name in backticks for SQL.
        """
        return f'`{name}`'

    @staticmethod
    def get_column_contents(
        column: str,
        table: str,
        cursor: sqlite3.Cursor
    ) -> List[str]:
        """
        Retrieve example values for a column (max 2) and filter short strings.
        """
        sql = f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT 2;"
        rows = SQLExecutor.execute_long(cursor, sql)
        return [str(r[0]).strip() for r in rows if 0 < len(str(r[0]).strip()) <= 25]

    @staticmethod
    def build_schema_sequence(
        schema: Dict[str, Any]
    ) -> str:
        """
        Format the schema dict into a prompt string for the LLM.
        """
        seq = ['database schema:']
        for item in schema['schema_items']:
            tbl = item['table_name']
            if SchemaBuilder.detect_special_char(tbl):
                tbl = SchemaBuilder.quote(tbl)
            cols_info = []
            for col, typ, com, cont, pk in zip(
                item['column_names'],
                item['column_types'],
                item['column_comments'],
                item['column_contents'],
                item['pk_indicators']
            ):
                if SchemaBuilder.detect_special_char(col):
                    col = SchemaBuilder.quote(col)
                info = [typ]
                if pk:
                    info.append('primary key')
                if com:
                    info.append(f'comment: {com}')
                if cont:
                    info.append('values: ' + ', '.join(cont))
                cols_info.append(f"{tbl}.{col} ({' | '.join(info)})")
            seq.append(f"table {tbl}, columns=[{'; '.join(cols_info)}]")

        fks = schema.get('foreign_keys', [])
        if fks:
            seq.append('foreign keys:')
            for src_tbl, src_col, tgt_tbl, tgt_col in fks:
                seq.append(f"{src_tbl}.{src_col} = {tgt_tbl}.{tgt_col}")
        else:
            seq.append('foreign keys: None')

        return '\n'.join(seq)

    @staticmethod
    def build_content_sequence(
        matched: Dict[str, List[str]]
    ) -> str:
        """
        Format matched contents into a prompt string.
        """
        if not matched:
            return 'matched contents: None'
        lines = ['matched contents:']
        for tc, vals in matched.items():
            tbl, col = tc.split('.')
            if SchemaBuilder.detect_special_char(tbl):
                tbl = SchemaBuilder.quote(tbl)
            if SchemaBuilder.detect_special_char(col):
                col = SchemaBuilder.quote(col)
            lines.append(f"{tbl}.{col} ({', '.join(vals)})")
        return '\n'.join(lines)


def get_db_schema(
    db_path: str,
    comments: Dict[str, Any],
    db_id: str
) -> Dict[str, Any]:
    """
    Load table/column metadata and comments from a SQLite DB into a schema dict.

    Args:
        db_path: Path to the .sqlite file.
        comments: Mapping of db_id to comment structures.
        db_id: Identifier for looking up comments.

    Returns:
        Schema dictionary with items and foreign_keys.
    """
    db_comment = comments.get(db_id, {})
    cursor = SQLExecutor._get_cursor(db_path)

    # Retrieve table names
    tables = SQLExecutor.execute(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0].lower() for r in tables if r[0].lower() != 'sqlite_sequence']

    schema = {'schema_items': [], 'foreign_keys': []}
    for tbl in tables:
        # Columns
        cols_info = SQLExecutor.execute(cursor, f"PRAGMA table_info('{tbl}')")
        col_names = [r[1].lower() for r in cols_info]
        col_types = [r[2].lower() for r in cols_info]
        pk_inds  = [r[5] for r in cols_info]

        # Contents
        col_contents = [
            SchemaBuilder.get_column_contents(col, tbl, cursor)
            for col in col_names
        ]

        # Foreign keys
        fks = SQLExecutor.execute(cursor, f"PRAGMA foreign_key_list('{tbl}');")
        for fk in fks:
            schema['foreign_keys'].append([
                tbl, fk[3].lower(), fk[2].lower(), fk[4].lower()
            ])

        # Comments
        tbl_comm = db_comment.get(tbl, {}).get('table_comment', '')
        col_comms = [
            db_comment.get(tbl, {}).get('column_comments', {}).get(col, '')
            for col in col_names
        ]

        schema['schema_items'].append({
            'table_name': tbl,
            'table_comment': tbl_comm,
            'column_names': col_names,
            'column_types': col_types,
            'column_comments': col_comms,
            'column_contents': col_contents,
            'pk_indicators': pk_inds
        })
    return schema
