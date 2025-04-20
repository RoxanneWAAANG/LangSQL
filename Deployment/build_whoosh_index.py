# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

"""
Module: db_indexer
Provides utilities to extract short text content from SQLite databases and build Whoosh indices.
"""
import os
import sqlite3
from typing import List, Tuple
from whoosh import index
from whoosh.fields import Schema, ID, TEXT


class DBIndexer:
    """
    Extracts text content from SQLite databases and builds/searches Whoosh indices.
    """
    def __init__(self, index_root: str = "db_contents_index") -> None:
        """
        Initialize the index root directory.

        Args:
            index_root: Base directory to store indices per database.
        """
        self.index_root = index_root
        os.makedirs(self.index_root, exist_ok=True)

    @staticmethod
    def extract_contents_from_db(db_path: str, max_len: int = 25) -> List[Tuple[str, str]]:
        """
        Extract all unique, non-null text entries up to max_len characters from every table/column.

        Args:
            db_path: Path to the SQLite file.
            max_len: Maximum length for text fields to index.

        Returns:
            A list of (doc_id, text) tuples.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        docs: List[Tuple[str, str]] = []

        # Iterate through user tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table,) in cur.fetchall():
            if table == "sqlite_sequence":
                continue

            # Get column names
            cur.execute(f"PRAGMA table_info('{table}')")
            cols = [row[1] for row in cur.fetchall()]

            for col in cols:
                query = f"SELECT DISTINCT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL"
                for (val,) in cur.execute(query):
                    text = str(val).strip()
                    if 0 < len(text) <= max_len:
                        doc_id = f"{table}-{col}-{hash(text)}"
                        docs.append((doc_id, text))

        conn.close()
        return docs

    def build_index_for_db(self, db_id: str, db_path: str) -> index.Index:
        """
        Build or open a Whoosh index for a given database.

        Args:
            db_id: Identifier for the database (used as subdirectory).
            db_path: Path to the SQLite database file.

        Returns:
            A Whoosh Index object for searching.
        """
        index_dir = os.path.join(self.index_root, db_id)
        os.makedirs(index_dir, exist_ok=True)

        # Define Whoosh schema
        schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True)
        )

        # Open existing index if available
        if index.exists_in(index_dir):
            return index.open_dir(index_dir)

        # Otherwise create and populate new index
        ix = index.create_in(index_dir, schema)
        writer = ix.writer()
        docs = DBIndexer.extract_contents_from_db(db_path)
        for doc_id, text in docs:
            writer.add_document(id=doc_id, content=text)
        writer.commit()
        return ix


def main(databases_root: str = "databases", index_root: str = "db_contents_index") -> None:
    """
    Build Whoosh indices for all SQLite databases in a root folder.

    Args:
        databases_root: Directory containing subdirectories per database with .sqlite files.
        index_root: Directory to store Whoosh indices.
    """
    indexer = DBIndexer(index_root=index_root)

    # Optionally clear existing indices
    if os.path.isdir(index_root):
        import shutil
        shutil.rmtree(index_root)
    os.makedirs(index_root, exist_ok=True)

    # Process each database
    for db_id in os.listdir(databases_root):
        db_file = os.path.join(databases_root, db_id, f"{db_id}.sqlite")
        if os.path.isfile(db_file):
            print(f"Building index for {db_id}...")
            indexer.build_index_for_db(db_id, db_file)

    print("All indexes built successfully.")


if __name__ == "__main__":
    main()