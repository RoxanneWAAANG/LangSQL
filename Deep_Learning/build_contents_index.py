# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Builds BM25 Lucene indexes for database column contents using Pyserini.

Attribution:
- Pyserini project: https://github.com/castorini/pyserini
- SQLite utilities from utils.db_utils
"""
import os
import shutil
import json
import subprocess
from typing import List, Dict

from utils.db_utils import get_cursor_from_path, execute_sql_long_time_limitation


def remove_directory_contents(path: str) -> None:
    """
    Ensure a directory exists and remove all its contents.

    Args:
        path (str): Path to the directory to clean.
    """
    os.makedirs(path, exist_ok=True)
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        try:
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.unlink(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
        except Exception as e:
            print(f"Failed to delete {full_path}: {e}")


class ContentIndexer:
    """
    Handles extraction of column contents from SQLite databases
    and creation of Lucene indexes via Pyserini.
    """
    def __init__(self, temp_dir: str = './data/temp_db_index'):
        """
        Initialize the temporary directory for intermediate JSON files.

        Args:
            temp_dir (str): Path for writing temporary JSON collections.
        """
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def _extract_column_contents(self, db_path: str) -> List[Dict[str, str]]:
        """
        Query distinct non-null column values up to length limit.

        Args:
            db_path (str): Path to the SQLite database file.

        Returns:
            List[Dict[str,str]]: A list of documents with 'id' and 'contents'.
        """
        cursor = get_cursor_from_path(db_path)
        tables = execute_sql_long_time_limitation(cursor,
            "SELECT name FROM sqlite_master WHERE type='table';")
        docs = []
        for (table,) in tables:
            if table == 'sqlite_sequence':
                continue
            cols = execute_sql_long_time_limitation(
                cursor, f"PRAGMA table_info('{table}')")
            col_names = [row[1] for row in cols]
            for col in col_names:
                try:
                    query = (
                        f"SELECT DISTINCT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL;"
                    )
                    rows = execute_sql_long_time_limitation(cursor, query)
                    for idx, (val,) in enumerate(rows):
                        text = str(val).strip()
                        if 0 < len(text) <= 25:
                            doc_id = f"{table}-**-{col}-**-{idx}".lower()
                            docs.append({'id': doc_id, 'contents': text})
                except Exception as e:
                    print(f"Error processing {table}.{col}: {e}")
        return docs

    def build_index(self, db_path: str, index_path: str) -> None:
        """
        Build a Lucene index for a single database's column contents.

        Args:
            db_path (str): Path to the SQLite file.
            index_path (str): Directory where the Lucene index will be stored.
        """
        # Clean existing index directory
        remove_directory_contents(index_path)

        # Extract contents and write to temporary JSON
        docs = self._extract_column_contents(db_path)
        temp_file = os.path.join(self.temp_dir, 'contents.json')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)

        # Construct Pyserini indexing command
        cmd = [
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', self.temp_dir,
            '--index', index_path,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', '16',
            '--storePositions',
            '--storeDocvectors',
            '--storeRaw'
        ]
        subprocess.run(cmd, check=True)

        # Clean up temporary file
        os.remove(temp_file)


def prepare_indices(db_root: str, index_root: str) -> None:
    """
    Walk through databases under db_root to build indexes in parallel directory structure under index_root.

    Args:
        db_root (str): Root directory containing subdirectories per database ID.
        index_root (str): Root directory for storing per-database indices.
    """
    indexer = ContentIndexer()
    for db_id in os.listdir(db_root):
        db_dir = os.path.join(db_root, db_id)
        sqlite_file = os.path.join(db_dir, f"{db_id}.sqlite")
        if not os.path.isfile(sqlite_file):
            continue
        idx_path = os.path.join(index_root, db_id)
        print(f"Indexing {db_id}...")
        indexer.build_index(sqlite_file, idx_path)


def main() -> None:
    """
    Main entry point: builds content indexes for predefined dataset directories.
    """
    # Example configurations; adjust paths as needed
    configs = [
        ('./data/sft_data_collections/bird/train/train_databases',
         './data/sft_data_collections/bird/train/db_contents_index'),
        ('./data/sft_data_collections/bird/dev/dev_databases',
         './data/sft_data_collections/bird/dev/db_contents_index'),
        ('./data/sft_data_collections/spider/database',
         './data/sft_data_collections/spider/db_contents_index'),
    ]
    for db_root, idx_root in configs:
        print(f"Preparing indices in {idx_root}")
        prepare_indices(db_root, idx_root)

if __name__ == '__main__':
    main()
