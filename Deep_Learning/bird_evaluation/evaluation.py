# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Parallel SQL execution and accuracy evaluation for predicted vs. ground-truth queries.

Attribution:
- func_timeout: https://github.com/ZuJiaYu/func_timeout
- SQLite via Python standard library
"""
import sys
import os
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut


def load_json(json_path: str) -> dict:
    """
    Load and return JSON content from file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class SQLExecutor:
    """
    Safely execute SQL queries against a SQLite database with timeout.
    """
    def __init__(self, timeout: float = 30.0):
        """
        Args:
            timeout (float): Seconds before query execution times out.
        """
        self.timeout = timeout

    def _run(self, predicted: str, ground_truth: str, db_path: str) -> bool:
        """
        Execute predicted and ground-truth SQL and compare results.

        Args:
            predicted (str): SQL string to execute.
            ground_truth (str): Reference SQL string.
            db_path (str): Path to SQLite database file.

        Returns:
            bool: True if result sets match (as sets), False otherwise.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(predicted)
        pred_rows = cur.fetchall()
        cur.execute(ground_truth)
        true_rows = cur.fetchall()
        conn.close()
        return set(pred_rows) == set(true_rows)

    def execute(self, task: tuple) -> dict:
        """
        Wrapper for multiprocessing: runs with timeout and catches errors.

        Args:
            task (tuple): (idx, predicted, ground_truth, db_path)

        Returns:
            dict: {'sql_idx': idx, 'res': 1 or 0}
        """
        idx, predicted, ground_truth, db_path = task
        try:
            success = func_timeout(self.timeout, self._run, args=(predicted, ground_truth, db_path))
            result = 1 if success else 0
        except FunctionTimedOut:
            result = 0
        except Exception:
            result = 0
        return {'sql_idx': idx, 'res': result}


class SQLPackager:
    """
    Package predicted and ground-truth SQLs with associated database paths.
    """
    def __init__(self, db_root: str):
        """
        Args:
            db_root (str): Root directory containing database subfolders.
        """
        self.db_root = db_root

    def load(self, sql_path: str, mode: str, data_mode: str) -> list:
        """
        Extract SQL statements and associated DB paths.

        Args:
            sql_path (str): Path to JSON or SQL file.
            mode (str): 'gpt' for predictions, 'gt' for ground truth.
            data_mode (str): Dataset split (e.g., 'dev', 'test').

        Returns:
            list of str: SQL statements.
            list of str: Corresponding DB file paths.
        """
        sqls, db_paths = [], []
        if mode == 'gpt':
            data = load_json(sql_path)
            for idx, sql_entry in data.items():
                sql, db = sql_entry.split('\t----- bird -----\t') if isinstance(sql_entry, str) else ('', 'unknown')
                sqls.append(sql)
                db_paths.append(f"{self.db_root}{db}/{db}.sqlite")
        else:  # ground truth
            lines = open(f"{sql_path}{data_mode}_gold.sql", 'r').read().splitlines()
            for line in lines:
                sql, db = line.strip().split('\t')
                sqls.append(sql)
                db_paths.append(f"{self.db_root}{db}/{db}.sqlite")
        return sqls, db_paths


class AccuracyCalculator:
    """
    Compute accuracy metrics overall and by difficulty level.
    """
    @staticmethod
    def compute(results: list, diff_path: str) -> tuple:
        """
        Args:
            results (list of dict): [{'sql_idx': int, 'res': 0/1}, ...]
            diff_path (str): Path to difficulty JSON with 'difficulty' labels.

        Returns:
            tuple: (simple_acc, moderate_acc, challenging_acc, overall_acc, counts)
        """
        res_map = {r['sql_idx']: r['res'] for r in results}
        diffs = load_json(diff_path)
        categories = {'simple': [], 'moderate': [], 'challenging': []}
        for i, entry in enumerate(diffs):
            d = entry.get('difficulty')
            categories.setdefault(d, []).append(res_map.get(i, 0))
        counts = [len(categories[c]) for c in ['simple','moderate','challenging']]
        counts.append(len(results))
        accs = [
            sum(categories[c]) / len(categories[c]) * 100 if categories[c] else 0.0
            for c in ['simple','moderate','challenging']
        ]
        overall = sum(res_map.values()) / len(results) * 100
        return *accs, overall, counts

    @staticmethod
    def display(accs: tuple, counts: list) -> None:
        """
        Print formatted accuracy table.
        """
        levels = ['simple','moderate','challenging','total']
        header = f"{{:>10}}" + "{:>12}"*4
        print(header.format('', *levels))
        print(header.format('count', *counts))
        print('-'*60)
        print(header.format('accuracy', *[f"{a:.2f}" for a in accs]))


class EvaluationRunner:
    """
    Orchestrate packaging, parallel execution, and reporting.
    """
    def __init__(self, args):
        self.args = args
        self.executor = SQLExecutor(timeout=args.meta_time_out)
        self.packager = SQLPackager(db_root=args.db_root_path)

    def run(self):
        # Load predicted and ground truth SQLs
        preds, dbs_pred = self.packager.load(
            self.args.predicted_sql_path, self.args.mode_predict, self.args.data_mode
        )
        gts, _ = self.packager.load(
            self.args.ground_truth_path, self.args.mode_gt, self.args.data_mode
        )
        tasks = [ (i, preds[i], gts[i], dbs_pred[i]) for i in range(len(preds)) ]

        # Parallel execution
        with mp.Pool(self.args.num_cpus) as pool:
            results = pool.map(self.executor.execute, tasks)

        # Sort by index
        results.sort(key=lambda x: x['sql_idx'])

        # Compute and display accuracies
        accs = AccuracyCalculator.compute(results, self.args.diff_json_path)
        AccuracyCalculator.display(accs[:-1], accs[-1])


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SQL predictions against ground truth.'
    )
    parser.add_argument('--predicted_sql_path', type=str, required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--data_mode', type=str, required=True)
    parser.add_argument('--db_root_path', type=str, required=True)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--meta_time_out', type=float, default=30.0)
    parser.add_argument('--mode_gt', type=str, default='gt')
    parser.add_argument('--mode_predict', type=str, default='gpt')
    parser.add_argument('--diff_json_path', type=str, default='')
    args = parser.parse_args()

    runner = EvaluationRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
