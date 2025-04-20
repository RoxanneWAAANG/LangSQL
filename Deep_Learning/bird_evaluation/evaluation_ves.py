# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Compute and evaluate Value Estimation Scores (VES) by comparing execution times
of predicted versus ground-truth SQL queries on SQLite databases.

Attribution:
- func_timeout: https://github.com/ZuJiaYu/func_timeout
- SQLite: Python standard library
"""
import sys
import os
import json
import math
import time
import sqlite3
import argparse
import multiprocessing as mp
from typing import List, Dict, Tuple

import numpy as np
from func_timeout import func_timeout, FunctionTimedOut


def load_json(path: str) -> List[Dict]:
    """
    Load JSON file and return its contents.

    Args:
        path (str): Path to JSON file.

    Returns:
        List[Dict]: Parsed JSON data.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def remove_outliers(data: List[float]) -> List[float]:
    """
    Remove values more than 3 standard deviations from the mean.

    Args:
        data (List[float]): List of numeric values.

    Returns:
        List[float]: Filtered list without outliers.
    """
    arr = np.array(data)
    m, std = arr.mean(), arr.std()
    return [x for x in arr if abs(x - m) <= 3 * std]


def time_query(sql: str, db_path: str) -> float:
    """
    Execute a SQL query and measure its execution time.

    Args:
        sql (str): SQL query string.
        db_path (str): Path to SQLite database.

    Returns:
        float: Execution time in seconds.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    start = time.time()
    cur.execute(sql)
    conn.close()
    return time.time() - start


def compute_time_ratio(
    predicted: str,
    ground_truth: str,
    db_path: str,
    iterations: int
) -> float:
    """
    Compute average ratio of ground-truth to predicted execution times,
    after removing outlier measurements.

    Args:
        predicted (str): Predicted SQL query.
        ground_truth (str): Reference SQL query.
        db_path (str): Path to SQLite database.
        iterations (int): Number of timing iterations.

    Returns:
        float: Average time ratio, or 0 on mismatch.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(predicted)
    res_pred = cur.fetchall()
    cur.execute(ground_truth)
    res_true = cur.fetchall()
    conn.close()

    if set(res_pred) != set(res_true):
        return 0.0

    ratios = []
    for _ in range(iterations):
        t_pred = time_query(predicted, db_path)
        t_true = time_query(ground_truth, db_path)
        ratios.append(t_true / t_pred if t_pred > 0 else 0.0)

    filtered = remove_outliers(ratios)
    return sum(filtered) / len(filtered) if filtered else 0.0


def worker_task(args: Tuple[int,str,str,str,int,float]) -> Dict:
    """
    Worker function for multiprocessing pool.

    Args:
        args: (idx, predicted, ground_truth, db_path, iterations, timeout)

    Returns:
        Dict: {'sql_idx': idx, 'time_ratio': computed ratio}
    """
    idx, pred, truth, db_path, iters, timeout = args
    try:
        ratio = func_timeout(timeout * iters, compute_time_ratio,
                             args=(pred, truth, db_path, iters))
    except FunctionTimedOut:
        ratio = 0.0
    except Exception:
        ratio = 0.0
    return {'sql_idx': idx, 'time_ratio': ratio}


def package_sqls(
    sql_path: str,
    db_root: str,
    mode: str,
    split: str
) -> Tuple[List[str], List[str]]:
    """
    Load predicted or ground-truth SQLs and corresponding DB paths.

    Args:
        sql_path (str): Path to JSON or SQL file prefix.
        db_root (str): Root directory for databases.
        mode (str): 'gpt' for predictions, 'gt' for ground truth.
        split (str): Data split identifier (e.g., 'dev').

    Returns:
        Tuple[List[str], List[str]]: SQL list and DB path list.
    """
    sqls, dbs = [], []
    if mode == 'gpt':
        data = load_json(sql_path)
        for idx, entry in data.items():
            sql, db = entry.split('\t----- bird -----\t') if isinstance(entry, str) else ('', '')
            sqls.append(sql)
            dbs.append(os.path.join(db_root, db, f"{db}.sqlite"))
    else:
        lines = open(f"{sql_path}{split}_gold.sql").read().splitlines()
        for line in lines:
            sql, db = line.split('\t')
            sqls.append(sql)
            dbs.append(os.path.join(db_root, db, f"{db}.sqlite"))
    return sqls, dbs


def compute_ves_by_difficulty(
    results: List[Dict],
    diff_path: str
) -> Tuple[List[float], List[int]]:
    """
    Compute VES metrics per difficulty category and overall.

    Args:
        results (List[Dict]): [{ 'sql_idx': int, 'time_ratio': float }, ...]
        diff_path (str): Path to difficulty JSON file.

    Returns:
        Tuple[List[float], List[int]]: VES scores and counts.
    """
    diffs = load_json(diff_path)
    buckets = {'simple': [], 'moderate': [], 'challenging': []}
    for res in results:
        diff = diffs[res['sql_idx']]['difficulty']
        buckets[diff].append(res['time_ratio'])

    scores, counts = [], []
    for level in ['simple', 'moderate', 'challenging']:
        vals = buckets[level]
        ves = math.sqrt(sum(vals) / len(vals)) * 100 if vals else 0.0
        scores.append(ves)
        counts.append(len(vals))
    overall = math.sqrt(sum(r['time_ratio'] for r in results) / len(results)) * 100
    scores.append(overall)
    counts.append(len(results))
    return scores, counts


def display_scores(scores: List[float], counts: List[int]) -> None:
    """
    Print a formatted table of VES scores and counts.

    Args:
        scores (List[float]): VES percentages for each category and total.
        counts (List[int]): Item counts for categories and total.
    """
    levels = ['simple','moderate','challenging','total']
    print(f"{'':>10}{''.join(f'{lvl:>12}' for lvl in levels)}")
    print(f"{'count':>10}{''.join(f'{c:12d}' for c in counts)}")
    print('-'*60)
    print(f"{'VES':>10}{''.join(f'{s:12.2f}' for s in scores)}")


def main() -> None:
    """Main execution: parse args, run parallel tasks, and report VES."""
    parser = argparse.ArgumentParser(description='Evaluate SQL VES metrics.')
    parser.add_argument('--predicted_sql_path', required=True)
    parser.add_argument('--ground_truth_path', required=True)
    parser.add_argument('--data_mode', required=True)
    parser.add_argument('--db_root_path', required=True)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--meta_time_out', type=float, default=30.0)
    parser.add_argument('--mode_predict', default='gpt')
    parser.add_argument('--mode_gt', default='gt')
    parser.add_argument('--diff_json_path', default='')
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()

    # Package SQL and DB lists
    preds, dbs = package_sqls(
        args.predicted_sql_path, args.db_root_path, args.mode_predict, args.data_mode
    )
    gts, _ = package_sqls(
        args.ground_truth_path, args.db_root_path, args.mode_gt, args.data_mode
    )

    # Prepare tasks and execute in parallel
    tasks = [ (i, preds[i], gts[i], dbs[i], args.iterations, args.meta_time_out) for i in range(len(preds)) ]
    with mp.Pool(args.num_cpus) as pool:
        results = pool.map(worker_task, tasks)

    results.sort(key=lambda x: x['sql_idx'])

    # Compute and display VES by difficulty
    scores, counts = compute_ves_by_difficulty(results, args.diff_json_path)
    display_scores(scores, counts)

if __name__ == '__main__':
    main()
