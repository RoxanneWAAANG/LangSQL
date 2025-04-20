# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Prepare Seq2SQL fine-tuning dataset in Spider style, with optional evidence.

Attribution:
- sqlparse: https://github.com/andialbrecht/sqlparse
- pyserini LuceneSearcher: https://github.com/castorini/pyserini
- sql_metadata Parser: https://github.com/macbre/sql-metadata
- NLTK utilities
"""
import json
import os
import re
import random
from typing import List, Dict, Any

import sqlparse
from nltk import ngrams, word_tokenize
from sql_metadata import Parser
from pyserini.search.lucene import LuceneSearcher

from utils.bridge_content_encoder import get_matched_entries
from utils.db_utils import get_db_schema

random.seed(42)


def extract_large_numbers(text: str) -> str:
    """
    Convert word-based large numbers (e.g., '3 million') into numeric form.
    Returns semicolon-separated evidence entries.
    """
    mappings = {'thousand': 10**3, 'million': 10**6, 'billion': 10**9, 'trillion': 10**12}
    infos = []
    for key, mult in mappings.items():
        for match in re.findall(r"(\d+\.?\d*)\s*%s" % key, text, flags=re.IGNORECASE):
            val = int(float(match) * mult)
            infos.append(f"{match} {key} = {val}")
    for phrase, mult in {'thousands of':10**3,'millions of':10**6}.items():
        if phrase in text.lower():
            infos.append(f"{phrase} = {mult}")
    return "; ".join(infos)


def sanitize_sql_alias(sql: str) -> str:
    """
    Remove table aliases and restore original table names in SQL.
    """
    try:
        aliases = Parser(sql).tables_aliases
    except Exception:
        return sql
    for alias, tbl in aliases.items():
        sql = re.sub(rf"\bAS\s+{alias}\b", '', sql, flags=re.IGNORECASE)
        sql = re.sub(rf"\b{alias}\b", tbl, sql)
    return sql


class EvidencePreprocessor:
    """
    Clean and normalize textual evidence relative to schema items.
    """
    @staticmethod
    def remove_similar(names: List[str], comments: List[str]) -> List[str]:
        """Remove comments identical to names (ignoring spaces/underscores)."""
        out, clean = [], None
        for nm, cm in zip(names, comments):
            if nm.replace('_','').replace(' ','') == cm.replace('_','').replace(' ', ''):
                out.append('')
            else:
                out.append(cm)
        return out

    @staticmethod
    def preprocess(evidence: str, schema: List[Dict[str,Any]]) -> str:
        """
        Ensure evidence ends with semicolon, normalize case for schema mentions.
        """
        ev = evidence.strip()
        if not ev: return ''
        if not ev.endswith(';'): ev += ';'
        for tbl in schema:
            nm = tbl['table_name']
            ev = re.sub(re.escape(nm), nm, ev, flags=re.IGNORECASE)
            for col in tbl['column_names']:
                ev = re.sub(re.escape(col), col, ev, flags=re.IGNORECASE)
        return ev.replace('< =', '<=').replace('> =','>=')


def get_ngrams(text: str, max_n: int=4) -> List[str]:
    """Return all n-grams up to length max_n from text."""
    tokens = word_tokenize(text)
    grams = []
    for n in range(1, max_n+1):
        grams.extend([' '.join(gram) for gram in ngrams(tokens, n)])
    return grams


class SpiderStyleLoader:
    """
    Load and prepare a Spider-style dataset with optional evidence and content matching.
    """
    def __init__(
        self,
        dataset_path: str,
        db_dir: str,
        index_dir: str,
        table_info_path: str,
        use_evidence: bool,
        mode: str
    ):
        self.data = json.load(open(dataset_path, 'r', encoding='utf-8'))
        self.db_dir = db_dir
        self.index_dir = index_dir
        self.comments = self._load_table_comments(table_info_path)
        self.use_evidence = use_evidence
        self.mode = mode
        self.searchers = {}

    def _load_table_comments(self, path: str) -> Dict[str, Any]:
        info = json.load(open(path, 'r', encoding='utf-8'))
        out = {}
        for db in info:
            tbls = {t.lower():c.lower() for t,c in zip(
                [n for _,n in db['table_names_original']],
                [c for _,c in db['table_names']]
            )}
            cols = {}
            names = [n.lower() for _,n in db['column_names_original']]
            comms = [c.lower() for _,c in db['column_names']]
            comms = EvidencePreprocessor.remove_similar(names, comms)
            for (ti,nm), cm in zip(db['column_names_original'], comms):
                cols.setdefault(db['table_names_original'][ti].lower(), {})[nm.lower()] = cm
            out[db['db_id']] = {'table_comments':tbls, 'column_comments':cols}
        return out

    def _get_searcher(self, db_id: str) -> LuceneSearcher:
        if db_id not in self.searchers:
            idx = os.path.join(self.index_dir, db_id)
            self.searchers[db_id] = LuceneSearcher(idx)
        return self.searchers[db_id]

    def load(self) -> List[Dict[str,Any]]:
        result = []
        for entry in self.data:
            db_id = entry['db_id']
            sample = {'db_id': db_id}
            sample['db_path'] = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            sample['schema'] = get_db_schema(sample['db_path'], self.comments, db_id)

            # question and evidence
            q = entry.get('question', entry.get('SpiderSynQuestion', ''))
            ev = entry.get('evidence','') if 'bird' in self.mode else ''
            if 'bank' in self.mode:
                ev = extract_large_numbers(q)
            ev = EvidencePreprocessor.preprocess(ev, sample['schema']['schema_items'])
            sample['text'] = f"{ev} {q}" if self.use_evidence and ev else q.replace('\n',' ')

            # SQL for train/dev
            if self.mode in ('train','dev'):
                raw_sql = entry.get('SQL') or entry.get('query','')
                sample['sql'] = sanitize_sql_alias(sqlparse.format(raw_sql, keyword_case='upper', identifier_case='lower'))
            else:
                sample['sql'] = ''

            # generate labels
            tokens = [t.value for t in Parser(sample['sql'].lower()).tokens] if sample['sql'] else []
            sample['table_labels'], sample['column_labels'] = [], []
            for tbl in sample['schema']['schema_items']:
                tnm = tbl['table_name']
                sample['table_labels'].append(int(tnm.lower() in tokens))
                cls = []
                for cn in tbl['column_names']:
                    cnm = f"{tnm.lower()}.{cn.lower()}"
                    cls.append(int(cn.lower() in tokens or cnm in tokens))
                sample['column_labels'].append(cls)

            # content matching
            grams = get_ngrams(sample['text'])
            hits = []
            searcher = self._get_searcher(db_id)
            for g in grams:
                hits.extend(searcher.search(g, k=10))
            coarse = {}
            for h in hits:
                rec = json.loads(h.raw)
                key = '.'.join(rec['id'].split('-**-')[:2])
                coarse.setdefault(key, set()).add(rec['contents'])
            fine = {}
            for k,v in coarse.items():
                for txt in get_matched_entries(sample['text'], list(v)) or []:
                    if txt[2] >= 0.9:
                        fine.setdefault(k, []).append(txt[0].strip())
            sample['matched_contents'] = fine
            result.append(sample)
        return result


def main():
    """Prepare and save various SFT datasets."""
    modes = [
        ('spider-train','./data/sft_data_collections/spider/train.json', False),
        ('bird-train', './data/.../bird/train.json', True),
        ('bank-train','./data/.../bank.json', True)
    ]
    # expand as needed per dataset
    for mode, path, use_ev in modes:
        loader = SpiderStyleLoader(
            dataset_path=path,
            db_dir='./data/.../database',
            index_dir='./data/.../db_contents_index',
            table_info_path='./data/.../tables.json',
            use_evidence=use_ev,
            mode=mode.split('-')[-1]
        )
        ds = loader.load()
        out_path = f"./data/sft_{mode}_text2sql.json"
        open(out_path,'w').write(json.dumps(ds, indent=2))

if __name__ == '__main__':
    main()
