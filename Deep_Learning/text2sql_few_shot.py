# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Few-shot cross-domain text-to-SQL generation and evaluation pipeline.

Attribution:
- HuggingFace Transformers: https://github.com/huggingface/transformers
- SimCSE: https://github.com/princeton-nlp/SimCSE
"""
import argparse
import json
import os
import time
from typing import List, Dict

import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from tqdm import tqdm

from simcse import SimCSE
from schema_item_filter import SchemaItemClassifierInference, filter_schema
from utils.db_utils import (
    check_sql_executability,
    detect_special_char,
    get_db_schema_sequence,
    get_matched_content_sequence
)
from utils.load_sft_dataset import SFTSQLGenerationDataset


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for model, data, and generation settings.
    """
    parser = argparse.ArgumentParser(
        description="Few-shot cross-domain text-to-SQL pipeline"
    )
    parser.add_argument('--llm_path', type=str, default='seeklhy/codes-1b',
                        help='Model path or HuggingFace ID')
    parser.add_argument('--sic_path', type=str, required=True,
                        help='Schema-item classifier checkpoint path')
    parser.add_argument('--table_num', type=int, default=5,
                        help='Max tables in context')
    parser.add_argument('--column_num', type=int, default=6,
                        help='Max columns per table')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to eval JSON dataset')
    parser.add_argument('--demonstration_set_path', type=str, required=True,
                        help='Path to demonstration JSON pool')
    parser.add_argument('--num_of_demonstrations', type=int, default=1,
                        help='Number of few-shot examples')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Input token limit')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Generated token limit')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


class SQLPostProcessor:
    """
    Clean and finalize generated SQL for execution.
    """
    @staticmethod
    def clean(sql: str, schema_items: List[Dict]) -> str:
        # Remove newlines
        sql = sql.replace('\n', ' ')
        # Quote special columns
        for table in schema_items:
            for col in table.get('column_names', []):
                if detect_special_char(col) and col in sql:
                    sql = sql.replace(col, f'`{col}`')
        # Collapse double backticks
        while '``' in sql:
            sql = sql.replace('``', '`')
        # Keep only first statement and terminate
        stmt = sql.split(';')[0].strip()
        return stmt + ';'


class SkeletonExtractor:
    """
    Extracts a simplified POS-based skeleton from text.
    """
    @staticmethod
    def extract(text: str) -> str:
        tokens_tags = nltk.pos_tag(nltk.word_tokenize(text))
        out = []
        for tok, tag in tokens_tags:
            if tag in ('NN','NNP','NNS','NNPS','CD','SYM','FW','IN'):
                out.append('_')
            elif tok in {'$','"','(',')',',','--','.',':'}:
                continue
            else:
                out.append(tok)
        skeleton = ' '.join(out).replace("_ 's", '_').replace(" 's", "'s")
        # remove repeats
        while '_ _' in skeleton:
            skeleton = skeleton.replace('_ _', '_')
        return skeleton.lstrip('_ ')


class DemonstrationSelector:
    """
    Selects top-k similar demonstrations for few-shot prompting.
    """
    def __init__(self, simcse_model: SimCSE, k: int):
        self.model = simcse_model
        self.k = k

    def select(self, eval_qs: List[str], demo_qs: List[str]) -> List[List[int]]:
        # Compute raw and skeleton similarities, then take max
        sim_raw = self.model.similarity(eval_qs, demo_qs)
        sims = []
        for i, q in enumerate(eval_qs):
            sims_sk = self.model.similarity([
                SkeletonExtractor.extract(q)
            ], [SkeletonExtractor.extract(demo_qs[j]) for j in range(len(demo_qs))])[0]
            sims.append(np.maximum(sim_raw[i], sims_sk))
        return [list(np.argsort(-s)[:self.k]) for s in sims]


class Text2SQLGenerator:
    """
    Handles model loading and generation.
    """
    def __init__(self, model_path: str, max_new_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.float16
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = next(self.model.parameters()).device
        # adjust eos token for SQL termination
        ids = self.tokenizer('SELECT * FROM tables ;')['input_ids']
        new_eos = ids[-2] if ids[-1] == self.tokenizer.eos_token_id else ids[-1]
        self.model.config.eos_token_id = new_eos
        self.tokenizer.eos_token_id = new_eos

    def generate(self, prompt: str, max_input_len: int) -> List[str]:
        ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()
        if len(ids) > max_input_len:
            ids = [self.tokenizer.bos_token_id] + ids[-(max_input_len-1):]
        input_ids = torch.tensor([ids], device=self.device)
        mask = torch.ones_like(input_ids)
        out = self.model.generate(
            input_ids=input_ids, attention_mask=mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=4, num_return_sequences=4
        )
        return self.tokenizer.batch_decode(
            out[:, input_ids.size(1):],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class EvaluationPipeline:
    """
    Orchestrates data loading, demonstration prep, generation, and evaluation.
    """
    def __init__(self, args: argparse.Namespace):
        set_seed(args.seed)
        # Load data
        self.eval_set = json.load(open(args.dataset_path, 'r', encoding='utf-8'))
        self.demo_set = json.load(open(args.demonstration_set_path, 'r', encoding='utf-8'))
        # Filter schema
        sic = SchemaItemClassifierInference(args.sic_path)
        self.demo_set = filter_schema(self.demo_set, 'train', None, args.table_num, args.column_num)
        self.eval_set = filter_schema(self.eval_set, 'eval', sic, args.table_num, args.column_num)
        del sic; torch.cuda.empty_cache()
        # Add sequences
        for item in self.demo_set:
            item['schema_sequence'] = get_db_schema_sequence(item['schema'])
            item['content_sequence'] = get_matched_content_sequence(item['matched_contents'])
        for item in self.eval_set:
            item['schema_sequence'] = get_db_schema_sequence(item['schema'])
            item['content_sequence'] = get_matched_content_sequence(item['matched_contents'])
        # Similarity and selection
        sim_model = SimCSE('princeton-nlp/sup-simcse-roberta-base')
        sel = DemonstrationSelector(sim_model, args.num_of_demonstrations)
        qs = [i['question'] for i in self.eval_set]
        dqs = [i['question'] for i in self.demo_set]
        self.topk = sel.select(qs, dqs)
        del sim_model
        self.generator = Text2SQLGenerator(args.llm_path, args.max_new_tokens)
        self.args = args

    def run(self) -> None:
        preds = []
        for idx, ex in enumerate(tqdm(self.eval_set, desc='Eval examples')):
            # Build prompt
            seqs = self.topk[idx]
            prompt = ''
            for j in seqs:
                demo = self.demo_set[j]
                prompt += f"{demo['schema_sequence']}\n{demo['content_sequence']}\n{demo['question']} ;\n{demo['sql'].rstrip(';')} ;\n\n"
            prompt += f"{ex['schema_sequence']}\n{ex['content_sequence']}\n{ex['question']}\n"
            # Generate and post-process
            cands = self.generator.generate(prompt, self.args.max_tokens - self.args.max_new_tokens)
            sqls = [SQLPostProcessor.clean(s, ex['schema']['schema_items']) for s in cands]
            # Pick first executable
            final = next((s for s in sqls if check_sql_executability(s, ex['db_path']) is None), sqls[0] or 'SQL placeholder')
            preds.append(final)
        # Save and evaluate
        out_file = 'pred_sqls.txt'
        with open(out_file, 'w', encoding='utf-8') as f:
            for s in preds: f.write(s + '\n')
        # Dataset-specific hook
        if 'spider' in self.args.dataset_path:
            os.system(
                f"python -u test_suite_sql_eval/evaluation.py --gold data/.../dev_gold.sql "
                f"--pred {out_file} --db data/.../database --etype exec"
            )


def main() -> None:
    args = parse_args()
    pipeline = EvaluationPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
