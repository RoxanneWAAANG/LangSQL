# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Utilities for schema-item classification, splitting, merging, and filtering
using a fine-tuned classifier.
"""
import json
import random
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

from utils.classifier_model import SchemaItemClassifier


def lista_contains_listb(lista: List[int], listb: List[int]) -> bool:
    """
    Check if all elements of listb are present in lista.

    Args:
        lista (List[int]): Superset list of indices.
        listb (List[int]): Subset list of indices to verify.

    Returns:
        bool: True if all elements of listb are in lista, False otherwise.
    """
    return all(b in lista for b in listb)


class InputPreparer:
    """
    Prepare tokenized inputs and index mappings for schema-item classifier.
    """
    @staticmethod
    def prepare(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> Any:
        """
        Tokenize question and schema items, and record token indices for each table/column.

        Args:
            sample (Dict[str, Any]): Single data point containing 'text' and 'schema'.
            tokenizer (AutoTokenizer): HuggingFace tokenizer instance.

        Returns:
            Tuple containing input IDs, attention mask, 
            column token indices, table token indices, and column counts per table.
        """
        # Extract table and column names
        schema_items = sample['schema']['schema_items']
        table_word_indices = []
        column_word_indices = []
        input_words = [sample['text']]

        # Build word sequence with separators
        for table in schema_items:
            input_words.extend(['|', table['table_name'], ':'])
            table_word_indices.append(len(input_words) - 2)
            for col in table['column_names']:
                input_words.append(col)
                column_word_indices.append(len(input_words) - 1)
            input_words.pop()  # remove trailing comma placeholder

        # Tokenize with word splitting
        tokens = tokenizer(
            input_words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        word_ids = tokens.word_ids(batch_index=0)

        # Map words to token positions
        col_token_indices = [
            [i for i, wid in enumerate(word_ids) if wid == widx]
            for widx in column_word_indices
        ]
        table_token_indices = [
            [i for i, wid in enumerate(word_ids) if wid == widx]
            for widx in table_word_indices
        ]

        # Move tensors to GPU if available
        input_ids = tokens['input_ids'].cuda() if torch.cuda.is_available() else tokens['input_ids']
        attention_mask = tokens['attention_mask'].cuda() if torch.cuda.is_available() else tokens['attention_mask']

        col_counts = []
        start = 0
        for table in schema_items:
            cnt = len(table['column_names'])
            col_counts.append(cnt)
            start += cnt

        return input_ids, attention_mask, col_token_indices, table_token_indices, col_counts


class SchemaSplitter:
    """
    Split samples with long schema sequences into smaller chunks.
    """
    @staticmethod
    def _get_sequence_length(text: str, tc_pairs: List[List[str]], tokenizer: AutoTokenizer) -> int:
        words = [text]
        for t, c in tc_pairs:
            words.extend(['|', t, ':', c])
        tokens = tokenizer(words, is_split_into_words=True)
        return len(tokens['input_ids'])

    @staticmethod
    def _build_schema(tc_pairs: List[List[str]]) -> Dict[str, Any]:
        tables = []
        for t, c in tc_pairs:
            tables.append({'table_name': t, 'column_names': [c]})
        return {'schema_items': tables}

    @classmethod
    def split(cls, sample: Dict[str, Any], tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
        """
        Break a sample into sub-samples each with manageable schema length.
        """
        text = sample['text']
        tc_pairs = []
        # prepare table-column pairs with comments
        for table in sample['schema']['schema_items']:
            tname = table['table_name']
            comment = table.get('table_comment', '')
            display_t = f"{tname} ({comment})" if comment else tname
            for cn, cc in zip(table['column_names'], table.get('column_comments', [])):
                display_c = f"{cn} ({cc})" if cc else cn
                tc_pairs.append([display_t, display_c])

        chunks = []
        buffer = []
        for pair in tc_pairs:
            if cls._get_sequence_length(text, buffer + [pair], tokenizer) < 500:
                buffer.append(pair)
            else:
                chunks.append({'text': text, 'schema': cls._build_schema(buffer)})
                buffer = [pair]
        if buffer:
            chunks.append({'text': text, 'schema': cls._build_schema(buffer)})
        return chunks


class ResultMerger:
    """
    Merge multiple split-sample predictions into final schema-item scores.
    """
    @staticmethod
    def merge(original: Dict[str, Any], partial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate table and column probabilities across splits.
        """
        merged = []
        schema_items = original['schema']['schema_items']
        # prepare names with comments
        names = []
        for tbl in schema_items:
            tname = tbl['table_name']
            cmt = tbl.get('table_comment', '')
            display_t = f"{tname} ({cmt})" if cmt else tname
            col_list = []
            for cn, cc in zip(tbl['column_names'], tbl.get('column_comments', [])):
                display_c = f"{cn} ({cc})" if cc else cn
                col_list.append(display_c)
            names.append((display_t, col_list))

        for display_t, col_list in names:
            best_tp = 0.0
            all_cprobs = []
            for res in partial_results:
                if display_t in res:
                    tp = res[display_t]['table_prob']
                    best_tp = max(best_tp, tp)
                    all_cprobs.extend(res[display_t]['column_probs'])
            merged.append({
                'table_name': display_t,
                'table_prob': best_tp,
                'column_names': col_list,
                'column_probs': all_cprobs
            })
        return merged


class SchemaFilter:
    """
    Filter schema items based on classifier predictions or labels.
    """
    @staticmethod
    def filter(
        dataset: List[Dict[str, Any]],
        mode: str,
        classifier: Any,
        top_tables: int = 5,
        top_columns: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retain top-k tables and columns per example for eval or train.
        """
        for item in tqdm(dataset, desc="Filtering schema items"):
            if mode == 'eval':
                preds = classifier.predict(item)
                table_scores = [preds[t]['table_prob'] for t in preds]
                tbl_idxs = list(np.argsort(-np.array(table_scores))[:top_tables])
            else:  # train
                positive = [i for i, l in enumerate(item['table_labels']) if l]
                neg = [i for i, l in enumerate(item['table_labels']) if not l]
                tbl_idxs = positive + random.sample(neg, min(len(neg), top_tables - len(positive)))

            new_schema = {'schema_items': [], 'foreign_keys': []}
            new_contents = {}

            for ti in tbl_idxs:
                tbl = item['schema']['schema_items'][ti]
                if mode == 'eval':
                    col_scores = preds[tbl['table_name']]['column_probs']
                    col_idxs = list(np.argsort(-np.array(col_scores))[:top_columns])
                else:
                    posc = [i for i, l in enumerate(item['column_labels'][ti]) if l]
                    negc = [i for i, l in enumerate(item['column_labels'][ti]) if not l]
                    col_idxs = posc + random.sample(negc, min(len(negc), top_columns - len(posc)))

                new_schema['schema_items'].append({
                    'table_name': tbl['table_name'],
                    'table_comment': tbl.get('table_comment', ''),
                    'column_names': [tbl['column_names'][ci] for ci in col_idxs],
                    'column_types': [tbl['column_types'][ci] for ci in col_idxs],
                    'column_comments': [tbl['column_comments'][ci] for ci in col_idxs],
                    'column_contents': [tbl['column_contents'][ci] for ci in col_idxs],
                    'pk_indicators': [tbl['pk_indicators'][ci] for ci in col_idxs]
                })
                # filter matched contents
                for cn in new_schema['schema_items'][-1]['column_names']:
                    key = f"{tbl['table_name']}.{cn}"
                    if key in item['matched_contents']:
                        new_contents[key] = item['matched_contents'][key]

            # preserve relevant foreign keys
            keep_tables = {s['table_name'] for s in new_schema['schema_items']}
            for fk in item['schema'].get('foreign_keys', []):
                if fk[0] in keep_tables and fk[2] in keep_tables:
                    new_schema['foreign_keys'].append(fk)

            item['schema'] = new_schema
            item['matched_contents'] = new_contents
        return dataset


class SchemaItemClassifierInference:
    """
    Load a trained schema-item classifier and provide inference methods.
    """
    def __init__(self, model_dir: str):
        """
        Initialize tokenizer and model, and load weights.
        """
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
        self.model = SchemaItemClassifier(model_dir, mode='test')
        self.model.load_state_dict(
            torch.load(f"{model_dir}/dense_classifier.pt", map_location='cpu'), strict=False
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def predict_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict table/column probabilities for a single sample.
        """
        (ids, mask, col_idx, tbl_idx, col_counts) = InputPreparer.prepare(sample, self.tokenizer)
        with torch.no_grad():
            outputs = self.model(
                ids, mask, [col_idx], [tbl_idx], [col_counts]
            )
        tbl_logits = outputs['batch_table_name_cls_logits'][0]
        tbl_probs = torch.softmax(tbl_logits, dim=-1)[:, 1].cpu().tolist()
        col_logits = outputs['batch_column_info_cls_logits'][0]
        col_probs_flat = torch.softmax(col_logits, dim=-1)[:, 1].cpu().tolist()
        # split flat column probs per table
        col_probs = []
        start = 0
        for cnt in col_counts:
            col_probs.append(col_probs_flat[start:start+cnt])
            start += cnt

        return {
            tbl['table_name']: {
                'table_name': tbl['table_name'],
                'table_prob': tbl_probs[i],
                'column_names': tbl['column_names'],
                'column_probs': col_probs[i]
            }
            for i, tbl in enumerate(sample['schema']['schema_items'])
        }

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle long schemas by splitting and merging predictions.
        """
        splits = SchemaSplitter.split(sample, self.tokenizer)
        partials = [self.predict_one(sp) for sp in splits]
        return ResultMerger.merge(sample, partials)

    def evaluate_coverage(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Compute table/column coverage metrics over the dataset.
        """
        max_k = 100
        tbl_cov = [0]*max_k
        col_cov = [0]*max_k
        total_tbl, total_col = 0, 0

        for item in dataset:
            true_tbls = [i for i, l in enumerate(item['table_labels']) if l]
            preds = self.predict(item)
            tbl_scores = [preds[t]['table_prob'] for t in preds]
            for k in range(max_k):
                topk = list(np.argsort(-np.array(tbl_scores))[:k+1])
                if lista_contains_listb(topk, true_tbls):
                    tbl_cov[k] += 1
            total_tbl += 1

            for ti, tbl in enumerate(item.get('column_labels', [])):
                true_cols = [i for i, l in enumerate(tbl) if l]
                if not true_cols:
                    continue
                col_scores = preds[item['schema']['schema_items'][ti]['table_name']]['column_probs']
                for k in range(max_k):
                    topk_c = list(np.argsort(-np.array(col_scores))[:k+1])
                    if lista_contains_listb(topk_c, true_cols):
                        col_cov[k] += 1
                total_col += 1

        print(f"Total tables evaluated: {total_tbl}")
        print("Table coverage:", tbl_cov)
        print(f"Total columns evaluated: {total_col}")
        print("Column coverage:", col_cov)


if __name__ == '__main__':
    # Example usage: python schema_item_filter.py bird_with_evidence
    import sys
    ds_name = sys.argv[1] if len(sys.argv) > 1 else 'bird'
    checkpoint = f"sic_ckpts/sic_{ds_name}"
    classifier = SchemaItemClassifierInference(checkpoint)
    data = json.load(open(f"./data/sft_eval_{ds_name}_text2sql.json", 'r', encoding='utf-8'))
    classifier.evaluate_coverage(data)
