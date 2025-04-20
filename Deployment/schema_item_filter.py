# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Module: schema_classifier_inference
Provides utilities to prepare model inputs, filter schemas, and perform schema item classification inference.
"""
import json
import re
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

from utils.classifier_model import SchemaItemClassifier


class InputPreparer:
    """
    Prepare tokenized inputs and index mappings for schema classification models.
    """

    @staticmethod
    def prepare_inputs_and_labels(sample: Dict[str, Any], tokenizer: AutoTokenizer):
        """
        Tokenize question and schema items, return tensors and index mappings.

        Args:
            sample: dict with 'text' and 'schema' keys.
            tokenizer: HuggingFace tokenizer.

        Returns:
            encoder_input_ids (torch.Tensor): Token IDs.
            encoder_input_attention_mask (torch.Tensor): Attention mask.
            column_name_token_indices (List[List[int]]): Token indices per column.
            table_name_token_indices (List[List[int]]): Token indices per table.
            column_counts (List[int]): Number of columns per table.
        """
        table_items = sample["schema"]["schema_items"]
        text = sample["text"]

        # Build word-level input sequence
        input_words = [text]
        column_word_positions = []
        table_word_positions = []
        column_counts = []

        for table in table_items:
            table_name = table["table_name"]
            columns = table["column_names"]

            input_words += ["|", table_name, ":"]
            table_word_positions.append(len(input_words) - 2)

            for col in columns:
                input_words.append(col)
                column_word_positions.append(len(input_words) - 1)
            column_counts.append(len(columns))

        # Tokenize with word-to-token alignment
        tokenized = tokenizer(
            input_words,
            return_tensors="pt",
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
            truncation=True
        )

        word_ids = tokenized.word_ids(batch_index=0)
        # Map word positions to token indices
        def _map_positions(word_positions):
            return [
                [i for i, wid in enumerate(word_ids) if wid == pos]
                for pos in word_positions
            ]

        column_token_positions = _map_positions(column_word_positions)
        table_token_positions = _map_positions(table_word_positions)

        # Move tensors to GPU if available
        ids = tokenized["input_ids"].cuda() if torch.cuda.is_available() else tokenized["input_ids"]
        mask = tokenized["attention_mask"].cuda() if torch.cuda.is_available() else tokenized["attention_mask"]

        return ids, mask, column_token_positions, table_token_positions, column_counts

    @staticmethod
    def get_sequence_length(text: str, tables_and_columns: List[List[str]], tokenizer: AutoTokenizer) -> int:
        """
        Estimate token sequence length for text and partial schema.
        Useful for splitting large schemas.
        """
        # Flatten into input words
        input_words = [text]
        for table, column in tables_and_columns:
            input_words += ["|", table, ":", column, ","]
        # remove trailing comma
        input_words = input_words[:-1]
        tk = tokenizer(input_words, is_split_into_words=True)
        return len(tk["input_ids"])

    @staticmethod
    def split_sample(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
        """
        Split a sample into smaller ones to fit the max token length.
        """
        text = sample["text"]
        tables_and_cols = []
        splitted = []

        # Prepare annotated names
        for item in sample["schema"]["schema_items"]:
            tbl = f"{item['table_name']} ({item.get('table_comment','')})"
            for col, cmt in zip(item["column_names"], item.get("column_comments", [])):
                col_full = f"{col} ({cmt})" if cmt else col
                tables_and_cols.append([tbl, col_full])

        buffer = []
        for pair in tables_and_cols:
            if InputPreparer.get_sequence_length(text, buffer + [pair], tokenizer) < 500:
                buffer.append(pair)
            else:
                splitted.append({"text": text, "schema": {"schema_items": []}})
                # build schema from buffer
                unique_tbls = list({t for t, _ in buffer})
                schema_items = []
                for t in unique_tbls:
                    cols = [c for tbl,c in buffer if tbl == t]
                    schema_items.append({"table_name": t, "column_names": cols})
                splitted[-1]["schema"]["schema_items"] = schema_items
                buffer = [pair]
        # Add last
        splitted.append({"text": text, "schema": {"schema_items": []}})
        # similar schema build for buffer
        unique_tbls = list({t for t,_ in buffer})
        schema_items = []
        for t in unique_tbls:
            cols = [c for tbl,c in buffer if tbl == t]
            schema_items.append({"table_name": t, "column_names": cols})
        splitted[-1]["schema"]["schema_items"] = schema_items

        return splitted


class SchemaFilter:
    """
    Filter schema items based on classifier predictions.
    """

    @staticmethod
    def filter_schema(
        data: Dict[str, Any],
        classifier: SchemaItemClassifierInference,
        top_k_tables: int = 5,
        top_k_columns: int = 5
    ) -> Dict[str, Any]:
        """
        Retain top-scoring tables and columns and filter matched contents.
        """
        pred = classifier.predict(data)

        # sort tables by prob
        table_probs = [v["table_prob"] for v in pred]
        best_tables = np.argsort(-np.array(table_probs))[:top_k_tables]

        new_schema = {"schema_items": [], "foreign_keys": []}
        new_contents = {}
        items = data["schema"]["schema_items"]

        for t_idx in best_tables:
            columns = pred[t_idx]["column_probs"]
            best_cols = np.argsort(-np.array(columns))[:top_k_columns]
            tbl = items[t_idx]
            # build filtered item
            new_item = {
                "table_name": tbl["table_name"],
                "table_comment": tbl.get("table_comment",""),
                "column_names": [tbl["column_names"][i] for i in best_cols],
                "column_types": [tbl.get("column_types",[])[i] for i in best_cols],
                "column_comments": [tbl.get("column_comments",[])[i] for i in best_cols],
                "column_contents": [tbl.get("column_contents",[])[i] for i in best_cols],
                "pk_indicators": [tbl.get("pk_indicators",[])[i] for i in best_cols]
            }
            new_schema["schema_items"].append(new_item)
            for col in new_item["column_names"]:
                key = f"{new_item['table_name']}.{col}"
                if key in data["matched_contents"]:
                    new_contents[key] = data["matched_contents"][key]

        # filter foreign keys
        fk = data["schema"].get("foreign_keys", [])
        valid_tbls = [items[i]["table_name"] for i in best_tables]
        for s,t,u,v in fk:
            if s in valid_tbls and t in valid_tbls:
                new_schema["foreign_keys"].append((s,t,u,v))

        data["schema"] = new_schema
        data["matched_contents"] = new_contents
        return data


def lista_contains_listb(lista: List[int], listb: List[int]) -> bool:
    """
    Check if all elements of listb are in lista.
    """
    return all(b in lista for b in listb)


class SchemaItemClassifierInference:
    """
    Load a trained SchemaItemClassifier and infer table/column relevance.
    """
    def __init__(self, model_path: str):
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        self.model = SchemaItemClassifier(model_path, "eval").eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict relevance probabilities for one split sample.
        """
        ids, mask, col_tokens, tbl_tokens, col_counts = \
            InputPreparer.prepare_inputs_and_labels(sample, self.tokenizer)
        ids, mask = ids.to(self.device), mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(ids, mask, [col_tokens], [tbl_tokens], [col_counts])

        tbl_logits = outputs["batch_table_name_cls_logits"][0]
        tbl_probs = torch.softmax(tbl_logits, dim=1)[:,1].cpu().tolist()
        col_logits = outputs["batch_column_info_cls_logits"][0]
        col_probs = torch.softmax(col_logits, dim=1)[:,1].cpu().tolist()

        # split columns per table
        split_cols = []
        idx = 0
        for cnt in col_counts:
            split_cols.append(col_probs[idx:idx+cnt])
            idx += cnt

        result = {}
        for i, item in enumerate(sample["schema"]["schema_items"]):
            result[item["table_name"]] = {
                "table_prob": tbl_probs[i],
                "column_probs": split_cols[i]
            }
        return result

    def predict(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle large schemas by splitting and merging predictions.
        """
        splits = InputPreparer.split_sample(sample, self.tokenizer)
        preds = [self.predict_one(s) for s in splits]
        return self._merge_predictions(sample, preds)

    @staticmethod
    def _merge_predictions(sample: Dict[str, Any], preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge split predictions into full-schema results.
        """
        tables = sample["schema"]["schema_items"]
        merged = []
        for idx, tbl in enumerate(tables):
            name = tbl["table_name"]
            max_prob = max(p.get(name, {}).get("table_prob",0) for p in preds)
            cols = []
            for p in preds:
                cols += p.get(name, {}).get("column_probs", [])
            merged.append({
                "table_name": name,
                "table_prob": max_prob,
                "column_probs": cols
            })
        return merged

    def evaluate_coverage(self, dataset: List[Dict[str, Any]]):
        """
        Compute top-k coverage metrics for tables and columns.
        """
        table_cov = np.zeros(100, dtype=int)
        column_cov = np.zeros(100, dtype=int)
        total_tables, total_columns = 0, 0

        for data in tqdm(dataset, desc="Evaluating coverage"):
            true_tbls = [i for i,l in enumerate(data["table_labels"]) if l]
            preds = self.predict(data)
            probs = [p["table_prob"] for p in preds]

            for k in range(100):
                topk = np.argsort(-np.array(probs))[:k+1]
                if lista_contains_listb(topk.tolist(), true_tbls):
                    table_cov[k] += 1
            total_tables += 1

            for t_idx, true_cols in enumerate(data["column_labels"]):
                if not any(true_cols): continue
                col_probs = preds[t_idx]["column_probs"]
                for k in range(100):
                    topk_cols = np.argsort(-np.array(col_probs))[:k+1]
                    if lista_contains_listb(topk_cols.tolist(), [i for i,l in enumerate(true_cols) if l]):
                        column_cov[k] += 1
                total_columns += 1

        print(f"Table coverage: {total_tables} -> {table_cov.tolist()}")
        print(f"Column coverage: {total_columns} -> {column_cov.tolist()}")


if __name__ == "__main__":
    dataset_name = "spider"
    model_dir = f"sic_ckpts/sic_{dataset_name}"
    inference = SchemaItemClassifierInference(model_dir)
    data = json.load(open(f"./data/sft_eval_{dataset_name}_text2sql.json"))
    inference.evaluate_coverage(data)
