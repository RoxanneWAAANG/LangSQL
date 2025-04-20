# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Zero-shot text-to-SQL generation and evaluation pipeline.

Attribution:
- HuggingFace Transformers examples: https://github.com/huggingface/transformers
"""

import argparse
import json
import os
import time
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char


def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments including model and dataset paths,
                            token limits, and schema parameters.
    """
    parser = argparse.ArgumentParser(
        description="Run zero-shot text-to-SQL generation and evaluation."
    )
    parser.add_argument(
        '--llm_path', type=str, required=True,
        help='Pretrained LLM model path or identifier'
    )
    parser.add_argument(
        '--sic_path', type=str, required=True,
        help='Schema item cache path'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to JSON dataset file'
    )
    parser.add_argument(
        '--table_num', type=int, default=6,
        help='Number of tables to include in prompt context'
    )
    parser.add_argument(
        '--column_num', type=int, default=10,
        help='Number of columns per table to include'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=4096,
        help='Maximum input tokens for model'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=256,
        help='Maximum tokens to generate'
    )
    return parser.parse_args()


class SQLPostProcessor:
    """
    Clean and normalize generated SQL queries for execution.
    """
    @staticmethod
    def clean(sql: str, schema_items: List[Dict]) -> str:
        """
        Remove unwanted characters and properly quote special columns.

        Args:
            sql (str): Raw generated SQL string.
            schema_items (List[Dict]): Schema metadata including column names.

        Returns:
            str: Cleaned SQL ready for execution.
        """
        # Flatten newlines
        sql = sql.replace("\n", " ")
        # Quote columns with special characters
        for table in schema_items:
            for column in table.get("column_names", []):
                if detect_special_char(column) and column in sql:
                    sql = sql.replace(column, f'"{column}"')
        # Remove any backticks
        return sql.replace('`', '')


class Text2SQLGenerator:
    """
    Encapsulates tokenizer/model setup and SQL generation.
    """
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int
    ):
        """
        Initialize tokenizer and model for generation.

        Args:
            model_path (str): Path or HuggingFace ID of the causal LM.
            max_new_tokens (int): Max tokens the model will generate.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # Use model device for tensors
        self.device = next(self.model.parameters()).device

    def generate(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """
        Generate multiple SQL candidates given tokenized inputs.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized input batch.

        Returns:
            List[str]: List of generated SQL strings.
        """
        input_ids = inputs['input_ids']
        start_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                num_return_sequences=4
            )
        # Decode only the newly generated portion
        return self.tokenizer.batch_decode(
            outputs[:, start_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )


class EvaluationPipeline:
    """
    Manages the entire generation and evaluation workflow.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Load dataset, initialize generator, and prepare dataloader.
        """
        self.args = args
        self.generator = Text2SQLGenerator(
            model_path=args.llm_path,
            max_new_tokens=args.max_new_tokens
        )
        # Load raw JSON for iteration
        raw_data = json.load(open(args.dataset_path, 'r', encoding='utf-8'))
        # Dataset handles tokenization and context building
        self.dataset = SFTSQLGenerationDataset(
            json_file=args.dataset_path,
            tokenizer=self.generator.tokenizer,
            max_input_length=args.max_tokens - args.max_new_tokens,
            mode='eval',
            table_num=args.table_num,
            column_num=args.column_num,
            sic_path=args.sic_path
        )
        self.dataloader = DataLoader(self.dataset, batch_size=1)
        self.raw_data = raw_data

    def run(self) -> None:
        """
        Iterate examples: generate SQL, post-process, check executability,
        collect final predictions, and run evaluation scripts.
        """
        start_time = time.time()
        predictions = []

        for raw_example, batch in tqdm(zip(self.raw_data, self.dataloader),
                                       total=len(self.dataset),
                                       desc='Generating SQL'):
            # Move tensors to model device
            batch = {k: v.to(self.generator.device) for k, v in batch.items()}
            candidates = self.generator.generate(batch)
            final_sql = self._best_executable(candidates, raw_example)
            print(final_sql)
            predictions.append(final_sql)

        total_time = time.time() - start_time
        avg_time = total_time / len(predictions)
        print(f"Model: {self.args.llm_path} | Total: {total_time:.2f}s | "
              f"Count: {len(predictions)} | Avg: {avg_time:.2f}s")

        self._save_and_evaluate(predictions)

    def _best_executable(self, sqls: List[str], example: Dict) -> str:
        """
        Select the first executable SQL, after cleaning. Fallback to placeholder.

        Args:
            sqls (List[str]): Candidate SQL strings.
            example (Dict): Raw example including schema and DB path.

        Returns:
            str: Executable or placeholder SQL.
        """
        schema_items = example['schema']['schema_items']
        db_path = example['db_path']
        for sql in sqls:
            cleaned = SQLPostProcessor.clean(sql, schema_items)
            if check_sql_executability(cleaned, db_path) is None:
                return cleaned
        # fallback
        return sqls[0].strip() or "SQL placeholder"

    def _save_and_evaluate(self, preds: List[str]) -> None:
        """
        Save predictions to file and invoke the appropriate evaluation script.

        Args:
            preds (List[str]): List of final SQL predictions.
        """
        basename = os.path.basename(self.args.dataset_path)
        output_file = 'pred_sqls.txt'
        with open(output_file, 'w', encoding='utf-8') as out:
            for sql in preds:
                out.write(sql + "\n")

        # Example: run Spider eval if dataset contains 'spider'
        if 'spider' in basename:
            eval_cmd = (
                f"python -u test_suite_sql_eval/evaluation.py "
                f"--gold data/.../dev_gold.sql --pred {output_file} "
                f"--db data/.../database --etype exec"
            )
            os.system(eval_cmd)
        # Additional dataset-specific evaluations can be added here


def main() -> None:
    """
    Entry point: parse arguments and execute the evaluation pipeline.
    """
    args = parse_args()
    pipeline = EvaluationPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
