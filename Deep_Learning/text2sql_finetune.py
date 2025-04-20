# Attribution: Original code by Ruoxin Wang
# Repository: https://github.com/RoxanneWAAANG/LangSQL

"""
Module: finetune_spider
Fine-tune a causal language model (e.g., StarCoder) on the Spider Text-to-SQL dataset.
Uses question-SQL pairs and trains the model to generate SQL conditioned on questions.
"""
import argparse
import os
import json
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)

def preprocess_function(example, tokenizer, max_length=1024):
    """
    Prepare input_ids and labels for causal LM fine-tuning.
    Masks question tokens so the model only learns to predict SQL tokens.

    Args:
        example (dict): Contains 'question' and 'query' fields.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        dict with 'input_ids', 'attention_mask', and 'labels'.
    """
    # Tokenize question and SQL separately
    question_tokens = tokenizer(
        example['question'],
        add_special_tokens=False
    )['input_ids']
    sql_tokens = tokenizer(
        example['query'],
        add_special_tokens=False
    )['input_ids']
    # Append EOS token
    eos = tokenizer.eos_token_id
    sql_tokens.append(eos)

    # Build combined sequence
    input_ids = question_tokens + sql_tokens
    # Truncate if necessary
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        # Ensure question is masked entirely if truncated
        mask_start = max_length - len(sql_tokens)
    else:
        mask_start = len(question_tokens)

    # Build labels: mask question part
    labels = [-100] * mask_start + input_ids[mask_start:]
    attention_mask = [1] * len(input_ids)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a causal LM on Spider')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Pretrained model identifier (e.g., bigcode/starcoder)')
    parser.add_argument('--output_dir', type=str, default='./finetuned-spider',
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--overwrite_cache', action='store_true')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load Spider dataset
    raw_datasets = load_dataset('spider')
    # Process splits
    def preprocess_split(split):
        return split.map(
            lambda ex: preprocess_function(ex, tokenizer, args.max_length),
            batched=False,
            remove_columns=split.column_names,
            load_from_cache_file=not args.overwrite_cache
        )

    tokenized_datasets = DatasetDict({
        'train': preprocess_split(raw_datasets['train']),
        'validation': preprocess_split(raw_datasets['validation'])
    })

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available()
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )

    # Fine-tune
    trainer.train()
    # Save final model
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    import torch
    main()
