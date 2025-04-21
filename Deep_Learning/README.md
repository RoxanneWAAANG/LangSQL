# Text2SQL -- Deep Learning Approach


## Overview
This model based on [CodeS model](https://arxiv.org/abs/2402.16347), which was pre-trained on a large SQL-related corpus. The models have shown good performance on benchmarks like [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/), making them highly suitable for real-world SQL generation tasks.

In this project, I leverage [CodeS-1B](https://huggingface.co/seeklhy/codes-1b), fine-tuned for translating natural language questions into SQL queries. I also integrate the Schema Item Classifier to filter relevant schema items, ensuring that the generated queries are both syntactically and semantically correct.

## Features
- **CodeS Model Integration**: The CodeS model is fine-tuned for Text-to-SQL tasks, providing accurate SQL query generation.
- **Schema Item Filtering**: A schema item classifier is used to filter relevant schema items, improving the SQL generation process.
- **Model Evaluation**: Model evaluated using two benchmarks, including Spider and BIRD, to assess the accuracy of generated SQL queries.

## Usage

### 1. **Downloading the Data and Checkpoints**

Download the required [Datasets](https://drive.google.com/file/d/189spLXUL3gF8k4sny5qiWMqW3wOzx5AD/view?usp=sharing) and [Schema Item Classifier Checkpoints](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing):

Unzip these files:

```bash
unzip data.zip
unzip sic_ckpts.zip
```

### 2. **Run Inference**

Once everything is set up, you can run the inference using:

```bash
python text2sql_few_shot.py
python text2sql_zero_shot.py
```

### 3. **Fine-Tune on Own Dataset**:

Prepare fine-tuning dataset:

```bash
python prepare_sft_datasets.py
```

Then run the fine-tune script:

```bash
text2sql_finetune.py
```

