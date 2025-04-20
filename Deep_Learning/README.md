# Text2SQL -- Deep Learning Approach


## Overview
This model based on CodeS model, which was pre-trained on a large SQL-related corpus. The models have shown excellent performance on benchmarks like Spider and BIRD, making them highly suitable for real-world SQL generation tasks.

In this project, I leverage CodeS-1B, fine-tuned for translating natural language questions into SQL queries. I also integrate the Schema Item Classifier to filter relevant schema items, ensuring that the generated queries are both syntactically and semantically correct.

## Features
- **CodeS Model Integration**: The CodeS model is fine-tuned for Text-to-SQL tasks, providing accurate SQL query generation.
- **Schema Item Filtering**: A schema item classifier is used to filter relevant schema items, improving the SQL generation process.
- **Model Evaluation**: Model evaluated using two benchmarks, including Spider and BIRD, to assess the accuracy of generated SQL queries.


## Usage

### 1. **Downloading the Data and Checkpoints**

You will need to download the required datasets, model checkpoints, and evaluation scripts:

- [Data](https://drive.google.com/file/d/189spLXUL3gF8k4sny5qiWMqW3wOzx5AD/view?usp=sharing)
- [Schema Item Classifier Checkpoints](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing)
- [Spider Evaluation Scripts](https://drive.google.com/file/d/1iNa1WgA9tN_OFna08nq_tHZdXx9Lz2vO/view?usp=sharing)

Unzip these files:

```bash
unzip data.zip
unzip sic_ckpts.zip
unzip test_suite_sql_eval.zip
```

### 2. **Run Inference**

Once everything is set up, you can run the inference using:

```bash
bash run_few_shot_evaluations.sh
```

For SFT results:

```bash
bash run_sft_evaluations.sh
```
