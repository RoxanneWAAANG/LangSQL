# Text2SQL

Demo Website: https://huggingface.co/spaces/Roxanne-WANG/LangSQL
Demo Video: 

## Project Description
The Text-to-SQL Module Project addresses the challenge of bridging the gap between non‑technical users and relational databases by translating natural language questions into executable SQL queries. By providing an intuitive interface, this system empowers stakeholders—such as business analysts, researchers, and educators—to interact with complex data without requiring deep SQL proficiency or programming knowledge.

At a high level, the system comprises three core stages:

- **Query Interpretation**: Incoming user questions are preprocessed and analyzed to extract intents, keywords, and schema-related entities. This stage includes tokenization, part-of-speech tagging, and mapping of semantic elements to table and column names.
- **Model-Based SQL Generation**: Three distinct approaches—rule‑based naive templates, classical machine learning classifiers, and neural network models—each generate candidate SQL queries. The ensemble of approaches provides a comparative baseline and a robust fallback mechanism.
- **Execution and Feedback**: Generated SQL queries are validated against the target database, executed in a controlled environment, and returned to the user along with execution metadata (e.g., timing, row count). An interactive frontend visualizes results and logs queries for auditability.

## Usage
### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/RoxanneWAAANG/LangSQL.git
   cd LangSQL
   ```
2. **Create and activate a virtual environment:**
   ```bash
   conda create -n codes
   conda activate codes
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install Java (for deployment):**
  ```bash
  apt-get update
  apt-get install -y openjdk-11-jdk
  ```
*Note: I also implemented a non-Java required version, using `whoosh` package for indexing, refer to `Deployment/build_whoosh_index.py`, drawback is the performance is not good.*

## Methods Overview

Below is a brief introduction to each approach along with their primary requirements. Click the links to view full details and setup instructions in their respective README files:

- **Naive Approach**: A rule‑based baseline that uses pattern matching to generate simple SQL queries without any training data.  
  [Naive Approach Details & Requirements](https://github.com/RoxanneWAAANG/LangSQL/blob/main/Naive_Approach/README.md)

- **Classical ML Approach**: Employs scikit‑learn models (i.e., RandomForestClassifier) to map engineered text features to SQL operations.  
  [Machine Learning Approach Details & Requirements](https://github.com/RoxanneWAAANG/LangSQL/blob/main/Machine_Learning/README.md)

- **Deep Learning Approach**: Fine‑tunes and adapts a pretrained CodeS model for Text-to-SQL tasks using few‑shot and zero‑shot inference.  
  [Deep Learning Approach Details & Requirements](https://github.com/RoxanneWAAANG/LangSQL/tree/main/Deep_Learning)

- **Deployment:** Provides instructions to deploy the Text-to-SQL application as a web service using Streamlit, including database indexing and model serving.
  [Deployment Details & Requirements](https://github.com/RoxanneWAAANG/LangSQL/blob/main/Deployment/README.md)


## Repository Structure

```sh
.
├── Deep_Learning
│   ├── README.md
│   ├── bird_evaluation
│   │   ├── evaluation.py
│   │   ├── evaluation_ves.py
│   │   └── run_evaluation.sh
│   ├── build_contents_index.py
│   ├── modeling_gpt_bigcode.py
│   ├── prepare_sft_datasets.py
│   ├── results
│   │   ├── pred_sqls-codes-1b-spider.txt
│   │   ├── predict_dev-codes-1b-bird-with-evidence.json
│   │   └── predict_dev-codes-1b-bird.json
│   ├── schema_item_filter.py
│   ├── text2sql_few_shot.py
│   ├── text2sql_finetune.py
│   ├── text2sql_zero_shot.py
│   └── utils
│       ├── bridge_content_encoder.py
│       ├── classifier_loss.py
│       ├── classifier_model.py
│       ├── db_utils.py
│       ├── download_nltk.py
│       ├── download_weights.py
│       ├── load_classifier_dataset.py
│       ├── load_pt_dataset.py
│       ├── load_sft_dataset.py
│       └── lr_scheduler.py
├── Deployment
│   ├── README.md
│   ├── app.py
│   ├── build_whoosh_index.py
│   ├── data
│   │   ├── history
│   │   │   └── history.sqlite
│   │   └── tables.json
│   ├── databases
│   │   └── singer
│   │       ├── schema.sql
│   │       └── singer.sqlite
│   ├── db_contents_index
│   │   └── singer
│   │       ├── MAIN_WRITELOCK
│   │       ├── MAIN_qq60yoh2am2v4iv7.seg
│   │       └── _MAIN_1.toc
│   ├── schema_item_filter.py
│   ├── text2sql.py
│   └── utils
│       ├── bridge_content_encoder.py
│       ├── classifier_model.py
│       └── db_utils.py
├── LICENSE
├── Machine_Learning
│   ├── README.md
│   ├── evaluation_results.json
│   └── ml_text2sql.py
├── Naive_Approach
│   ├── README.md
│   └── naive_text2sql.py
├── README.md
├── dataset  # sample data.
│   ├── concert_singer
│   │   ├── concert_singer.sqlite
│   │   └── schema.sql
│   └── sample_spider_data.json
└── requirements.txt
```

## Ethical Statement

This project has been designed with the following ethical considerations in mind:

- **Data Privacy**: Ensure that no sensitive or personal data is exposed. All data used in training and evaluation is sourced from publicly available or properly anonymized datasets.
- **Bias and Fairness**: Assess and mitigate potential biases in model predictions across different groups. Performance metrics will be reported separately for key subpopulations to ensure equitable behavior.
- **Transparency**: Document model architectures, training procedures, and data characteristics. Provide users with clear explanations of model limitations and decision logic.
- **Responsible Deployment**: Include disclaimers about possible model inaccuracies and encourage human oversight when interpreting results. Limit automated query executions to prevent misuse.

## License

This repository is released under the Apache License 2.0, following the same licensing as the CodeS.

--- 
_Some Sections of the README are re-articulated using ChatGPT._
