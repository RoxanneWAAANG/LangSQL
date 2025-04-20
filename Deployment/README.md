# Text-to-SQL Demo

## Overview

This deployment creates a text-to-SQL chatbot, referenced to [CodeS](https://arxiv.org/abs/2402.16347), a language model specifically tailored for text-to-SQL translation. 

## Usage

### Step 1: Install Java
Execute the following commands in your terminal:
```bash
apt-get update
apt-get install -y openjdk-11-jdk
```

### Step 2: Create and Activate a Virtual Anaconda Environment
Run these commands to set up your virtual environment:
```bash
conda create -n demo
conda activate demo
```

### Step 3: Install Required Python Modules
Ensure you have all necessary packages by running:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Prerequisites
### Step 1: Download Classifier Weights
Download the the fine-tune checkpoint from [Huggingface](https://huggingface.co/Roxanne-WANG/LangSQL) for the schema item classifier. 

### Step 2: Set Up Databases
By default, this project includes only one database (i.e., `singer`) in the `databases` folder. 

- To access other databases available in our online demo:
  1. Download all databases from the [BIRD](https://bird-bench.github.io) and [Spider](https://yale-lily.github.io/spider) benchmarks.

- To add and use your own databases:
  1. Place your SQLite database file in the `databases` directory.
  2. Update the `./data/tables.json` file with the necessary information about your database, including:
     - `db_id`: The name of your database (e.g., `my_db` for a database file located at `databases/my_db/my_db.sqlite`).
     - `table_names_original` and `column_names_original`: The original names of tables and columns in your database.
     - `table_names` and `column_names`: The semantic names (or comments) for the tables and columns in your database.

### Step 3: Build the Whoose Index

To build the Whoosh index for fast, relevance‑based content lookup on your SQLite databases, run:

```bash
python build_whoosh_index.py
```

Please note that this process might take a considerable amount of time, depending on the size and content of the databases. Upon completing these steps, the demo should be fully configured.

## Launch services on localhost
To initiate the website, execute the following command:
```
streamlit run app.py
```
This action will start the web application, making it accessible at `http://localhost:8501/`. Please note that the user's history questions will be logged and can be accessed in the `data/history/history.sqlite` file.

