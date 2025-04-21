# Text2SQL
Deep Learning Project at Duke University



Repository Structure

```
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
├── dataset
│   ├── concert_singer
│   │   ├── concert_singer.sqlite
│   │   └── schema.sql
│   └── sample_spider_data.json
└── requirements.txt
```
