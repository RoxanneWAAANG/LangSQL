from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="seeklhy/codes-1b",
    local_dir="/home/jack/Projects/yixin-llm/yixin-llm-data/Text2SQL/cods-1b"
)
print(f"Model downloaded to: {model_path}")