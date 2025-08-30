from huggingface_hub import create_repo, upload_folder

repo_id = "wala0404/fake-news-mbert"
create_repo(repo_id, exist_ok=True, private=True)

upload_folder(
    folder_path="models/mbert-fake-news-bf16/best",
    repo_id=repo_id,
    commit_message="Upload trained DistilBERT (REAL=1, FAKE=0)"
)

print("Done. Repo:", f"https://huggingface.co/{repo_id}")
