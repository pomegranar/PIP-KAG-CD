from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chengpingan/PIP-KAG-7B",
    local_dir="./pip-kag-7b",
    resume_download=True
)


