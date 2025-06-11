from huggingface_hub import snapshot_download

local_folder = snapshot_download(
    repo_id="chengpingan/PIP-KAG-7B",
    # local_dir="./Models/pip-kag-7b",
    # cache_dir="/datapool/huggingface/hub/models--chengpingan--PIP-KAG-7B",
    resume_download=True,     # pick up if you get interrupted
)

print("All files downloaded to:", local_folder)
