#!/usr/bin/env python3
"""
serve_model.py

"""

from sglang.launch_server import main as launch_server

if __name__ == "__main__":
    launch_server(
        model_path="/datapool/huggingface/hub/models--pip-kag-7b",
        served_model_name="pip-kag-7b",
        dtype="bf16",
        port=8004,
    )
