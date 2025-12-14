#!/usr/bin/env python3
"""
Upload the 45-episode subset to HuggingFace.
"""

from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo
import json
import shutil
from pathlib import Path


def upload_to_huggingface(
    local_path: str = "./lekiwi-subset-45",
    repo_id: str = "CRPlab/lekiwi-subset-45",
):
    """Upload the subset dataset to HuggingFace."""
    
    local_path = Path(local_path)
    
    # Load the dataset
    print(f"Loading dataset from {local_path}/data...")
    dataset = load_from_disk(str(local_path / "data"))
    print(f"Loaded {len(dataset)} frames")
    
    # Create the repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    # Push the dataset
    print(f"\nUploading to {repo_id}...")
    dataset.push_to_hub(repo_id, split="train")
    
    # Also upload the info.json
    info_path = local_path / "meta" / "info.json"
    if info_path.exists():
        api.upload_file(
            path_or_fileobj=str(info_path),
            path_in_repo="meta/info.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Uploaded meta/info.json")
    
    print(f"\n✅ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")
    
    return repo_id


if __name__ == "__main__":
    upload_to_huggingface()
