#!/usr/bin/env python3
"""
Download a subset of episodes from the lekiwi-full-dataset for ACT training.

Dataset: CRPlab/lekiwi-full-dataset (150 episodes total)
Strategy: Take 15 episodes from each of 3 sets of 50 episodes = 45 episodes total

Set 1: Episodes 0-49   -> Take episodes 0-14  (first 15)
Set 2: Episodes 50-99  -> Take episodes 50-64 (first 15)
Set 3: Episodes 100-149 -> Take episodes 100-114 (first 15)
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
import json
import shutil


def download_subset_episodes(
    repo_id: str = "CRPlab/lekiwi-full-dataset",
    output_dir: str = "./lekiwi-subset-45",
    episodes_per_set: int = 15,
):
    """
    Download and create a subset of the dataset with specified episodes.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        output_dir: Local directory to save the subset
        episodes_per_set: Number of episodes to take from each set of 50
    """
    # Define which episodes to keep (15 from each set of 50)
    selected_episodes = []
    
    # Set 1: episodes 0-14 from range 0-49
    selected_episodes.extend(range(0, episodes_per_set))
    
    # Set 2: episodes 50-64 from range 50-99
    selected_episodes.extend(range(50, 50 + episodes_per_set))
    
    # Set 3: episodes 100-114 from range 100-149
    selected_episodes.extend(range(100, 100 + episodes_per_set))
    
    print(f"Selected {len(selected_episodes)} episodes: {selected_episodes}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download the full dataset first
    print(f"\nDownloading dataset from {repo_id}...")
    cache_dir = output_path / ".cache"
    
    try:
        # Use snapshot_download to get all files
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir,
        )
        print(f"Downloaded to: {local_path}")
        
        # Load the dataset and filter episodes
        print("\nLoading and filtering dataset...")
        dataset = load_dataset(repo_id, split="train")
        
        # Filter to keep only selected episodes
        filtered_dataset = dataset.filter(
            lambda x: x["episode_index"] in selected_episodes,
            num_proc=4,
        )
        
        print(f"\nOriginal dataset size: {len(dataset)} frames")
        print(f"Filtered dataset size: {len(filtered_dataset)} frames")
        
        # Remap episode indices to be contiguous (0-44)
        episode_mapping = {old: new for new, old in enumerate(sorted(selected_episodes))}
        
        def remap_episode(example):
            example["episode_index"] = episode_mapping[example["episode_index"]]
            return example
        
        filtered_dataset = filtered_dataset.map(remap_episode)
        
        # Save the filtered dataset
        final_output = output_path / "data"
        filtered_dataset.save_to_disk(str(final_output))
        print(f"\nSaved filtered dataset to: {final_output}")
        
        # Create info.json for the new dataset
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so100_follower",
            "total_episodes": len(selected_episodes),
            "total_frames": len(filtered_dataset),
            "total_tasks": 1,
            "fps": 30,
            "splits": {"train": f"0:{len(selected_episodes)}"},
            "source_dataset": repo_id,
            "selected_episodes": selected_episodes,
        }
        
        meta_dir = output_path / "meta"
        meta_dir.mkdir(exist_ok=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\nDataset subset created successfully!")
        print(f"Location: {output_path.absolute()}")
        print(f"Episodes: {len(selected_episodes)}")
        print(f"Frames: {len(filtered_dataset)}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def download_with_lerobot(
    repo_id: str = "CRPlab/lekiwi-full-dataset",
    output_dir: str = "./lekiwi-subset-45",
    episodes_per_set: int = 15,
):
    """
    Alternative method using LeRobot library directly.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed. Install with: pip install lerobot")
        return None
    
    # Define which episodes to keep
    selected_episodes = []
    selected_episodes.extend(range(0, episodes_per_set))
    selected_episodes.extend(range(50, 50 + episodes_per_set))
    selected_episodes.extend(range(100, 100 + episodes_per_set))
    
    print(f"Selected {len(selected_episodes)} episodes: {selected_episodes}")
    
    # Load the dataset with specific episodes
    print(f"\nLoading dataset {repo_id}...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        episodes=selected_episodes,
    )
    
    print(f"Loaded {dataset.num_episodes} episodes with {len(dataset)} frames")
    
    # The dataset is now ready for ACT training
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download subset of lekiwi dataset for ACT training")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lekiwi-subset-45",
        help="Output directory for the subset dataset",
    )
    parser.add_argument(
        "--episodes-per-set",
        type=int,
        default=15,
        help="Number of episodes to take from each set of 50 (default: 15)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hf", "lerobot"],
        default="lerobot",
        help="Download method: 'hf' for HuggingFace datasets, 'lerobot' for LeRobot library",
    )
    
    args = parser.parse_args()
    
    if args.method == "lerobot":
        download_with_lerobot(
            output_dir=args.output_dir,
            episodes_per_set=args.episodes_per_set,
        )
    else:
        download_subset_episodes(
            output_dir=args.output_dir,
            episodes_per_set=args.episodes_per_set,
        )
