#!/usr/bin/env python3
"""
Create and upload a proper LeRobot v3.0 dataset subset to HuggingFace.

This script:
1. Loads the original dataset using LeRobot
2. Filters to the selected 45 episodes
3. Saves and uploads in proper LeRobot format
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Define which episodes to keep (15 from each set of 50)
SELECTED_EPISODES = list(range(0, 15)) + list(range(50, 65)) + list(range(100, 115))

def main():
    print(f"Selected {len(SELECTED_EPISODES)} episodes:")
    print(f"  Set 1 (0-14): {list(range(0, 15))}")
    print(f"  Set 2 (50-64): {list(range(50, 65))}")
    print(f"  Set 3 (100-114): {list(range(100, 115))}")
    
    # Load the original dataset with only the selected episodes
    print("\nLoading dataset with selected episodes...")
    dataset = LeRobotDataset(
        repo_id="CRPlab/lekiwi-full-dataset",
        episodes=SELECTED_EPISODES,
    )
    
    print(f"Loaded {dataset.num_episodes} episodes with {len(dataset)} frames")
    
    # Push to hub with new name
    print("\nUploading to CRPlab/lekiwi-subset-45...")
    dataset.push_to_hub(
        repo_id="CRPlab/lekiwi-subset-45",
        tags=["lerobot", "so100", "act"],
    )
    
    print("\n✅ Dataset uploaded in proper LeRobot format!")
    print("View at: https://huggingface.co/datasets/CRPlab/lekiwi-subset-45")


if __name__ == "__main__":
    main()
