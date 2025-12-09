#!/usr/bin/env python3
"""
Test script for TARS audio using the setup defined in README.md
Plays the tars.wav file with the TARS-like robotic voice effect.
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the script directory and construct path to assets
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets"
TARS_AUDIO = ASSETS_DIR / "tars.wav"


def check_sox_installed():
    """Check if sox is installed on the system."""
    try:
        subprocess.run(["sox", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def play_tars_audio(audio_file: Path):
    """
    Play audio file with TARS-like robotic voice effect.

    Effect parameters from README.md:
    - highpass 300: Remove frequencies below 300 Hz
    - lowpass 3000: Remove frequencies above 3000 Hz
    - compand: Dynamic range compression
    - chorus: Add chorus effect for robotic sound
    - reverb 10: Add reverb
    - gain -n -3: Normalize and reduce gain by 3dB
    """
    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file}")
        return False

    print(f"Playing {audio_file.name} with TARS effect...")
    print("Press Ctrl+C to stop playback\n")

    try:
        # Build the play command with TARS effect
        cmd = [
            "play",
            str(audio_file),
            "highpass",
            "300",
            "lowpass",
            "3000",
            "compand",
            "0.01,0.20",
            "-60,-40,-10",
            "-5",
            "-90",
            "0.1",
            "chorus",
            "0.7",
            "0.9",
            "55",
            "0.4",
            "0.25",
            "2",
            "-t",
            "reverb",
            "10",
            "gain",
            "-n",
            "-3",
        ]

        # Run the play command
        subprocess.run(cmd, check=True)
        print("\nPlayback completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running play command: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user.")
        return False


def main():
    """Main function to test TARS audio."""
    print("=" * 60)
    print("TARS Audio Test Script")
    print("=" * 60)
    print()

    # Check if sox is installed
    if not check_sox_installed():
        print("Error: sox is not installed or not found in PATH")
        print("Please install sox and libsox-fmt-all:")
        print("  sudo apt install sox libsox-fmt-all")
        sys.exit(1)

    print("✓ sox is installed")

    # Check if audio file exists
    if not TARS_AUDIO.exists():
        print(f"Error: Audio file not found at {TARS_AUDIO}")
        print(f"Expected path: {TARS_AUDIO.absolute()}")
        sys.exit(1)

    print(f"✓ Audio file found: {TARS_AUDIO}")
    print()

    # Play the audio with TARS effect
    success = play_tars_audio(TARS_AUDIO)

    if success:
        print("\nTest completed successfully!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
