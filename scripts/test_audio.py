#!/usr/bin/env python3
"""
Test script for TARS audio using the setup defined in README.md
Plays the tars.wav file with the TARS-like robotic voice effect.
"""

import subprocess
import sys
import os
import tempfile
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


def validate_audio_file(audio_file: Path):
    """Validate audio file using soxi. Returns True if valid, False otherwise."""
    try:
        result = subprocess.run(
            ["soxi", str(audio_file)], capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_audio_file(input_file: Path, output_file: Path):
    """Convert audio file to a valid WAV format using sox.
    Tries auto-detection first, then explicit format specification if needed.
    """
    # First try auto-detection (sox will try to detect format automatically)
    try:
        result = subprocess.run(
            ["sox", str(input_file), "-t", "wav", str(output_file)],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        # If auto-detection fails, try common audio formats
        formats_to_try = ["wav", "mp3", "ogg", "flac", "aiff", "au", "raw"]
        for fmt in formats_to_try:
            try:
                result = subprocess.run(
                    ["sox", "-t", fmt, str(input_file), "-t", "wav", str(output_file)],
                    capture_output=True,
                    check=True,
                )
                print(f"  Detected format: {fmt}")
                return True
            except subprocess.CalledProcessError:
                continue

        # If all formats fail, return False
        print(
            f"Error: Could not convert audio file. File may be corrupted or in unsupported format."
        )
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

    # Validate the audio file first
    if not validate_audio_file(audio_file):
        print(f"Warning: Audio file appears to be invalid or corrupted.")
        print(f"Attempting to convert {audio_file.name} to valid WAV format...")

        # Create a temporary converted file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            converted_file = Path(tmp_file.name)

        if not convert_audio_file(audio_file, converted_file):
            print(f"Error: Failed to convert audio file. The file may be corrupted.")
            return False

        print(f"✓ Successfully converted audio file")
        # Use the converted file for playback
        audio_file = converted_file
        # Clean up temp file after playback
        cleanup_file = converted_file
    else:
        cleanup_file = None

    print(f"Playing {audio_file.name} with TARS effect...")
    print("Press Ctrl+C to stop playback\n")

    try:
        # Build the play command with TARS effect
        # Note: -t before reverb is the tail time parameter, not a format specifier
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

        # Clean up temporary converted file if created
        if cleanup_file and cleanup_file.exists():
            cleanup_file.unlink()

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running play command: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")

        # Clean up temporary converted file if created
        if cleanup_file and cleanup_file.exists():
            cleanup_file.unlink()

        return False
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user.")

        # Clean up temporary converted file if created
        if cleanup_file and cleanup_file.exists():
            cleanup_file.unlink()

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
