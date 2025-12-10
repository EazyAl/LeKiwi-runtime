#!/usr/bin/env python3
"""
Test script to record audio using LiveKit's exact settings
and play it back to verify what LiveKit hears.
"""

import sounddevice as sd
import numpy as np
import wave
import time
from pathlib import Path
import argparse
import os

# LiveKit's audio settings (matching main_voice_only.py)
SAMPLE_RATE = 16000
CHANNELS = 2  # Stereo
DEVICE = 0  # wm8960 soundcard

def record_livekit_audio(duration=5, output_file="livekit_test_recording.wav"):
    """
    Record audio using LiveKit's exact settings:
    - 16kHz sample rate
    - Stereo (2 channels)
    - Same device LiveKit uses (device 0 with ALSA plug)
    """
    print(f"Recording for {duration} seconds using LiveKit settings...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS} (stereo)")
    print(f"Device: {DEVICE}")
    print("Speak normally...\n")
    
    # Record audio
    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=DEVICE,
            dtype='int16'
        )
    except Exception as e:
        print(f"Error starting recording: {e}")
        return None, None
    
    # Show real-time levels
    print("Recording...")
    for i in range(duration):
        if i < len(recording):
            chunk = recording[i*SAMPLE_RATE:(i+1)*SAMPLE_RATE]
            if len(chunk) > 0:
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                db_level = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
                bars = int((db_level + 60) / 3)
                bars = max(0, min(20, bars))
                level_bar = "#" * bars + "-" * (20 - bars)
                print(f"\r[{level_bar}] {db_level:.2f} dBFS", end="", flush=True)
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    print("\n\nRecording complete!")
    
    # Save to WAV file
    output_path = Path(output_file)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())
    
    print(f"Saved to: {output_path.absolute()}")
    
    # Calculate statistics
    audio_float = recording.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(audio_float**2))
    max_level = np.max(np.abs(audio_float))
    db_rms = 20 * np.log10(rms + 1e-10)
    db_max = 20 * np.log10(max_level + 1e-10)
    
    print(f"\nAudio Statistics:")
    print(f"  RMS Level: {db_rms:.2f} dBFS")
    print(f"  Peak Level: {db_max:.2f} dBFS")
    print(f"  Average RMS: {rms:.4f} ({rms*100:.1f}%)")
    
    return recording, output_path

def play_recording(audio_file):
    """Play back the recorded audio"""
    print(f"\nPlaying back: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return

    with wave.open(str(audio_file), 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # Reshape if stereo
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2)
    
    sd.play(audio_data, samplerate=sample_rate, device=DEVICE)
    sd.wait()
    print("Playback complete!")

def monitor_live_audio(duration=10):
    """Monitor audio levels in real-time (what LiveKit sees)"""
    print(f"Monitoring audio for {duration} seconds...")
    print("This shows what LiveKit sees in real-time.\n")
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        
        # Calculate RMS level in dBFS
        rms = np.sqrt(np.mean(indata**2))
        if rms > 0:
            db_level = 20 * np.log10(rms)
        else:
            db_level = -np.inf
        
        # Create level meter
        bars = int((db_level + 60) / 3)  # Scale -60dB to 0dB into 20 bars
        bars = max(0, min(20, bars))
        level_bar = "#" * bars + "-" * (20 - bars)
        
        print(f"\r[{level_bar}] {db_level:.2f} dBFS", end="", flush=True)
    
    try:
        with sd.InputStream(
            device=DEVICE,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=1024
        ):
            time.sleep(duration)
        print("\n\nMonitoring complete!")
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError starting input stream: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test LiveKit audio input settings")
    parser.add_argument(
        "--mode",
        choices=["record", "monitor", "both"],
        default="both",
        help="Mode: record (record and play), monitor (real-time levels), or both"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="livekit_test_recording.wav",
        help="Output filename for recording"
    )
    
    args = parser.parse_args()
    
    # Configure sounddevice to use same settings as LiveKit
    # This is critical - matches what main.py does
    os.environ["PA_ALSA_PLUGHW"] = "1"
    try:
        sd.default.device = DEVICE
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS
        print(f"Configured sounddevice: Device {DEVICE}, {SAMPLE_RATE}Hz, {CHANNELS}ch")
    except Exception as e:
        print(f"Warning: Could not set sounddevice defaults: {e}")
    
    if args.mode in ["record", "both"]:
        recording, output_path = record_livekit_audio(args.duration, args.output)
        if output_path:
            play_recording(output_path)
    
    if args.mode in ["monitor", "both"]:
        if args.mode == "both":
            input("\nPress Enter to start monitoring (Ctrl+C to stop)...")
        monitor_live_audio(args.duration)

if __name__ == "__main__":
    main()


