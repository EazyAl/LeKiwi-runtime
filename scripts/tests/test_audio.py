#!/usr/bin/env python3
"""Test LeKiwi audio input and output"""

import sounddevice as sd
import numpy as np

# LeKiwi audio settings
SAMPLE_RATE = 16000
CHANNELS = 2
DEVICE = 0  # wm8960 soundcard

# Test parameters
DURATION = 3  # seconds

# --- Test Speaker ---
print("Playing test tone...")
frequency = 440  # Hz (A4 note)
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
tone = 0.3 * np.sin(2 * np.pi * frequency * t)
# Duplicate for stereo
tone_stereo = np.column_stack([tone, tone])
sd.play(tone_stereo, samplerate=SAMPLE_RATE, device=DEVICE)
sd.wait()

# --- Test Microphone ---
print("Recording from microphone...")
recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    device=DEVICE,
    dtype="int16",
)
sd.wait()
print("Recording complete.")

# --- Playback Recorded Audio ---
print("Playing back recorded audio...")
sd.play(recording, samplerate=SAMPLE_RATE, device=DEVICE)
sd.wait()
print("Done.")
