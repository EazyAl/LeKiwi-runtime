# LeKiwi-runtime

Controlling LeKiwi (the robot) can be done either on the Pi host (`LeKiwi`) or from the computer client via `LeKiwiClient`.

## How to run the robot

... on a preconfigured pi (i.e. lerobot installed, conda env setup, etc...)

> Setup on Pi host

```bash
ssh lekiwi@lekiwi.local
```

One you've `ssh`'d in run the following:

```bash
cd Documents/lerobot
conda activate lerobot
```

> Setup on local computer client

```bash
cd lerobot
conda activate lerobot
```

### Run teleoperation

> First call this from the Pi host

```bash
# Without cameras
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=biden_kiwi --robot.cameras="{}"

# With cameras
python -m lerobot.robots.lekiwi.lekiwi_host \
    --robot.id=biden_kiwi \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, "wrist": {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
```

> And this from the computer client

```bash
python examples/lekiwi/teleoperate.py
```

My port
"/dev/tty.usbmodem5AB90687441"

## Sound Setup

For recording:

```bash
arecord -D hw:2,0 -f S16_LE -r 16000 -c 2 audio.wav
```

Always do it at 16kHz to avoid high frequency digital noise (it's ideal for recording voices). You press ctrl + C to stop recording

For playing audio with the robotic voice effect

```bash
sudo apt install sox libsox-fmt-all
```

```bash
# Play audio file with TARS-like effect
play audio.wav \
  highpass 600 \
  lowpass 2500 \
  overdrive 10 \
  compand 0.1,0.3 -60,-40,-10,-5 -5 -90 0.1 \
  echo 0.8 0.8 2 0.4 \
  gain -n -3
```

For speaker test

```
speaker-test -Dhw:2,0 -c2 -twav
```
