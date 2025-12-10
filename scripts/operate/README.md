Remote control scripts for teleoperation and recording. First start the LeKiwi host with `python -m lekiwi.host`. Then run client scripts (defaults work out of the box):

```bash
# Teleoperate with leader arm + keyboard
python -m scripts.operate.teleoperate

# Record arm movements
python -m scripts.operate.record_arm --name movement_name

# Record wheel movements
python -m scripts.operate.record_wheels --name movement_name

# Replay recordings
python -m scripts.operate.replay --name movement_name --type arm
```

_All scripts use sensible defaults. Override with flags like `--ip`, `--port`, `--leader_id` if needed._
