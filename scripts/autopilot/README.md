Autonomous operation with optional data visualization. If the LeKiwi robot is running in autonomous mode (using `python main.py --stream`), then you can observe the robot state in rerun by running the following on a client computer on the same hotspot.

```bash
# On your computer (requires main.py running with --stream)
python -m scripts.autopilot.observe --ip 172.20.10.2 --port 5556
```
