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
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=biden_kiwi --robot.cameras="{}"
```

> And this from the computer client

```bash
python examples/lekiwi/teleoperate.py
```
