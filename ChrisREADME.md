
**Calibrate leader arm first** (you cancelled it earlier):

```bash
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem5AB90687441 \
  --teleop.id=my_leader
```

---

**Then, the commands:**

### Teleoperate (test without recording):
```bash
lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432781 \
  --robot.id=my_follower \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem5AB90687441 \
  --teleop.id=my_leader
```

### Record episodes:
```bash
rm -rf ~/.cache/huggingface/lerobot/CRPlab/lekiwi-dataset

lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432781 \
  --robot.id=my_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 180}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem5AB90687441 \
  --teleop.id=my_leader \
  --dataset.repo_id=CRPlab/lekiwi-dataset \
  --dataset.single_task="Pick up the object" \
  --dataset.num_episodes=10 \
  --dataset.fps=30
```

### To Calibrate the arms (follower and leader)
```
conda activate lerobot
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432781 \
  --robot.id=my_follower
```

```
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem5AB90687441 \
  --teleop.id=my_leader
```


### Run the policy
CRPlab/lekiwi_subset45_act_policy_1
```
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432781 \
  --robot.id=my_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 180}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=CRPlab/eval[add in the path for the file from huggingface]\
  --dataset.single_task="Pick up the object" \
  --dataset.num_episodes=5 \
  --dataset.fps=30 \
  --policy.path=CRPlab[add in the path for the file from huggingface]
```

### To run the policy again 

```
rm -rf ~/.cache/huggingface/lerobot/CRPlab/eval_lekiwi_test_act_policy_2
```


### Record episodes:

```
bash
rm -rf ~/.cache/huggingface/lerobot/CRPlab/lekiwi-full-dataset

lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432781 \
  --robot.id=my_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: 180}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem5AB90687441 \
  --teleop.id=my_leader \
  --dataset.repo_id=CRPlab/lekiwi-full-dataset \
  --dataset.single_task="Grab the Epi-pen from the medkit and administer it to the patient" \
  --dataset.num_episodes=150 \
  --dataset.fps=30
```

# To view camera fall detection.
```
rm -rf ~/.cache/huggingface/lerobot/CRPlab/lekiwi-full-dataset
```

# how to access lekiwi runtime 

```

```



# Step 1: Record a motion
python -m scripts.operate.record_arm_direct --name [name of the recording]

# Step 2: Replay the motion later
python -m scripts.operate.replay_arm_direct --name [name of the recording]