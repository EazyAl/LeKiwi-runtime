"""
View the pose detection stream from a running LeKiwi robot.

This script connects to the ZMQ stream from the PoseDetectionService
and displays the annotated camera feed with pose/fall detection overlay.

Usage:
    # Run this AFTER starting the robot with streaming enabled:
    uv run python scripts/view_pose_stream.py

    # Or specify a different host/port:
    uv run python scripts/view_pose_stream.py --host 192.168.1.100 --port 5557

Press 'q' to quit.
"""

import argparse
import cv2
import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser(description="View pose detection stream from LeKiwi")
    parser.add_argument("--host", default="localhost", help="Host running the robot (default: localhost)")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ stream port (default: 5557)")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest frame
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
    
    address = f"tcp://{args.host}:{args.port}"
    print(f"Connecting to pose stream at {address}...")
    socket.connect(address)
    print("Connected. Waiting for frames... (press 'q' to quit)")

    try:
        while True:
            try:
                # Receive JPEG frame with timeout
                if socket.poll(timeout=1000):  # 1 second timeout
                    jpeg_bytes = socket.recv()
                    
                    # Decode JPEG to numpy array
                    nparr = np.frombuffer(jpeg_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        cv2.imshow("LeKiwi Pose Detection", frame)
                else:
                    print(".", end="", flush=True)  # Show waiting indicator
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except zmq.ZMQError as e:
                print(f"ZMQ error: {e}")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
