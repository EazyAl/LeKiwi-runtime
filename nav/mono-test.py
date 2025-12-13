import cv2
import torch
import numpy as np
import time

class MonoPilot:
    def __init__(self):
        print("Loading MiDaS AI model...")
        # Load small model (fastest)
        model_type = "MiDaS_small" 
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Move to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Setup Transform (resize/normalize image for AI)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        print(f"Model loaded on {self.device}")

    def get_depth_map(self, frame):
        """
        Input: BGR Frame (from OpenCV)
        Output: Depth Map (numpy array), Higher Value = CLOSER
        """
        input_batch = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def process_frame(self, frame):
        """
        Returns a control command (vx, vy, omega) based on depth.
        """
        depth_map = self.get_depth_map(frame)
        h, w = depth_map.shape
        
        # --- Heuristic Logic ---
        # 1. Look at the center strip (Horizon)
        # Avoid looking at floor (bottom) or ceiling (top)
        strip = depth_map[int(h*0.4) : int(h*0.7), :]
        
        # 2. Divide into Left / Center / Right
        third = w // 3
        left_zone = strip[:, :third]
        center_zone = strip[:, third:2*third]
        right_zone = strip[:, 2*third:]
        
        # 3. Calculate "Closeness" (Mean value)
        # Remember: In MiDaS, Higher Value = CLOSER
        l_score = np.mean(left_zone)
        c_score = np.mean(center_zone)
        r_score = np.mean(right_zone)
        
        print(f"L: {l_score:.1f} | C: {c_score:.1f} | R: {r_score:.1f}")
        
        # 4. Calibration (You must tune this!)
        # Check what value you get when 1 meter away from a wall.
        # Let's assume ~500 is "Too Close" for MiDaS small
        THRESHOLD = 500.0 
        
        vx, vy, omega = 0.0, 0.0, 0.0
        
        # --- CONTROL LOGIC ---
        
        if c_score > THRESHOLD:
            # CENTER BLOCKED!
            print("OBSTACLE AHEAD!")
            
            if l_score < r_score:
                # Left is clearer -> Turn Left
                omega = 0.5
            else:
                # Right is clearer -> Turn Right
                omega = -0.5
                
            # Stop moving forward, maybe backup?
            vy = -0.1
            
        else:
            # PATH CLEAR
            # Drive forward
            vy = 0.3
            
            # Minor correction to steer towards most open area (lowest score)
            # Push away from the closer side
            push = (l_score - r_score) * 0.001 
            vx = push # Strafe slightly away from walls
            
        return vx, vy, omega, depth_map

# --- Main Loop (Simulated) ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(2) # Use 0 for webcam
    pilot = MonoPilot()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        vx, vy, omega, dmap = pilot.process_frame(frame)
        
        # Visualize
        # Normalize depth map for display (0-255)
        dmap_norm = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dmap_color = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_MAGMA)
        
        cv2.imshow("Depth Perception", dmap_color)
        cv2.imshow("Input", frame)
        
        if cv2.waitKey(1) == 27: break
        
    cap.release()
    cv2.destroyAllWindows()