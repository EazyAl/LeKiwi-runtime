import sys
import os
import cv2
import numpy as np
import rerun as rr

# Add project root to path so we can import scripts and lekiwi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.test_pose_viewer import process_pose_stream

def create_status_image(is_fall, width=400, height=120):
    """
    Creates an RGB image with big bold text for status.
    """
    # Black background, RGB
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if is_fall:
        text = "FALL DETECTED"
        # Red
        color = (255, 0, 0)
    else:
        text = "STATUS: OK"
        # Green
        color = (0, 255, 0)
    
    # Calculate text size to center it
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    x = max(0, (width - text_width) // 2)
    y = (height + text_height) // 2
    
    cv2.putText(img, text, (x, y), font, scale, color, thickness)
    return img

def main():
    rr.init("lekiwi_pose_viewer", spawn=True)
    
    print("Starting pose stream...")
    
    # Try multiple camera indices: [0, 4] (assuming user might have a second cam at 4, or 2)
    # The user previously used 4, so let's try [4, 0] or [0, 4]
    # Let's assume 4 is the primary (as per last edit) and we look for another one, maybe 0 or 2.
    # We'll pass a list of potential indices.
    camera_indices = [4, 2] 
    
    # Pass draw_text=False so stats aren't burned into the video
    try:
        # process_pose_stream now accepts a list and returns a list of frames
        for frames, context in process_pose_stream(camera_indices, draw_text=False):
            
            # Log Primary Camera (Index 4)
            if len(frames) > 0 and frames[0] is not None:
                frame_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                rr.log("video/stream_primary", rr.Image(frame_rgb))
            
            # Log Secondary Camera (Index 0)
            if len(frames) > 1 and frames[1] is not None:
                frame_rgb_sec = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)
                rr.log("video/stream_secondary", rr.Image(frame_rgb_sec))
            
            # Extract stats from context
            event = context.get("event")
            is_fall = context.get("is_fall")
            fps = context.get("fps", 0)
            quality = context.get("quality", {})
            face_stats = context.get("face_stats", {})
            
            # 1. Log Status Image (Big, Bold, Colored)
            status_img = create_status_image(is_fall)
            rr.log("video/status", rr.Image(status_img))
            
            ratio_txt = f"{event.ratio:.2f}" if event else "--"
            score_txt = f"{event.score:.2f}" if event else "--"
            
            # 2. Log Numerical Stats (Markdown)
            md_text = f"""
# Statistics
- **FPS**: {fps:.1f}

## Torso
- Ratio: {ratio_txt}
- Score: {score_txt}

## Quality
- Brightness: {quality.get('brightness', 0):.0f}
- Blur: {quality.get('blur', 0):.0f}
- Motion: {quality.get('motion', 0):.1f}
- Visibility Min: {quality.get('visibility_min', 0):.2f}
- Visibility Mean: {quality.get('visibility_mean', 0):.2f}

## Face
- Awake Likelihood: {face_stats.get('awake_likelihood', -1):.2f}
- Blinks/min: {face_stats.get('blinks_per_min', 0):.1f}
- Perclos: {face_stats.get('perclos', 0):.2f}
- Eyes Open Prob: {face_stats.get('eyes_open_prob', 0):.2f}
- HR: {face_stats.get('hr_bpm', -1):.0f} bpm (q: {face_stats.get('hr_quality', 0):.2f})
- Face Status: {face_stats.get('face_status', 'no_face')}
"""
            rr.log("video/stats", rr.TextDocument(md_text, media_type=rr.MediaType.MARKDOWN))

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
