import torch
import numpy as np
import cv2
import os
import time
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)

# Load YOLOv5 model from the GitHub repository
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='github')
model.conf = 0.4  # confidence threshold
model.classes = [0]  # Person class (COCO dataset has 0 as person)

# Classification based on bounding box height
def classify_person(current_height, all_heights):
    """
    Classify a person as 'Child' if their height is relatively smaller compared to others, and 'Adult' otherwise.
    :param current_height: Height of the person to classify.
    :param all_heights: List of heights of all detected persons in the current frame.
    :return: 'Child' or 'Adult'.
    """
    if not all_heights:
        return "Adult/Child"
    
    avg_height = np.mean(all_heights)
    
    if current_height < avg_height:  # Adjust the ratio based on empirical observation
        return "Child"
    else:
        return "Adult"

# Function to generate unique IDs
def generate_unique_id():
    return int(time.time() * 1000)  # Simple unique ID based on the current time

# Function to compute Intersection over Union (IoU) for tracking
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# Get the directory where the script is located (the src folder)
src_dir = os.path.dirname(os.path.abspath(__file__))

# Move one level up to the project directory
project_dir = os.path.dirname(src_dir)

# Specify input and output folders relative to the project directory
input_folder = os.path.join(project_dir, 'input')
output_folder = os.path.join(project_dir, 'output')

# Ensure the input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder not found at: {input_folder}")

# Ensure the output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each video in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other video formats if needed
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, f'processed_{video_file}')

        # Initialize video capture
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties (width, height, frames per second)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize VideoWriter to save the output video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        start_time = time.time()  # Start time for estimating processing duration

        unique_id_counter = 0
        person_data = {}  # Dictionary to store person data
        person_tracks = defaultdict(list)  # Dictionary to store bounding boxes for tracking
        lost_persons = []  # List to store UIDs of lost persons (for reassignment)

        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break  # Break when video ends

            # YOLOv5 Detection
            results = model(frame)

            # Create a list to store detected heights
            all_heights = []
            detections = []

            # Process results and draw bounding boxes
            for result in results.xyxy[0]:  # result.xyxy[0] contains detections
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0:  # Person class
                    height = y2 - y1
                    all_heights.append(height)  # Add height to list
                    detections.append((x1, y1, x2, y2, conf, height))

            # Track detected persons
            new_person_data = {}
            for x1, y1, x2, y2, conf, height in detections:
                detection_id = None

                # Find if this person is already detected and assign ID
                for person_id, data in person_data.items():
                    prev_x1, prev_y1, prev_x2, prev_y2, _ = data
                    # Check if the new detection overlaps with previous detection
                    if compute_iou((x1, y1, x2, y2), (prev_x1, prev_y1, prev_x2, prev_y2)) > 0.5:
                        detection_id = person_id
                        break

                if detection_id is None:
                    if lost_persons:
                        detection_id = lost_persons.pop()  # Reuse lost UID
                    else:
                        detection_id = unique_id_counter  # Assign new UID
                        unique_id_counter += 1
                    
                    # Classify new detection
                    person_class = classify_person(height, all_heights)
                    # Store new person data
                    new_person_data[detection_id] = (x1, y1, x2, y2, person_class)
                else:
                    # Update existing person data
                    new_person_data[detection_id] = (x1, y1, x2, y2, person_data[detection_id][4])

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Position label on the left side of the bounding box
                label_x = int(x1) - 10  # 10 pixels to the left of the bounding box
                label_y = int(y1) + (y2 - y1) // 2  # Vertical center of the bounding box

                # Ensure the label does not go out of frame bounds
                if label_x < 0:
                    label_x = int(x1) + 10
                    label_y = int(y1) + (y2 - y1) // 2

                # Convert label coordinates to integers
                label_x = int(label_x)
                label_y = int(label_y)

                # Label with class, confidence, and unique ID
                label_text = f'{new_person_data[detection_id][4]} {conf:.2f} ID:{detection_id}'
                cv2.putText(frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Update person_data with new_person_data
            lost_persons += [uid for uid in person_data.keys() if uid not in new_person_data]  # Add lost persons
            person_data = new_person_data

            # Write the processed frame to the output video
            out.write(frame)

            # Calculate and display progress
            elapsed_time = time.time() - start_time
            frames_processed = frame_num + 1
            progress = (frames_processed / total_frames) * 100
            estimated_time_remaining = (elapsed_time / frames_processed) * (total_frames - frames_processed)

            print(f'Processing {video_file}: {frames_processed}/{total_frames} frames ({progress:.2f}%) - Estimated time remaining: {estimated_time_remaining/60:.2f} minutes', end='\r')

        # Release video capture and writer objects
        cap.release()
        out.release()

        print(f'\nProcessing of {video_file} completed. Output saved to {output_video_path}')
