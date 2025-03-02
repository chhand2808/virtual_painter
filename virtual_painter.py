import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)

# Color options
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]  # Red, Blue, Yellow
active_color = colors[0]
hover_time = [0, 0, 0]  # Track hover time per color

# Canvas setup
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
last_positions = []
thickness = 5

# Start capturing
cap = cv2.VideoCapture(0)

# Function to update brush size
def update_thickness(value):
    global thickness
    thickness = value

cv2.namedWindow("Virtual Painter")
cv2.createTrackbar("Brush Size", "Virtual Painter", 5, 20, update_thickness)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Apply slight smoothing filter
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)

    # Draw color selection boxes
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i * 80 + 20, 20), (i * 80 + 100, 60), color, -1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape

            # Extract necessary fingers
            index_finger = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            middle_finger = hand_landmarks.landmark[12]
            
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
            
            # Calculate distances
            thumb_index_dist = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])
            index_middle_dist = np.linalg.norm([index_x - middle_x, index_y - middle_y])
            thumb_middle_dist = np.linalg.norm([thumb_x - middle_x, thumb_y - middle_y])

            # Color selection (hover over box for 3 seconds)
            for i in range(len(colors)):
                if 20 + i * 80 < index_x < 100 + i * 80 and 20 < index_y < 60:
                    hover_time[i] += 1
                    if hover_time[i] > 90:
                        active_color = colors[i]
                        hover_time = [0, 0, 0]
                else:
                    hover_time[i] = max(hover_time[i] - 1, 0)
            
            # Display active color
            cv2.rectangle(frame, (500, 20), (620, 60), active_color, -1)
            
            # Check if thumb and index finger are **very close** -> Start Freehand Drawing
            if thumb_index_dist < 30:
                last_positions.append((index_x, index_y, active_color, thickness))
            
            # Check if all three fingers (thumb, index, middle) are **very close** -> Eraser Mode
            if thumb_index_dist < 30 and index_middle_dist < 30 and thumb_middle_dist < 30:
                active_color = (0, 0, 0)  # Eraser Mode
    
    # Draw on canvas
    for x, y, color, thick in last_positions:
        cv2.circle(canvas, (x, y), thick, color, -1)
    
    # Merge canvas with frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    # Display
    cv2.imshow('Virtual Painter', frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'drawing_{timestamp}.png'
        cv2.imwrite(filename, canvas)

cap.release()
cv2.destroyAllWindows()
