import cv2
import pickle

# Load the video and parking space data
video = cv2.VideoCapture('input/parking.m')

with open('park_positions', 'rb') as file:
    parking_spots = pickle.load(file)

# Font settings for text overlay
font_type = cv2.FONT_HERSHEY_SIMPLEX

# Define parking space dimensions and thresholds
spot_width, spot_height = 40, 19
total_pixels = spot_width * spot_height
vacancy_threshold = 0.22

def count_available_spaces(processed_image):
    global available_count

    available_count = 0

    for spot in parking_spots:
        x, y = spot

        # Extract the region of interest for each parking space
        cropped_img = processed_image[y:y + spot_height, x:x + spot_width]
        white_pixels = cv2.countNonZero(cropped_img)

        # Compute the occupancy ratio
        occupancy_ratio = white_pixels / total_pixels

        # Check if the space is vacant or occupied
        if occupancy_ratio < vacancy_threshold:
            rectangle_color = (0, 255, 0)  # Green for vacant
            available_count += 1
        else:
            rectangle_color = (0, 0, 255)  # Red for occupied

        # Draw rectangle and display occupancy ratio
        cv2.rectangle(overlay_frame, spot, (x + spot_width, y + spot_height), rectangle_color, -1)
        cv2.putText(overlay_frame, f"{occupancy_ratio:.2f}", (x + 5, y + spot_height - 5), 
                    font_type, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

while True:
    # Restart the video once it reaches the end
    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = video.read()
    if not ret:
        break
    
    overlay_frame = frame.copy()

    # Convert frame to grayscale, blur it, and apply threshold
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 1)
    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 25, 16)

    # Count available spaces in the current frame
    count_available_spaces(binary_frame)

    # Blend the original frame and overlay
    blend_ratio = 0.7
    final_frame = cv2.addWeighted(overlay_frame, blend_ratio, frame, 1 - blend_ratio, 0)

    # Display the counter of available parking spots
    counter_box_w, counter_box_h = 220, 60
    cv2.rectangle(final_frame, (0, 0), (counter_box_w, counter_box_h), (255, 0, 255), -1)
    cv2.putText(final_frame, f"{available_count}/{len(parking_spots)}", 
                (int(counter_box_w / 10), int(counter_box_h * 3 / 4)), 
                font_type, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame in fullscreen mode
    cv2.namedWindow('Parking Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Parking Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Parking Detection', final_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'ESC' is pressed
        break

# Release video and close windows
video.release()
cv2.destroyAllWindows()
