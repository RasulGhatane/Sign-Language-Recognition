import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# Model setup
image_x, image_y = 64, 64
classifier = load_model('Improved_Trained_model.h5')

# Print model summary
print("Model Summary:")
classifier.summary()

# Get the number of classes from the model's last layer
num_classes = classifier.layers[-1].units
print(f"Number of classes in the model: {num_classes}")

# Create class labels based on the number of classes
class_labels = [chr(65 + i) for i in range(min(26, num_classes))]
if num_classes > 26:
    class_labels.extend(['nothing', 'space'])
class_labels = class_labels[:num_classes]  
print("Class labels:", class_labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

@tf.function
def predictor(hand_image):
    # Resize the image to 64x64
    test_image = tf.image.resize(hand_image, (image_x, image_y))
    
    # Ensure the image is RGB
    if tf.shape(test_image)[-1] == 1:
        test_image = tf.image.grayscale_to_rgb(test_image)
    
    # Normalize the image
    test_image = test_image / 255.0
    
    # Add batch dimension
    test_image = tf.expand_dims(test_image, axis=0)
    
    # Ensure the shape is (1, 64, 64, 3)
    tf.debugging.assert_equal(tf.shape(test_image), (1, 64, 64, 3), message="Unexpected shape")
    
    result = classifier(test_image, training=False)
    predicted_class = tf.argmax(result[0])
    confidence = result[0][predicted_class]
    
    return class_labels[predicted_class], confidence

# Initialize camera
cam = cv2.VideoCapture(0)

cv2.namedWindow("Gesture Recognition")

current_word = ''
predicted_text = ''
last_prediction = ''
prediction_count = 0
prediction_threshold = 5  # Number of consistent predictions to accept a letter

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Convert the image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box of hand
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            
            # Add padding to bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Extract hand image
            hand_image = frame[y_min:y_max, x_min:x_max]
            
            # Predict gesture
            img_text, confidence = predictor(hand_image)
            
            # Update prediction count
            if img_text == last_prediction:
                prediction_count += 1
            else:
                prediction_count = 0
            last_prediction = img_text
            
            # Update current word and predicted text
            if prediction_count >= prediction_threshold:
                if img_text == 'space':
                    predicted_text += current_word + ' '
                    current_word = ''
                elif img_text != 'nothing':
                    current_word += img_text
                prediction_count = 0
            
            # Display results
            cv2.putText(frame, f"Predicted: {img_text}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Current Word: {current_word}", (30, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Text: {predicted_text}", (30, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)
    
    # Exit on 'ESC' key, clear text on 'c' key
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'): 
        current_word = ''
        predicted_text = ''

cam.release()
cv2.destroyAllWindows()