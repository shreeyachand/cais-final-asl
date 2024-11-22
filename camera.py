
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import mediapipe as mp

# Load the model
model = torch.load('5-epochs.pth', weights_only=False)
model.eval()

# Classes
classes = list('ABCDEFGHIJKLMNOPQRSTUVWYXZ')
classes.extend(['del', 'nothing', 'space'])

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Hand detection using MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Prediction function
def predict(hand_roi):
    try:
        img_tensor = transform(hand_roi).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return classes[predicted.item()], confidence.item()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Access webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch frame from webcam.")
        break

    # Detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Predict
            prediction, confidence = predict(hand_roi)
            if prediction is not None:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'Prediction: {prediction}, Confidence: {confidence:.2f}',
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()