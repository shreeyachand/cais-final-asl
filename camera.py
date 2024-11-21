import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


model = torch.load('5-epochs.pth', weights_only=False)
model.eval()

classes = list('ABCDEFGHIJKLMNOPQRSTUVWYXZ')
classes.append('del')
classes.append('nothing')
classes.append('space')
# model outputs an number 0-28, which corresponds to the index of the label in this classes list.



#image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet standards
])

# make predictions
def predict(frame):
    try:
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        img_tensor = transform(img).unsqueeze(0)  
        
        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1) 
            confidence, predicted = torch.max(probabilities, 1)  

        # Return the predicted class and confidence score
        return classes[predicted.item()], confidence.item()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


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

    # Predict on the frame
    prediction, confidence = predict(frame)

    if prediction is not None:
        cv2.putText(frame, f'Prediction: {prediction}, Confidence: {confidence:.2f}', 
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Live Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
