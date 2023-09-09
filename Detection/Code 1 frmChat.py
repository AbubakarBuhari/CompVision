import cv2 
# Load the pre-trained Haar cascade file for human detection 
cascade_path = 'haarcascade_fullbody.xml'
cascade_classifier = cv2.CascadeClassifier(cascade_path) 
# Initialize webcam 
webcam = cv2.VideoCapture(0) 
# 0 represents the default webcam index
while True: 
# Read frame from webcam
 ret, frame = webcam.read() 
# Convert frame to grayscale for better detection
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
# Perform human detection
 humans = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
# Draw bounding boxes around detected humans 
for (x, y, w, h) in humans: 
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
# Display the resulting frame 
cv2.imshow('Human Detection', frame) 
# Stop the loop if 'q' is pressed
k = cv2.waitKey(1) & 0xFF 
if k == ord('q'): 
    break 
  #Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
