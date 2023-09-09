 import cv2 
 # Load the pre-trained model for human detection
 human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml') 
 # Load the video file or use the webcam feed 
 video_capture = cv2.VideoCapture(0) 
 while True: 
 # Read frame by frame from the video
 ret, frame = video_capture.read() 
 # Convert the frame to grayscale for better detection
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 # Detect humans in the frame 
 humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
 # Draw rectangles around the detected humans 
 for (x, y, w, h) in humans: cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
 # Display the resulting frame
 cv2.imshow('Human Detection', frame) 
 # Exit the loop if 'q' is pressed 
 if cv2.waitKey(1) & 0xFF == ord('q'): break 
 # Release the video capture and close all windows 
 #video_capture.release() 
 cv2.destroyAllWindows()

