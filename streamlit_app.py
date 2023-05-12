import streamlit as st
from fastcore.all import *
import fastai
from fastai.vision.all import *
import cv2

def get_x(row):
    return row['image']

def get_y(row):
        return row['age']

st.set_page_config(page_title="Age Detection",layout="wide")

st.markdown("""# Mongolian Four Food Classifier

Get ready to impress your Mongolian friends with your knowledge of traditional holiday foods! With this app, you can upload an image of Tsuivan, Khuushuur, Buuz, or Niislel salad and find out which delicious dish it is. Who needs a taste tester when you've got this app? 

This app was created as a fun demo for the Deep Learning course at LETU Mongolia American University, but we won't judge if you use it to win food trivia night. 🍴""")

# Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1gJNYV3KB_oeS7scI9lpQIfSuj-Lb9Og0'
        
    learn = load_learner('export.pkl')
else:
    learn = load_learner('export.pkl')


# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Create a VideoCapture object to read from the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection using the Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) or the cropped face image
        cropped_face = frame[y:y + h, x:x + w]
        age = learn.predict(cropped_face)[0][0]

        # Perform age detection on the cropped face image using your custom age detection algorithm
        # age = custom_age_detection_algorithm(cropped_face)

        # Display the predicted age on the frame
        age_text = "Age: {}".format(round(age,0))
        cv2.putText(frame, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes around the detected faces
        cv2.imshow("Real-time Face Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
video_capture.release()
cv2.destroyAllWindows()
        