import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from fastai.vision.all import *
import gdown
# Check if the Haar Cascade XML file exists, otherwise download it
xml_file_path = "haarcascade_frontalface_default.xml"
if not os.path.isfile(xml_file_path):
    with st.spinner("Downloading Haar Cascade XML file..."):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        gdown.download(url, xml_file_path, quiet=False)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(xml_file_path)

def get_x(row):
    return row['image']

def get_y(row):
        return row['age']

model_path = Path("export.pkl")
if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1cs7nWO-XIQBWqLnTjzuG_lOkEnXwyrpd'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn = load_learner('export.pkl')
else:
    learn = load_learner('export.pkl')
# Load the age detection model

class AgeDetector(VideoTransformerBase):
    def __init__(self):
        super().__init__()
    
    def transform(self, frame):
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

            # Perform age detection on the cropped face image
            age, _ = learn.predict(cropped_face)

            # Display the predicted age on the frame
            age_text = "Age: {}".format(round(age))
            cv2.putText(frame, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

def main():
    st.set_page_config(page_title="Age Detection", layout="wide")

    st.markdown("""# Real-Time Age Prediction with Webcam Access

    Using this app, you can predict the age of a person in real-time by accessing your webcam. Impress your friends with this cutting-edge deep learning technology!

    This app was created as a project for the Deep Learning course at LETU Mongolia American University. Have fun exploring the world of age detection with live video!""")

    # Configure the Streamlit WebRTC component
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=AgeDetector)

if __name__ == "__main__":
    main()
