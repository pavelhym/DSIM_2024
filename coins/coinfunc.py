import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import ultralytics
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


import numpy as np
from PIL import Image
import av
import cv2


SHOW_PLOTS = True

WEBCAM_PATH = 0
# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

def show_plot():
    if SHOW_PLOTS:
        st.pyplot()

def detect_yolo(image_path,model, streamlit = True):
    image_path = image_path 
    image = cv2.imread(image_path)
    
    # Perform object detection
    results = model(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the predictions
    boxes = results[0].boxes  # Get the detected boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get the coordinates of the box
        label = box.cls.item()  # Get the class label
        confidence = box.conf.item()  # Get the confidence score as a Python float
        # Draw the box and label on the image
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image_rgb, f'{model.names[int(label)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    plt.imshow(image_rgb)
    plt.axis('off')
    if streamlit:
        show_plot()
    else:
        plt.show()


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))




def play_stored_video(conf, model, path):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """

    is_display_tracker, tracker = True, "bytetrack.yaml"

    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)


    try:
        vid_cap = cv2.VideoCapture(
            str(path))
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf,
                                         model,
                                         st_frame,
                                         image,
                                         is_display_tracker,
                                         tracker
                                         )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.sidebar.error("Error loading video: " + str(e))


    