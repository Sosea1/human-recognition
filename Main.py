import flet as ft
import os
import json
from IPython import display

import ultralytics

from ultralytics import YOLO

from IPython.display import display, Image
from flet_ivid import VideoContainer 
import cv2
import matplotlib.pyplot as plt
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from flet import AppBar, ElevatedButton, Page, Text, View, colors

import numpy as np

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
mp_pose = mp.solutions.pose

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    # mp_image = mp.Image.create_from_file('/path/to/image')
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)# Image is no longer writeable
    results = model.detect(mp_image)                 # Make prediction      # Image is now writeable
    return results

from mediapipe.framework.formats import landmark_pb2
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_file = open('pose_landmarker_lite.task', "rb")
model_data = model_file.read()
model_file.close()
    
base_options = python.BaseOptions(model_asset_buffer=model_data)
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    num_poses =10)


def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Append the normalized frame into the frames list
        frames_list.append(resized_frame)
    
    # Release the VideoCapture object. 
    video_reader.release()

    # Return the frames list.
    return frames_list

def extract_keypoints(pose_landmarks):
    pose =[]
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten() if pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 256, 256

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the directory containing the UCF50 dataset. 

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.

actions = np.array(['cartwheel', 'catch', 'clap', 'climb', 'dive', 'draw_sword', 'dribble', 'fencing',
                    'flic_flac', 'golf', 'handstand', 'hit', 'jump',
                    'pick', 'pour', 'pullup', 'push', 'pushup', 'shoot_ball', 'sit', 
                    'situp', 'swing_baseball', 'sword_exercise', 'throw'])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

def model_1():

    
    model_2 = Sequential()
    model_2.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,132)))
    model_2.add(LSTM(128, return_sequences=True, activation='relu'))
    model_2.add(LSTM(64, return_sequences=False, activation='relu'))
    model_2.add(Dense(64, activation='relu'))
    model_2.add(Dense(32, activation='relu'))
    model_2.add(Dense(actions.shape[0], activation='softmax'))

    model_2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model_2

from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from collections import deque 
from collections import Counter
import pandas as pd
def create_LRCN_model():
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()
    
    # Define the Model Architecture.
    ########################################################################################################################
    
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (20, 256, 256, 3)))
    
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(TimeDistributed(Flatten()))
                                      
    model.add(LSTM(32))
                                      
    model.add(Dense(len(actions), activation = 'softmax'))

    ########################################################################################################################

    # Display the models summary.
    model.summary()
    
    # Return the constructed LRCN model.
    return model

# Construct the required LRCN model.


vc = None
	
def main(page: ft.Page):

    file_path_images = []
    
    #диалог выбора файлов
    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            str(len(e.files)) + " files selected" if e.files else "Cancelled!"
        ),
        
        selected_files.update()
        
        
    
        
    
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    
    selected_files = ft.Text()
    

    
    def get_file(e):
        global vc
        print(pick_files_dialog.result.files[0].path)
        if pick_files_dialog.result.files[0].path is not None:
            
            LRCN_model = create_LRCN_model()
            model = model_1()
            model.load_weights('modern_model_32.h5')
            # LRCN_model.load_weights('LRCN_model.h5')
            with PoseLandmarker.create_from_options(options) as landmarker:  
                sequence = []
                sentence = []
                predictions = []
                threshold = 0.5
                actions_ = []

                cap = cv2.VideoCapture(pick_files_dialog.result.files[0].path)
                
                original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (original_video_width,original_video_height))
                while cap.isOpened():
                    sequences = []
                    frames_queue = []
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
            # Make detections
                    results = mediapipe_detection(frame, landmarker)
                    annotated_image = draw_landmarks_on_image(frame, results)
            # 2. Prediction logic
                    window = []
                    if(len(results.pose_landmarks) == 0):
                        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                        frames_queue.append(resized_frame)
                        if len(frames_queue) >= SEQUENCE_LENGTH:
                            frames_queue = frames_queue[-20:]
                            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]
                            predicted_label = np.argmax(predicted_labels_probabilities)
                            predicted_class_name = actions[predicted_label]
                            actions_.append(predicted_class_name)
                    
                    
                    
                    for idx in range(len(results.pose_landmarks)):
                        pose_landmarks = results.pose_landmarks[idx]
                        keypoints = extract_keypoints(pose_landmarks)
                        sequence.append(keypoints)
                        
                         
                        

                        if(len(sequence) > 20):
                            sequence = sequence[-20:]
                    # keypoints = np.expand_dims(keypoints, axis=0)
                    # keypoints = np.expand_dims(keypoints, axis=0)
                            res = model.predict(np.expand_dims(sequence, axis=0))
                            act = np.argmax(res)
                            action = actions[np.argmax(res)]
                    # action = actions[np.argmax(res)]
                            actions_.append(act)
                            h, w, c = frame.shape
                            x_max = 0
                            y_max = 0
                            x_min = w
                            y_min = h
                            for lm in pose_landmarks:
                                x, y = int(lm.x * w), int(lm.y * h)
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                            x_max = x_max + int(h/15)
                            y_max = y_max + int(h/15)
                            x_min = x_min - int(h/15)
                            y_min = y_min - int(h/15)
                            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(annotated_image, ' '.join(action), (x_min,y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                    out.write(annotated_image)
                
                
                cap.release()
                out.release()
               
                sequences = []
                frames = frames_extraction(pick_files_dialog.result.files[0].path)
                if len(frames) == SEQUENCE_LENGTH:
                    for frame in frames:
                        window = []
                        results = mediapipe_detection(frame, landmarker)
                        for idx in range(len(results.pose_landmarks)):
                            pose_landmarks = results.pose_landmarks[idx]
                            keypoints = extract_keypoints(pose_landmarks)

                            window.append(keypoints)
                    # keypoints = np.expand_dims(keypoints, axis=0)
                    # keypoints = np.expand_dims(keypoints, axis=0)
                            
                    sequences.append(window)
                X = np.asarray(sequences)
                res = model.predict(X)
                action = actions[np.argmax(res)]
                # cv2.putText(annotated_image, ' '.join(action), (3,103),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                page.add(
                ft.Row(
                    [
                        ft.Text("Распознанное движение: "),
                            ft.Text(action)
                    ]
                )
                )
                page.update()
                # common_ = Counter(actions_).most_common(1) # [('a', 3)]
                # if(len(common_) > 0):
                #     common, number = common_[0]
                # else:
                #     common = 0
            
            if vc is not None:
                page.controls.pop()
                page.update()
                vc = VideoContainer(pick_files_dialog.result.files[0].path, play_after_loading=True, border_radius=18, expand=True) # This is a VideoContainer
                page.add(ft.Row([vc], alignment="center", width=600, height=600))
                page.update()
            else:
                vc = VideoContainer(pick_files_dialog.result.files[0].path, play_after_loading=True, border_radius=18, expand=True) # This is a VideoContainer
                page.add(ft.Row([vc], alignment="center", width=600, height=600))
                page.update()
            vc.play()
             
    def play_again(e):
        global vc
        if vc is not None:
            vc.pause()
            vc.play()

    
    page.overlay.append(pick_files_dialog)
    
    
    page.add(
        ft.Row(
            [
            ft.ElevatedButton(
                    "Pick files",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False, file_type=ft.FilePickerFileType.VIDEO 
                    ),
                ),
                selected_files,
                ft.ElevatedButton(
                    "Send files to model",
                    on_click=get_file,
                ),
                ft.ElevatedButton(
                    "Play again",
                    on_click=play_again,
                ),
                
            ]
        )
    )
    
    page.title = "Number Recognition"
    page.update()

    
ft.app(target=main, assets_dir="train_dataset_dataset")



def model_1():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard
    
    model_2 = Sequential()
    model_2.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,132)))
    model_2.add(LSTM(128, return_sequences=True, activation='relu'))
    model_2.add(LSTM(64, return_sequences=False, activation='relu'))
    model_2.add(Dense(64, activation='relu'))
    model_2.add(Dense(32, activation='relu'))
    model_2.add(Dense(actions.shape[0], activation='softmax'))

    model_2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model_2

