"""
This program recognizes human faces and overlays eye detection.
Note: This is a python 3.6 only program, it does not work on python 2.7
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time

from keras.models import load_model

sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
# glasses_shrunk = np.copy(sunglasses)
glasses_shrunk  = cv2.resize(sunglasses, (150, 50))
glasses_RGB     = glasses_shrunk[:, :, 0:3]
glasses_alpha   = glasses_shrunk[:, :, 3]
glasses_wide    = glasses_shrunk.shape[0]
glasses_high    = glasses_shrunk.shape[1]
EYEBROW_INDICES = [9, 8, 6, 7]

model = load_model('my_final_model.h5')


def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # Keep the video stream open
    while rval:
        # Plot the image from camera with all the face and eye detections marked
        cv2.imshow("face detection activated", frame)
        
        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # Exit by pressing any key
            # Destroy windows 
            cv2.destroyAllWindows()
            
            # Make sure window closes on OSx
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()


def detect_face_features(source_img, detect_eyes=True, scaleFactor=1.25, minNeighbours=3):
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    painted_img = np.copy(source_img)
    eyes = []
    crop_areas = []

    for (x, y, w, h) in faces:
        cv2.rectangle(painted_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        crop_area = gray_img[y:y + h, x:x + w]
        crop_areas.append(crop_area)

        if not detect_eyes: continue
        eyes_in_face = eye_cascade.detectMultiScale(crop_area)

        for (xe, ye, we, he) in eyes_in_face:
            eyes.append((x + xe, y + ye, we, he))

    for (x, y, w, h) in eyes:
        cv2.rectangle(painted_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return faces, eyes, painted_img, crop_areas


def laptop_camera_go(process_func=None, frame_gap=0.05):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read();
        if process_func != None: frame = process_func(frame)
        cv2.imshow('press q to exit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            time.sleep(frame_gap)

    cap.release()
    cv2.destroyAllWindows()


def process_function_face_eye(frame):
    faces, eyes, painted_img, crop_area = detect_face_features(frame, True, 1.25, 5)
    return painted_img


def canny_edges(image_path='images/fawzia.jpg'):
    # la()
    #image1 = cv2.imread('images/james.jpg')

    image2 = cv2.imread(image_path)
    # print('Gray Image Shape {}'.format(image1.shape))
    print('Color Image Shape {}'.format(image2.shape))
    image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Dilate the image to amplify edges
    edges = cv2.dilate(edges, None)

    # Plot the RGB and edge-detected image
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(121)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title('Original Image')
    ax1.imshow(image)

    ax2 = fig.add_subplot(122)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_title('Canny Edges')
    ax2.imshow(edges, cmap='gray')
    plt.show()


def blur_areas(source_img, areas):
    blur_img = np.copy(source_img)
    for (x, y, w, h) in areas:
        sub_img = blur_img[y:y+h, x:x+w]
        sub_img = cv2.blur(sub_img, (50, 50))
        blur_img[y:y+h, x:x+w] = sub_img[0:h,0:w]
    return blur_img

# canny_edges()


def process_func_blur_faces(frame):
    blurred = np.copy(frame)
    faces, eyes, painted_img, crop_areas = detect_face_features(frame, False, 1.25, 3)
    blurred = blur_areas(blurred, faces)
    return blurred


def laptop_camera_identity_hider():
    laptop_camera_go(process_func_blur_faces)


def detect_face_points(source_img, detector_model, scaleFactor=1.25, minNeighbours=3):
    faces, eyes, painted_img, crop_areas = detect_face_features(source_img, False, scaleFactor, minNeighbours)
    all_keypoints = []

    for i in range(len(faces)):
        crop_area = cv2.resize(crop_areas[i], (96, 96))
        crop_area = cv2.normalize(crop_area, -1, 1).reshape(96, 96, 1)
        keypoints = model.predict(np.array([crop_area]))[0]
        keypoints = [k for k in zip(keypoints[0::2], keypoints[1::2])]
        keypoints_out = []
        all_keypoints.append(keypoints_out)

        x, y, w, h = faces[i]
        for (xk, yk) in keypoints:
            xk = int(x + (w * (1 + xk) / 2))
            yk = int(y + (h * (1 + yk) / 2))
            keypoints_out.append((xk, yk))
            cv2.circle(painted_img, (xk, yk), 3, (0, 255, 0), 3)

    return faces, all_keypoints, painted_img, crop_areas


def process_func_add_points(frame):
    faces, keypoints, painted_img, crop_areas = detect_face_points(frame, model)
    return painted_img


def laptop_camera_add_face_points():
    laptop_camera_go(process_func_add_points)


def add_glasses(source_img, detection_model):
    faces, all_keypoints, painted_img, crop_areas = detect_face_points(source_img, detection_model)
    painted_img = np.copy(source_img)

    for keypoints in all_keypoints:
        keypoints = [keypoints[i] for i in EYEBROW_INDICES]
        off_x = sum([p[1] for p in keypoints]) / len(keypoints)
        off_y = sum([p[0] for p in keypoints]) / len(keypoints)
        off_x = int(off_x)
        off_y = int(off_y - (glasses_high / 2))

        for x in range(glasses_wide):
            for y in range(glasses_high):
                if glasses_alpha[x][y] == 0: continue
                painted_img[x + off_x][y + off_y] = glasses_RGB[x][y]

    return painted_img


def process_func_add_glasses(frame):
    return add_glasses(frame, model)


def laptop_camera_add_glasses():
    laptop_camera_go(process_func_add_glasses)


laptop_camera_go(process_func=process_function_face_eye, frame_gap=0.15)
# laptop_camera_identity_hider()
# laptop_camera_add_face_points()
# laptop_camera_add_glasses()
