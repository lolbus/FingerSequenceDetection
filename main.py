import copy

import cv2
import os
import time


import pandas as pd

from metadata import DatasetMeta
from model import modelloader
from threading import Thread
from MMPoseInferencer import MMPoseInferencerObj
from MMDetInferencer import MMDetInferencerObj
from inferencer_function import cam_1_inference, cam_2_inference
import DataCollectionHandler as DCH
import numpy as np

import KeyListener

key_listener = KeyListener.KeyListener()
metadata = DatasetMeta()

# Define the path to the video stream (replace with your actual RTSP URL)
# rtsp_url = 'rtsp://root:pass@192.168.70.2/axis-media/media.amp?videocodec=h264&resolution=640x360'
rtsp_url = 'rtsp://admin:P@ssw0rd@192.168.1.100:554/profile1/media.smp'
rtsp_url2 = 'rtsp://admin:P@ssw0rd@192.168.1.101:554/profile1/media.smp'


# Define the directory where to save the images
class camStatusHandler_object():
    def __init__(self):
        self.frame_1 = None
        self.processed_frame_1 = None
        self.frame_2 = None
        self.processed_frame_2 = None
        self.mp_image = None
        self.ret_cam_1 = False
        self.ret_cam_2 = False
        self.FirstLoop_cam_1 = True
        self.FirstLoop_cam_2 = True
        self.pax_counter = 0
        # self.pax_counter_list = [0, 0, 0, 0, 0]
        self.pax_box_list = []
        self.right_hand = (0, 0.0)
        self.left_hand = (0, 0.0)
        self.hands_counter = 0
        # self.hands_counter_list = [0, 0, 0, 0, 0]
        self.save_data_when_not_capture_validity = True
        self.save_empty_frame = True
        self.save_hand_landmark_json = False
        self.save_vectors_info = False
        self.save_fullroi = False
        self.save_personcrop = False
        self.save_rawroi = True


class systemStatusHandler_object():
    def __init__(self):
        self.last_inferenced_cam = -1
        self.save_data = True
        self.no_of_cameras = 2
        self.record_mode = "semi-automatic"  # automatic (continuing stream) or semi-automatic (1 record each cam for activation)
        self.operation_mode = 1  # 0 is Left Thumb, 1 Left Index, 2 Left Four, 3 Right Thumb, 4 Right Index, 5 Right Four, 6 two thumbs
        self.recorded_memory = [0 for x in range(self.no_of_cameras)]
        # self.truelabel = metadata.operation_mode_to_true_label_dict[self.operation_mode]
        self.data_df = pd.read_csv(metadata.HOME_DIR + 'data_info.csv')
        self.idx = 0 if self.data_df.shape[0] == 0 else self.data_df['index'].iloc[-1]
        self.webcam = True


statusHandler = camStatusHandler_object()
statusHandler2 = copy.deepcopy(camStatusHandler_object())
mainHandler = systemStatusHandler_object()

# cam 1 and 2  models
paxmodel = modelloader("CCTV_MP_PERSON_PREDICTOR")
paxmodel_2 = modelloader("CCTV_MP_PERSON_PREDICTOR")

# handmodel = modelloader("CCTV_MP_HAND_PREDICTOR")
LRhandmodel = modelloader("CCTV_MP_LRHAND_PREDICTOR",
                          threshold=metadata.LRhand_detector_threshold_dict[mainHandler.operation_mode])
LRhandmodel_2 = modelloader("CCTV_MP_LRHAND_PREDICTOR",
                          threshold=metadata.LRhand_detector_threshold_dict[mainHandler.operation_mode])

handlandmarkmodel = MMPoseInferencerObj(threshold=metadata.hand_detector_threshold_dict[mainHandler.operation_mode],
                                        model_name='HRNET_1',
                                        finger_of_interest=metadata.operation_mode_to_hand_foi_dict[
                                                               mainHandler.operation_mode][1:],
                                        hand_of_interest=
                                        metadata.operation_mode_to_hand_foi_dict[mainHandler.operation_mode][0],
                                        device='cpu',
                                        cam_index=1,
                                        operation_mode=mainHandler.operation_mode)
handdetectormodel = MMDetInferencerObj(model_name='FASTERRCNN_LH_DETECTOR_V3')

handlandmarkmodel_2 = MMPoseInferencerObj(threshold=0.0,
                                        model_name='HRNET_1',
                                        finger_of_interest=metadata.operation_mode_to_hand_foi_dict[
                                                               mainHandler.operation_mode][1:],
                                        hand_of_interest=
                                        metadata.operation_mode_to_hand_foi_dict[mainHandler.operation_mode][0],
                                        device='cuda',
                                        cam_index=2,
                                        operation_mode=mainHandler.operation_mode)
handdetectormodel_2 = MMDetInferencerObj(model_name='FASTERRCNN_LH_DETECTOR_V3')

hand_formation_detector_top_view = MMDetInferencerObj(model_name='FASTERRCNN_FORMATION_DETECTOR_V1', device='cpu')


def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_folder_if_not_exists(metadata.hand_landmark_json_folder)
create_folder_if_not_exists(metadata.vectors_folder)
create_folder_if_not_exists(metadata.fullroi_folder)
create_folder_if_not_exists(metadata.fullroi_folder_2)
create_folder_if_not_exists(metadata.personcrop_folder)
create_folder_if_not_exists(metadata.rawroi_folder)
create_folder_if_not_exists(metadata.rawroi_folder_2)
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

'''def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image'''


'''def count_people(pax_predictor_output):
    count = 0
    box_list = []
    for o in pax_predictor_output.detections:
        category_name = o.categories[0].category_name
        if category_name == 'person':
            count += 1
            box_list.append(o.bounding_box)
    return count, box_list'''


'''def count_hands(hands_predictor_output):
    count = len(hands_predictor_output.handedness)
    if not count == 2:
        for o in hands_predictor_output.handedness:
            category_name = o[0].category_name
            if category_name == 'Right':
                statusHandler.right_hand = 1
            elif category_name == 'Left':
                statusHandler.left_hand = 1
    elif count == 1:
        category_name = hands_predictor_output.handedness[0][0]
        if category_name == 'Right':
            statusHandler.right_hand = 1
            statusHandler.left_hand = 0
        elif category_name == 'Left':
            statusHandler.left_hand = 1
            statusHandler.right_hand = 0
    elif count == 0:
        statusHandler.right_hand = 0
        statusHandler.left_hand = 0
    return count'''


def draw_fs_scanner_box_2(frame: np.ndarray) -> np.ndarray:
    box = metadata.finger_reader_quadrilateral_2
    for i in range(4):
        cv2.line(frame, box[i], box[(i + 1) % 4], color=(0, 255, 0), thickness=1)
    return frame


def check_record_memory_all_1(rm)-> bool:
    valid = False
    for m in rm:
        if m == 0:
            valid = False
            break
        else:
            valid = True
            continue
    return valid


def draw_handlandmarkmodel_status(f, cam_index):
    # Update the key listener status if required first
    if key_listener.record_key_pressed and mainHandler.record_mode == "semi-automatic" and check_record_memory_all_1(mainHandler.recorded_memory):
        key_listener.record_key_pressed = False
        mainHandler.recorded_memory = [0 for _ in range(mainHandler.no_of_cameras)]
    f0 = copy.deepcopy(f)
    f1 = handlandmarkmodel.draw_status_message(f0) if cam_index == 1 else handlandmarkmodel_2.draw_status_message(f0)
    f2 = key_listener.draw_key_status_message(f1)
    return f2


'''def draw_fs_scanner_box(frame: np.ndarray) -> np.ndarray:
    box = metadata.finger_reader_quadrilateral
    for i in range(4):
        cv2.line(frame, box[i], box[(i + 1) % 4], color=(0, 255, 0), thickness=1)
    return frame'''


'''def update_hand_status(LRDetection):
    # Update using prediction of LR dector, If hands are present in image(frame)
    rh = 0
    lh = 0
    detection = 0
    if LRDetection.multi_hand_landmarks:
        detection = len(LRDetection.multi_handedness)
        for i in LRDetection.multi_handedness:
            label = MessageToDict(i)['classification'][0]['label']
            if label == 'Left':
                lh = 1
            if label == 'Right':
                rh = 1
    return lh, rh, detection
'''

'''def process_vectors_list(critical_points:tuple, capture_validity:bool)->dict:
    cp_str = str(critical_points)
    cv_str = str(capture_validity)
    with open()
    '''
def get_next_idx():
    mainHandler.idx = mainHandler.idx + 1
    return mainHandler.idx


def infer_and_save_cam1():
    counter = 0
    try:
        while True:
            # cam 1 inferencer
            if statusHandler.ret_cam_1:
                time_of_frame_received = time.time()
                # print("new inference", 1, counter)
                idx = get_next_idx()
                cam_1_inference(counter, paxmodel, handdetectormodel, LRhandmodel, handlandmarkmodel, statusHandler,
                                mainHandler, key_listener, time_of_frame_received, idx)
                # print('total time taken 1', time.time() - time_of_frame_received)
                statusHandler.ret_cam_1 = False
                counter += 1
            else:
                time.sleep(0.2)
    except Exception as e:
        print('error')
        print(e)



def infer_and_save_cam2():
    counter = 0
    try:
        while True:
            # cam 2 inferencer
            if statusHandler2.ret_cam_2:
                time_of_frame_received = time.time()
                # print("new inference", 2, counter)
                idx = get_next_idx()
                cam_2_inference(counter, paxmodel_2, handdetectormodel_2, LRhandmodel_2, handlandmarkmodel_2, statusHandler2, mainHandler, key_listener, time_of_frame_received, idx, hand_formation_detector_top_view)
                # print('total time taken 2', time.time() - time_of_frame_received)
                statusHandler2.ret_cam_2 = False
                counter += 1
            else:
                time.sleep(0.2)
    except Exception as e:
        print('error 2')
        print(e)

'''def infer_and_save_cam2():
    counter = 0
    while True:
        cam_index = 2
        time.sleep(0.3)
        # cam 2 inferencer
        if statusHandler.ret_cam_2:
            roiframe_2 = statusHandler.frame_2[250:, 170:650]
            roiframe_2 = cv2.flip(roiframe_2, 1)
            draw_fs_scanner_box_2(roiframe_2)
            # roiframe = draw_fs_scanner_box(roiframe)
            rgbframe_2 = cv2.cvtColor(roiframe_2, cv2.COLOR_BGR2RGB)
            bgrframe_2 = copy.copy(roiframe_2)
            capture_validity_2 = False

            if mainHandler.save_data and key_listener.record_key_pressed:
                idx = get_next_idx()
                # Append new row in df if saving data
                time_now = time.time()
                # cam_index = 2
                new_row = [idx] + metadata.operation_mode_to_true_label_dict[
                    mainHandler.operation_mode] + [capture_validity_2, time_now, cam_index]
                statusHandler.data_df.loc[len(statusHandler.data_df)] = new_row
                statusHandler.data_df.to_csv(metadata.HOME_DIR + "data_info.csv", index=False)
                print('d')
                DCH.save_data(statusHandler, None, bgrframe_2, None, None,
                              capture_validity_2, idx, cam_index)
            statusHandler.processed_frame_2 = bgrframe_2
            # Base case status update
            if statusHandler.FirstLoop_cam_2:
                statusHandler.FirstLoop_cam_2 = False
            statusHandler.ret_cam_2 = False'''


def stream_cam_1():
    # Use OpenCV to read the video stream
    cap = cv2.VideoCapture(rtsp_url)
    cv2.namedWindow("CCTV Stream 1", cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            statusHandler.ret_cam_1 = ret
            if ret:
                statusHandler.frame_1 = frame
                if not statusHandler.FirstLoop_cam_1:
                    df = draw_handlandmarkmodel_status(statusHandler.processed_frame_1, cam_index=1)
                    cv2.imshow("CCTV Stream 1", df)
            else:
                print('warning no ret')

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        print('error on cam 1.. restarting..')
        print(e)
        stream_cam_1()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def stream_cam_2():
    # Use OpenCV to read the video stream
    cap = cv2.VideoCapture(rtsp_url2)
    cv2.namedWindow("CCTV Stream 2", cv2.WINDOW_NORMAL)
    print("Starting 2nd stream")
    try:
        while True:
            ret, frame = cap.read()
            statusHandler2.ret_cam_2 = ret
            if ret:
                # print("detect 2nd cam frame!", statusHandler2.ret_cam_2 )
                statusHandler2.frame_2 = frame
                '''if statusHandler2.FirstLoop_cam_2:
                    time.sleep(1)'''
                if not statusHandler2.FirstLoop_cam_2:
                    df = draw_handlandmarkmodel_status(statusHandler2.processed_frame_2, cam_index=2)
                    cv2.imshow("CCTV Stream 2", df)
            else:
                print('warning no ret')
            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        print('error on cam 2.. restarting..')
        print(e)
        stream_cam_2()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def stream_webcam():
    # Use OpenCV to read the video stream
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    print("Starting webcam stream")
    try:
        while True:
            ret, frame = cap.read()
            statusHandler2.ret_cam_2 = ret
            if ret:
                # print("detect 2nd cam frame!", statusHandler2.ret_cam_2 )
                statusHandler2.frame_2 = frame
                '''if statusHandler2.FirstLoop_cam_2:
                    time.sleep(1)'''
                if not statusHandler2.FirstLoop_cam_2:
                    df = draw_handlandmarkmodel_status(statusHandler2.processed_frame_2, cam_index=2)
                    cv2.imshow("CCTV Stream 2", df)
            else:
                print('warning no ret')
            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        print('error on cam 2.. restarting..')
        print(e)
        stream_webcam()
    finally:
        cap.release()
        cv2.destroyAllWindows()

#time.sleep(0.1)
key_listener_thread = Thread(target=key_listener.start_listener)
key_listener_thread.start()

'''streaming_thread_1 = Thread(target=stream_cam_1)
streaming_thread_1.start()

infer_thread_1 = Thread(target=infer_and_save_cam1)
infer_thread_1.start()
'''
if mainHandler.webcam:
    streaming_thread_2 = Thread(target=stream_webcam)
    streaming_thread_2.start()
else:
    streaming_thread_2 = Thread(target=stream_cam_2)
    streaming_thread_2.start()

infer_thread_2 = Thread(target=infer_and_save_cam2)
infer_thread_2.start()
