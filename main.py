import cv2
import os
import time

import numpy as np
import pandas as pd

from metadata import DatasetMeta
import mediapipe as mp
from model import modelloader
from threading import Thread
from MMPoseInferencer import MMPoseInferencerObj
from google.protobuf.json_format import MessageToDict
import DataCollectionHandler as DCH

metadata = DatasetMeta()

# Define the path to the video stream (replace with your actual RTSP URL)
# rtsp_url = 'rtsp://root:pass@192.168.70.2/axis-media/media.amp?videocodec=h264&resolution=640x360'
rtsp_url = 'rtsp://admin:P@ssw0rd@192.168.1.100:554/profile1/media.smp'


# Define the directory where to save the images
class statusHandler():
    def __init__(self):
        self.frame = None
        self.mp_image = None
        self.ret = False
        self.processed_frame = None
        self.FirstLoop = True
        self.pax_counter = 0
        # self.pax_counter_list = [0, 0, 0, 0, 0]
        self.pax_box_list = []
        self.right_hand = 0
        self.left_hand = 0
        self.hands_counter = 0
        # self.hands_counter_list = [0, 0, 0, 0, 0]
        self.operation_mode = 5  # 0 is Left Thumb, 1 Left Index, 2 Left Four, 3 Right Thumb, 4 Right Index, 5 Right Four, 6 two thumbs
        self.save_data = True
        self.save_data_when_not_capture_validity = False
        self.save_hand_landmark_json = True
        self.save_vectors_info = True
        self.save_fullroi = False
        self.save_personcrop = False
        self.save_rawroi = True
        # self.truelabel = metadata.operation_mode_to_true_label_dict[self.operation_mode]
        self.data_df = pd.read_csv(metadata.HOME_DIR + 'data_info.csv')


statusHandler = statusHandler()

paxmodel = modelloader("CCTV_MP_PERSON_PREDICTOR")
handmodel = modelloader("CCTV_MP_HAND_PREDICTOR")
LRhandmodel = modelloader("CCTV_MP_LRHAND_PREDICTOR",
                          threshold=metadata.LRhand_detector_threshold_dict[statusHandler.operation_mode])
handlandmarkmodel = MMPoseInferencerObj(threshold=metadata.hand_detector_threshold_dict[statusHandler.operation_mode],
                                        model_name='HRNET_1',
                                        finger_of_interest=metadata.operation_mode_to_hand_foi_dict[
                                                               statusHandler.operation_mode][1:],
                                        hand_of_interest=
                                        metadata.operation_mode_to_hand_foi_dict[statusHandler.operation_mode][0])


def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_folder_if_not_exists(metadata.hand_landmark_json_folder)
create_folder_if_not_exists(metadata.vectors_folder)
create_folder_if_not_exists(metadata.fullroi_folder)
create_folder_if_not_exists(metadata.personcrop_folder)
create_folder_if_not_exists(metadata.rawroi_folder)

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


def count_people(pax_predictor_output):
    count = 0
    box_list = []
    for o in pax_predictor_output.detections:
        category_name = o.categories[0].category_name
        if category_name == 'person':
            count += 1
            box_list.append(o.bounding_box)
    return count, box_list


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


def draw_handlandmarkmodel_status(f):
    # if not statusHandler.FirstLoop:
    # print('d')
    f = handlandmarkmodel.draw_status_message(f)
    return f

def draw_fs_scanner_box(frame:np.ndarray)->np.ndarray:
    box = metadata.finger_reader_quadrilateral
    for i in range(4):
        cv2.line(frame, box[i], box[(i + 1) % 4], color=(0,255,0), thickness=1)
    return frame
def update_hand_status(LRDetection):
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

'''def process_vectors_list(critical_points:tuple, capture_validity:bool)->dict:
    cp_str = str(critical_points)
    cv_str = str(capture_validity)
    with open()
    '''


def infer_and_save():
    counter = 0
    while True:
        if statusHandler.ret:
            roiframe = statusHandler.frame[200:, 164:570]
            # print("shape of roiframe", roiframe.shape) # 376. 406
            roiframe = cv2.flip(roiframe, 1)
            roiframe = draw_fs_scanner_box(roiframe)
            rgbframe = cv2.cvtColor(roiframe, cv2.COLOR_BGR2RGB)
            # print('format compare', rgbframe)
            statusHandler.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbframe)
            counter += 1
            person_output = paxmodel.person_modeler(statusHandler.mp_image, counter)
            statusHandler.pax_counter, statusHandler.pax_box_list = count_people(person_output[-1])
            # statusHandler.pax_counter_list.append(statusHandler.pax_counter)
            if statusHandler.pax_counter == 1:
                statusHandler.pax_box_list[0].origin_x = max(0, statusHandler.pax_box_list[0].origin_x +
                                                             metadata.operation_mode_xy_offset[
                                                                 statusHandler.operation_mode][0])
                statusHandler.pax_box_list[0].origin_y = max(0, statusHandler.pax_box_list[0].origin_y +
                                                             metadata.operation_mode_xy_offset[
                                                                 statusHandler.operation_mode][1])
                statusHandler.pax_box_list[0].width += \
                    metadata.operation_mode_widthheight_offset[statusHandler.operation_mode][0]
                statusHandler.pax_box_list[0].height += \
                    metadata.operation_mode_widthheight_offset[statusHandler.operation_mode][1]
                person_frame_min_x = statusHandler.pax_box_list[0].origin_x
                person_frame_max_x = statusHandler.pax_box_list[0].origin_x + statusHandler.pax_box_list[0].width
                person_frame_min_y = statusHandler.pax_box_list[0].origin_y
                person_frame_max_y = statusHandler.pax_box_list[0].origin_y + statusHandler.pax_box_list[0].height
                personframe = rgbframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
                personframe_bgr = roiframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
                LRDetection = LRhandmodel.LRhand_modeler(personframe)  # must be RGB not BGR if not wont work

                statusHandler.left_hand, statusHandler.right_hand, hand_detections = update_hand_status(LRDetection)
                # statusHandler.left_hand, statusHandler.right_hand = 1, 1 # Bypass handcounter for now.
                print("LH/RH ", statusHandler.left_hand, statusHandler.right_hand)
                LR_status_pass = (statusHandler.left_hand, statusHandler.right_hand) == \
                                 metadata.operation_mode_to_hand_foi_dict[statusHandler.operation_mode][0] \
                                 and hand_detections == sum(handlandmarkmodel.hand_of_interest)
                if LR_status_pass:
                    # Feed new frame into handlandmarkmodel
                    landmarks = handlandmarkmodel(personframe_bgr, statusHandler.pax_box_list[0],
                                                  file_name=counter)
                    statusHandler.hands_counter = handlandmarkmodel.landmarks_dict['hands_count']
                    if statusHandler.hands_counter == sum(
                            handlandmarkmodel.hand_of_interest):  # No of hand of interest matches
                        # preds_list = landmarks[0]
                        prediction_1 = landmarks
                        _, success = handlandmarkmodel.get_and_update_landmarks_dict(prediction_1,
                                                                                     statusHandler.pax_box_list[0])
                        vectors = handlandmarkmodel.get_all_vectors_list()
                        print(vectors)
                        capture_validity = handlandmarkmodel.validity_check()
                        # vector_dict = process_vectors_list(v, capture_validity)
                        # print("FOI vectors text", len(handlandmarkmodel.finger_of_interest), sum(handlandmarkmodel.hand_of_interest))
                        if capture_validity:
                            print("Capture success! Sending signal to Ah Boon")
                            if len(handlandmarkmodel.finger_of_interest) == 1 and sum(
                                    handlandmarkmodel.hand_of_interest) == 1:  # Label the angles on frame if only 1 FOI and 1 HOI
                                statusHandler.processed_frame = handlandmarkmodel.draw_foi_vectors_text(
                                    roiframe)
                                statusHandler.processed_frame = handlandmarkmodel.draw_fingers_vectors(
                                    statusHandler.processed_frame)
                            else:
                                statusHandler.processed_frame = handlandmarkmodel.draw_fingers_vectors(
                                    roiframe)
                            idx = 0 if statusHandler.data_df.shape[0] == 0 else statusHandler.data_df['index'].iloc[
                                                                                    -1] + 1
                            if statusHandler.save_data:
                                # Append new row in df if saving data
                                new_row = [idx] + metadata.operation_mode_to_true_label_dict[
                                    statusHandler.operation_mode] + [capture_validity]
                                statusHandler.data_df.loc[len(statusHandler.data_df)] = new_row
                                statusHandler.data_df.to_csv(metadata.HOME_DIR + "data_info.csv", index=False)
                                DCH.save_data(statusHandler, landmarks, roiframe, personframe, vectors, capture_validity, idx)
                            statusHandler.processed_frame = paxmodel.visualize(statusHandler.processed_frame,
                                                                               person_output[-1])
                        else:
                            idx = 0 if statusHandler.data_df.shape[0] == 0 else statusHandler.data_df['index'].iloc[
                                                                                    -1] + 1
                            if statusHandler.save_data_when_not_capture_validity:
                                # Append new row in df if saving data
                                new_row = [idx] + metadata.operation_mode_to_true_label_dict[
                                    statusHandler.operation_mode] + [capture_validity]
                                statusHandler.data_df.loc[len(statusHandler.data_df)] = new_row
                                statusHandler.data_df.to_csv(metadata.HOME_DIR + "data_info.csv", index=False)
                                DCH.save_data(statusHandler, landmarks, roiframe, personframe, vectors,
                                              capture_validity, idx)

                            # Draw finger vectors after capture validity fail to visualize which non FOI is inside FS
                            statusHandler.processed_frame = handlandmarkmodel.draw_fingers_vectors(
                                roiframe)

                            # Draw normal ROI frame as failed capture validity test
                            statusHandler.processed_frame = paxmodel.visualize(statusHandler.processed_frame,
                                                                           person_output[-1])
                    else:
                        # Draw normal ROI since no. of hand predict by handlandmark isnt matching required
                        statusHandler.processed_frame = paxmodel.visualize(roiframe, person_output[-1])
                else:
                    handlandmarkmodel.status_message = ("Wrong hand presented", (0, 0, 255))
                    statusHandler.processed_frame = paxmodel.visualize(roiframe, person_output[-1])
                handlandmarkmodel.clear_cache()  # Remove this inference memory after retrieving output
            else:
                statusHandler.processed_frame = roiframe
            # Base case status update
            if statusHandler.FirstLoop and statusHandler.ret:
                handlandmarkmodel.firstLoop_Inference(statusHandler.frame)
                statusHandler.FirstLoop = False
            statusHandler.ret = False


def stream():
    # Use OpenCV to read the video stream
    cap = cv2.VideoCapture(rtsp_url)
    cv2.namedWindow("CCTV Stream", cv2.WINDOW_NORMAL)
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            statusHandler.ret = ret
            if ret:

                statusHandler.frame = frame
                if not statusHandler.FirstLoop:
                    statusHandler.processed_frame = draw_handlandmarkmodel_status(statusHandler.processed_frame)
                    cv2.imshow("CCTV Stream", statusHandler.processed_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


streaming_thread = Thread(target=stream)
streaming_thread.start()

infer_thread = Thread(target=infer_and_save)
infer_thread.start()
