import cv2
import copy
from metadata import DatasetMeta
import mediapipe as mp
import numpy as np
import time
from google.protobuf.json_format import MessageToDict
import DataCollectionHandler as DCH

metadata = DatasetMeta()

def convert_int_prediction_to_str_prediction(i:int)-> str:
    d = {0:'leftthumb',1:'leftindex',2:'leftfour',3:'rightthumb',4:'rightindex',5:'rightfour',6:'emptyscanner'}
    p = d[i]
    return p
def draw_fs_scanner_box_1(frame: np.ndarray) -> np.ndarray:
    box = metadata.finger_reader_quadrilateral_1
    for i in range(4):
        cv2.line(frame, box[i], box[(i + 1) % 4], color=(0, 255, 0), thickness=1)
    return frame


def draw_fs_scanner_box_2(frame: np.ndarray) -> np.ndarray:
    box = metadata.finger_reader_quadrilateral_2
    for i in range(4):
        cv2.line(frame, box[i], box[(i + 1) % 4], color=(0, 255, 0), thickness=1)
    return frame


def count_people(pax_predictor_output):
    count = 0
    box_list = []
    for o in pax_predictor_output.detections:
        category_name = o.categories[0].category_name
        if category_name == 'person':
            count += 1
            box_list.append(o.bounding_box)
    return count, box_list

def update_hand_status(LRDetection):
    # Update using prediction of LR dector, If hands are present in image(frame)
    rh = (0, 0.0)
    lh = (0, 0.0)
    detection = 0
    if LRDetection.multi_hand_landmarks:
        detection = len(LRDetection.multi_handedness)
        for i in LRDetection.multi_handedness:
            label = MessageToDict(i)['classification'][0]['label']
            score = MessageToDict(i)['classification'][0]['score']
            if label == 'Left':
                lh = (1, score)
            if label == 'Right':
                rh = (1, score)
    return lh, rh, detection


def calculate_hand_detection_scores(detection: dict) -> tuple:
    predictions = detection['predictions'][0]
    labels = predictions['labels']
    scores = predictions['scores']
    score_dict = {0: 0.0, 1: 0.0}
    for i, label in enumerate(labels):
        score = scores[i]
        if score > 0.5:
            score_dict[label] += score
    # Filter the hand score that is lower than the other, reason being, current model only detect 1 hand optimally
    if score_dict[0] > score_dict[1]:
        score_dict[1] = 0.
    else:
        score_dict[0] = 0.
    return score_dict

def calibrate_hand_detection_score(MP_Result, custom_object_detection_score)->float:
    ''' Ensemble the prediction scores of MP hand detection model and custom object detection scores
        Input: MP result (prediction:bool, score:float) and Custom ObjectDetection score:float
        Output: valid:bool 1 if total score crosses defined threshold else 0
    '''
    total_score = MP_Result[1] + custom_object_detection_score
    valid = 1 if total_score > 0.6 else 0
    return valid


def cam_1_inference(counter, paxmodel, handdetectormodel, LRhandmodel, handlandmarkmodel, statusHandler, mainHandler, key_listener, time_of_frame_received, idx):
    roiframe = statusHandler.frame_1[200:, 164:570]
    cam_index = 1
    # print("shape of roiframe", roiframe.shape) # 376. 406
    roiframe = cv2.flip(roiframe, 1)
    roiframe = draw_fs_scanner_box_1(roiframe)
    rgbframe = cv2.cvtColor(roiframe, cv2.COLOR_BGR2RGB)


    bgrframe = copy.copy(roiframe)
    customhanddetection = handdetectormodel(bgrframe)
    score_dict = calculate_hand_detection_scores(customhanddetection)
    # print("custom detection 1", score_dict)
    # base case status update
    if statusHandler.FirstLoop_cam_1:
        handlandmarkmodel.firstLoop_Inference(roiframe)
        statusHandler.FirstLoop_cam_1 = False
        # print("Completed first loop")

    statusHandler.mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbframe)
    person_output = paxmodel.person_modeler(statusHandler.mp_image_1, counter)
    statusHandler.pax_counter, statusHandler.pax_box_list = count_people(person_output[-1])
    # statusHandler.pax_counter_list.append(statusHandler.pax_counter)
    if statusHandler.pax_counter == 1:
        statusHandler.pax_box_list[0].origin_x = max(0, statusHandler.pax_box_list[0].origin_x +
                                                     metadata.operation_mode_xy_offset[
                                                         mainHandler.operation_mode][0])
        statusHandler.pax_box_list[0].origin_y = max(0, statusHandler.pax_box_list[0].origin_y +
                                                     metadata.operation_mode_xy_offset[
                                                         mainHandler.operation_mode][1])
        statusHandler.pax_box_list[0].width += \
            metadata.operation_mode_widthheight_offset[mainHandler.operation_mode][0]
        statusHandler.pax_box_list[0].height += \
            metadata.operation_mode_widthheight_offset[mainHandler.operation_mode][1]
        person_frame_min_x = statusHandler.pax_box_list[0].origin_x
        person_frame_max_x = statusHandler.pax_box_list[0].origin_x + statusHandler.pax_box_list[0].width
        person_frame_min_y = statusHandler.pax_box_list[0].origin_y
        person_frame_max_y = statusHandler.pax_box_list[0].origin_y + statusHandler.pax_box_list[0].height
        personframe = rgbframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
        personframe_bgr = bgrframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
        LRDetection = LRhandmodel.LRhand_modeler(personframe)  # must be RGB not BGR if not wont work
        LH_MP, RH_MP, hand_detections = update_hand_status(LRDetection)
        statusHandler.left_hand = calibrate_hand_detection_score(LH_MP, score_dict[0]) if mainHandler.operation_mode == 0 or mainHandler.operation_mode == 3 else LH_MP[0]
        statusHandler.right_hand = calibrate_hand_detection_score(RH_MP, score_dict[1]) if mainHandler.operation_mode == 0 or mainHandler.operation_mode == 3 else RH_MP[0]
        # statusHandler.left_hand, statusHandler.right_hand = 1, 0  # Bypass handcounter for now.
        # print("HAND PREDICTION: ", statusHandler.left_hand, statusHandler.right_hand)
        # hand_detections = 1 # by pass
        LR_status_pass = (statusHandler.left_hand, statusHandler.right_hand) == \
                         metadata.operation_mode_to_hand_foi_dict[mainHandler.operation_mode][0]
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

                # Update vecotrs list
                handlandmarkmodel.get_all_vectors_list()
                vectors = handlandmarkmodel.finger_vectors_dict
                # print(vectors)

                capture_validity = handlandmarkmodel.validity_check()
                # vector_dict = process_vectors_list(v, capture_validity)
                # print("FOI vectors text", len(handlandmarkmodel.finger_of_interest), sum(handlandmarkmodel.hand_of_interest))
                if capture_validity:
                    # print("Capture success! Sending signal to Ah Boon. Time of frame inferred", time_of_frame_received)
                    if len(handlandmarkmodel.finger_of_interest) == 1 and sum(
                            handlandmarkmodel.hand_of_interest) == 1:  # Label the angles on frame if only 1 FOI and 1 HOI
                        statusHandler.processed_frame_1 = handlandmarkmodel.draw_foi_vectors_text(
                            roiframe)
                        statusHandler.processed_frame_1 = handlandmarkmodel.draw_fingers_vectors(
                            statusHandler.processed_frame_1)
                    else:
                        statusHandler.processed_frame_1 = handlandmarkmodel.draw_fingers_vectors(
                            roiframe)
                    if mainHandler.save_data and key_listener.record_key_pressed:
                        DCH.save_data(statusHandler, mainHandler, landmarks, bgrframe, personframe_bgr, vectors,
                                      capture_validity, idx, cam_index)
                    statusHandler.processed_frame_1 = paxmodel.visualize(statusHandler.processed_frame_1,
                                                                         person_output[-1])
                else:
                    if statusHandler.save_data_when_not_capture_validity and key_listener.record_key_pressed:
                        DCH.save_data(statusHandler, mainHandler, landmarks, bgrframe, personframe_bgr, vectors,
                                      capture_validity, idx, cam_index)

                    # Draw finger vectors after capture validity fail to visualize which non FOI is inside FS
                    statusHandler.processed_frame_1 = handlandmarkmodel.draw_fingers_vectors(
                        roiframe)

                    # Draw normal ROI frame as failed capture validity test
                    statusHandler.processed_frame_1 = paxmodel.visualize(statusHandler.processed_frame_1,
                                                                         person_output[-1])
            else:
                # Draw normal ROI since no. of hand predict by handlandmark isnt matching required
                statusHandler.processed_frame_1 = paxmodel.visualize(roiframe, person_output[-1])
        else:
            handlandmarkmodel.status_message = ("Wrong hand presented", (0, 0, 255))
            statusHandler.processed_frame_1 = paxmodel.visualize(roiframe, person_output[-1])
        handlandmarkmodel.clear_cache()  # Remove this inference memory after retrieving output
        if statusHandler.save_empty_frame and key_listener.record_key_pressed:
            capture_validity = False
            DCH.save_data(statusHandler, mainHandler, None, statusHandler.processed_frame_1, bgrframe, None,
                          capture_validity, idx, cam_index)
    else:
        statusHandler.processed_frame_1 = roiframe
        if statusHandler.save_empty_frame and key_listener.record_key_pressed:
            capture_validity = False
            DCH.save_data(statusHandler, mainHandler, None, statusHandler.processed_frame_1, bgrframe, None,
                          capture_validity, idx, cam_index)
    mainHandler.last_inferenced_cam = cam_index
    return paxmodel, handdetectormodel, LRhandmodel, handlandmarkmodel, statusHandler, key_listener

def get_first_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return None

def cam_2_inference(counter, paxmodel_2, handdetectormodel_2, LRhandmodel_2, handlandmarkmodel_2, statusHandler2, mainHandler, key_listener, time_of_frame_received, idx, hand_formation_detector_top_view):
    roiframe = statusHandler2.frame_2[247:, 260:545]
    cam_index = 2
    roiframe = cv2.flip(roiframe, 1)
    roiframe = draw_fs_scanner_box_2(roiframe)
    rgbframe = cv2.cvtColor(roiframe, cv2.COLOR_BGR2RGB)
    rgbframe_copy = copy.deepcopy(rgbframe)
    bgrframe = copy.deepcopy(roiframe)
    # base case status update
    if statusHandler2.FirstLoop_cam_2:
        handlandmarkmodel_2.firstLoop_Inference(roiframe)
        statusHandler2.FirstLoop_cam_2 = False
        print("Completed first loop", 2)
    start_custom_cal_time = time.time()
    customdetection = hand_formation_detector_top_view(bgrframe)
    print("CUSTOM DETECTIOn", customdetection)
    if len(customdetection['predictions'][0]['labels']) > 0:
        emptyscanner_idx = get_first_index(customdetection['predictions'][0]['labels'], 6)
        if emptyscanner_idx == None or (emptyscanner_idx != None and customdetection['predictions'][0]['scores'][emptyscanner_idx] < 0.4): # No empty scanner detected
            predicted_int = customdetection['predictions'][0]['labels'][0]
            predicted_score = customdetection['predictions'][0]['scores'][0]
            predicted = convert_int_prediction_to_str_prediction(predicted_int)
            print('FINAL PREDICTION', predicted, predicted_score)
        else:
            print(f"Empty scanner detected {emptyscanner_idx}")
    print(f"Calculation time {round(time.time() - start_custom_cal_time, 3)} \n")

    statusHandler2.mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbframe)
    person_output = paxmodel_2.person_modeler(statusHandler2.mp_image_1, counter)
    statusHandler2.pax_counter, statusHandler2.pax_box_list = count_people(person_output[0])
    # statusHandler.pax_counter_list.append(statusHandler.pax_counter)
    if statusHandler2.pax_counter == 1:
        person_frame_min_x = statusHandler2.pax_box_list[0].origin_x
        person_frame_max_x = statusHandler2.pax_box_list[0].origin_x + statusHandler2.pax_box_list[0].width
        person_frame_min_y = statusHandler2.pax_box_list[0].origin_y
        person_frame_max_y = statusHandler2.pax_box_list[0].origin_y + statusHandler2.pax_box_list[0].height
        personframe = rgbframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
        personframe_bgr = bgrframe[person_frame_min_y:person_frame_max_y, person_frame_min_x:person_frame_max_x]
        LRDetection = LRhandmodel_2.LRhand_modeler(rgbframe)  # must be RGB not BGR if not wont work

        LH_MP, RH_MP, hand_detections = update_hand_status(LRDetection)
        statusHandler2.left_hand = LH_MP[0]
        statusHandler2.right_hand = RH_MP[0]
        # statusHandler2.left_hand, statusHandler2.right_hand = 1, 0  # Bypass handcounter for now.
        # hand_detections = 1 # by pass
        LR_status_pass = (statusHandler2.left_hand, statusHandler2.right_hand) == \
                         metadata.operation_mode_to_hand_foi_dict[mainHandler.operation_mode][0]
        # print("HAND PREDICTION 2: ", statusHandler2.left_hand, statusHandler2.right_hand, LR_status_pass)
        if LR_status_pass:
            # Feed new frame into handlandmarkmodel
            landmarks = handlandmarkmodel_2(bgrframe, statusHandler2.pax_box_list[0],
                                          file_name=counter, use_pf=False)
            statusHandler2.hands_counter = handlandmarkmodel_2.landmarks_dict['hands_count']
            if statusHandler2.hands_counter == sum(
                    handlandmarkmodel_2.hand_of_interest):  # No of hand of interest matches
                # preds_list = landmarks[0]
                prediction_1 = landmarks
                _, success = handlandmarkmodel_2.get_and_update_landmarks_dict(prediction_1,
                                                                               statusHandler2.pax_box_list[0],
                                                                               use_pax_box_offset=False)

                # Update vecotrs list
                handlandmarkmodel_2.get_all_vectors_list()
                vectors = handlandmarkmodel_2.finger_vectors_dict
                # print(vectors)

                capture_validity = handlandmarkmodel_2.validity_check()
                # vector_dict = process_vectors_list(v, capture_validity)
                # print("FOI vectors text", len(handlandmarkmodel.finger_of_interest), sum(handlandmarkmodel.hand_of_interest))
                if capture_validity:
                    print("Capture success! Sending signal to Ah Boon. Time of frame inferred",
                          time_of_frame_received)
                    if len(handlandmarkmodel_2.finger_of_interest) == 1 and sum(
                            handlandmarkmodel_2.hand_of_interest) == 1:  # Label the angles on frame if only 1 FOI and 1 HOI
                        statusHandler2.processed_frame_2 = handlandmarkmodel_2.draw_foi_vectors_text(roiframe)
                        statusHandler2.processed_frame_2 = handlandmarkmodel_2.draw_fingers_vectors(statusHandler2.processed_frame_2)
                    else:
                        statusHandler2.processed_frame_2 = handlandmarkmodel_2.draw_fingers_vectors(roiframe)
                    if mainHandler.save_data and key_listener.record_key_pressed:
                        DCH.save_data(statusHandler2, mainHandler, landmarks, bgrframe, personframe_bgr, vectors,
                                      capture_validity, idx, cam_index)
                    statusHandler2.processed_frame_2 = paxmodel_2.visualize(statusHandler2.processed_frame_2,
                                                                         person_output[-1])
                else:
                    if statusHandler2.save_data_when_not_capture_validity and key_listener.record_key_pressed:
                        DCH.save_data(statusHandler2, mainHandler, landmarks, bgrframe, personframe_bgr, vectors,
                                      capture_validity, idx, cam_index)

                    # Draw finger vectors after capture validity fail to visualize which non FOI is inside FS
                    statusHandler2.processed_frame_2 = handlandmarkmodel_2.draw_fingers_vectors(
                        roiframe)

                    # Draw normal ROI frame as failed capture validity test
                    statusHandler2.processed_frame_2 = paxmodel_2.visualize(statusHandler2.processed_frame_2,
                                                                         person_output[-1])
            else:
                # Draw normal ROI since no. of hand predict by handlandmark isnt matching required
                statusHandler2.processed_frame_2 = paxmodel_2.visualize(roiframe, person_output[-1])
        else:
            handlandmarkmodel_2.status_message = ("Wrong hand presented", (0, 0, 255))
            statusHandler2.processed_frame_2 = paxmodel_2.visualize(roiframe, person_output[-1])
        handlandmarkmodel_2.clear_cache()  # Remove this inference memory after retrieving output
        if statusHandler2.save_empty_frame and key_listener.record_key_pressed:
            capture_validity = False
            DCH.save_data(statusHandler2, mainHandler, landmarks=None, roiframe=bgrframe, personframe=bgrframe, vectors=None,
                          capture_validity=capture_validity, idx=idx, cam_index=cam_index)
    else:
        statusHandler2.processed_frame_2 = roiframe
        if statusHandler2.save_empty_frame and key_listener.record_key_pressed:
            capture_validity = False
            DCH.save_data(statusHandler2, mainHandler, landmarks=None, roiframe=bgrframe, personframe=bgrframe, vectors=None,
                          capture_validity=capture_validity, idx=idx, cam_index=cam_index)
    mainHandler.last_inferenced_cam = cam_index
    return paxmodel_2, handdetectormodel_2, LRhandmodel_2, handlandmarkmodel_2, statusHandler2, key_listener
