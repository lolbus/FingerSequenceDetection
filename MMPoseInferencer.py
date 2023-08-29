
from mmpose.apis import MMPoseInferencer
from metadata import DatasetMeta
import numpy as np
import cv2
metadata = DatasetMeta()
'''#from mmdet.apis.det_inferencer import DetInferencer


# build the inferencer with model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose2d='C:/Users/WayneGuangWayTENGSof/Desktop/PoseMM/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py',
    pose2d_weights='C:/Users/WayneGuangWayTENGSof/Desktop/PoseMM/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth')


import time

start = time.time()
result_generator = inferencer(('C:/Users/WayneGuangWayTENGSof/Desktop/PoseMM/test'), batch_size=1, kpt_thr=0., vis_out_dir='vis_results')
for _ in result_generator:
    result = _

print("time_taken", time.time() - start)
'''
COCO_LABELS = {0: "Wrist", 1: "Thumbbase1", 2: "Thumbbase2", 3: "Thumbtip1", 4: "Thumbtip2",
               5: "Indexbase1", 6: "Indexbase2", 7: "Indextip1", 8: "Indextip2",
               9: "Middlebase1", 10: "Middlebase2", 11: "Middletip1", 12: "Middletip2",
               13: "Ringbase1", 14: "Ringbase2", 15: "Ringtip1", 16: "Ringtip2",
               17: "Pinkybase1", 18: "Pinkybase2", 19: "Pinkytip1", 20: "Pinkytip2"}
COCO_LABELS_REVERSE = {"Wrist": 0, "Thumbbase1": 1, "Thumbbase2": 2, "Thumbtip1": 3, "Thumbtip2": 4,
               "Indexbase1": 5, "Indexbase2": 6, "Indextip1": 7, "Indextip2": 8,
               "Middlebase1": 9, "Middlebase2": 10, "Middletip1": 11, "Middletip2": 12,
               "Ringbase1": 13, "Ringbase2": 14, "Ringtip1": 15, "Ringtip2": 16,
               "Pinkybase1": 17, "Pinkybase2": 18, "Pinkytip1": 19, "Pinkytip2": 20}

def convert_landmark_point_to_xy_point(point)->np.ndarray:
    return np.array((point[1], point[0]))

def calculate_vector_angle(start_point, end_point) -> float:
    vector = end_point - start_point
    direction = np.arctan2(vector[1], vector[0])
    return direction

def calculate_vector_length(start_point, end_point) -> float:
    vector = end_point - start_point
    d = np.linalg.norm(vector, ord=2)
    return d

def compute_barycentric_coords(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0).astype(np.int64)
    d01 = np.dot(v0, v1).astype(np.int64)
    d11 = np.dot(v1, v1).astype(np.int64)
    d20 = np.dot(v2, v0).astype(np.int64)
    d21 = np.dot(v2, v1).astype(np.int64)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return [u, v, w]

def is_inside_finger_reader_space(p, cam_index):
    if cam_index == 2:
        a = metadata.finger_reader_quadrilateral_2[0]
        b = metadata.finger_reader_quadrilateral_2[1]
        c = metadata.finger_reader_quadrilateral_2[2]
        d = metadata.finger_reader_quadrilateral_2[3]
    else:
        a = metadata.finger_reader_quadrilateral_1[0]
        b = metadata.finger_reader_quadrilateral_1[1]
        c = metadata.finger_reader_quadrilateral_1[2]
        d = metadata.finger_reader_quadrilateral_1[3]
    triangles = [(a, b, c), (a, c, d)]

    for triangle in triangles:
        u, v, w = compute_barycentric_coords(np.array(p), *triangle)
        if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
            return True
    return False

def calibrate_personFramePoint_to_roiFramePoint(p, personframe_bounding_box, use_pf=True) -> np.ndarray:
    if use_pf:
        personframe_min_x = personframe_bounding_box.origin_x
        personframe_min_y = personframe_bounding_box.origin_y
        return p + np.array([personframe_min_x, personframe_min_y])
    else:
        return p



class MMPoseInferencerObj(object):
    def __init__(self, threshold=0.3, vis_out_dir='vis_results', batch_size=1, model_name='HRNET_1', finger_of_interest="Thumb", hand_of_interest=(1,0), device='cuda', cam_index=1, operation_mode=0):
        self.threshold = threshold
        self.vis_out_dir = vis_out_dir
        self.batch_size = 1
        self.landmarks_dict_list = []
        if model_name == 'HRNET_1':
            self.inferencer = MMPoseInferencer(
                pose2d= metadata.mmPoseModelDir + '/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py',
                pose2d_weights=metadata.mmPoseModelDir + '/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth',
                show_progress=False,
                device=device)
        self.finger_of_interest = finger_of_interest
        self.result = []
        self.landmarks_dict = {'hands_count': 0}
        # self.finger_angles_dict = {'Thumbbase': -3.1416, 'Thumbtipbase': -3.1416, 'Thumbtip': -3.1416}
        self.finger_vectors_dict = {}
        self.foi_tip_point_dict = {}
        self.hand_of_interest = hand_of_interest
        self.status_message = ("Starting up..", (0, 255, 255))
        self.status_updated = False
        self.cam_index = cam_index
        self.operatation_mode = operation_mode
        if self.cam_index == 2:
            self.keypoint_score_threshold = metadata.KEYPOINT_SCORES_THRESHOLD_DICT[f'camindex{self.cam_index} operationmode{self.operatation_mode}']
        else:
            self.keypoint_score_threshold = metadata.KEYPOINT_SCORES_THRESHOLD_DICT['default']

    def __call__(self, input_data, pax_box, file_name=999, use_pf=True):
        result_list = []
        personframe_bounding_box = pax_box
        result_generator = self.inferencer(input_data, batch_size=self.batch_size,
                                           kpt_thr=self.threshold,
                                           vis_out_dir=self.vis_out_dir,
                                           file_name=file_name)
        for r in result_generator:
            predictions = r['predictions'][0]
            if sum(self.hand_of_interest) == 2 and len(self.finger_of_interest) == 1:  # 2 Thumbs Special Case, has 2 hands of interest
                if len(predictions) == sum(self.hand_of_interest):
                    # No capability to distinguish left/right hand for now, just treat as hand 1 or 2
                    hand_1 = predictions[0]
                    hand_2 = predictions[1]
                    foi_tip_index = COCO_LABELS_REVERSE[self.finger_of_interest[0] + "tip2"]
                    foi_tip_point1 = calibrate_personFramePoint_to_roiFramePoint(
                        np.round(np.array(hand_1['keypoints'][foi_tip_index]), 1),
                        personframe_bounding_box, use_pf=use_pf)
                    foi_tip_point2 = calibrate_personFramePoint_to_roiFramePoint(
                        np.round(np.array(hand_2['keypoints'][foi_tip_index]), 1),
                        personframe_bounding_box, use_pf=use_pf)
                    self.foi_tip_point_dict[self.finger_of_interest[0]] = (list(foi_tip_point1), list(foi_tip_point2))
                    if is_inside_finger_reader_space(foi_tip_point1, cam_index=self.cam_index) and is_inside_finger_reader_space(foi_tip_point2, cam_index=self.cam_index):
                        result_list.append(hand_1)
                        result_list.append(hand_2)
                    else:
                        print("Rejecting this predict becus cannot find 2 thumb tip inside FS")
                else:
                    print("Rejecting becus really cant even detect right number of predicts! Predicts:", len(predictions))
            else: # All other case that are not based on 2 hands of interest
                #print(f'analysing {len(predictions)} predictions')
                for p_dict in predictions: # 1 Hand 1 finger results appending
                    if sum(self.hand_of_interest) == 1 and len(self.finger_of_interest) == 1:
                        foi_tip_index = COCO_LABELS_REVERSE[self.finger_of_interest[0] + "tip2"]
                        '''try:
                            print('pdict', p_dict['keypoint_scores'])
                        except Exception as e:
                            pass'''
                        foi_tip_point = calibrate_personFramePoint_to_roiFramePoint(
                        np.round(np.array(p_dict['keypoints'][foi_tip_index]), 1),
                        personframe_bounding_box, use_pf=use_pf)
                        if is_inside_finger_reader_space(foi_tip_point, cam_index=self.cam_index):
                            #print("FOI is inside")
                            result_list.append(p_dict)
                        #else:
                            #print("Rejected as FOI is outside!!", p_dict)
                        self.foi_tip_point_dict[self.finger_of_interest[0]] = list(foi_tip_point)
                    else:
                        # print(f"Incomplete development for multi fingers of interest")
                        pass_check = 0
                        for f in self.finger_of_interest:
                            foi_tip_index = COCO_LABELS_REVERSE[f + "tip2"]
                            foi_tip_point = calibrate_personFramePoint_to_roiFramePoint(
                                np.round(np.array(p_dict['keypoints'][foi_tip_index]), 1),
                                personframe_bounding_box, use_pf=use_pf)
                            if is_inside_finger_reader_space(foi_tip_point, cam_index=self.cam_index):
                                pass_check += 1
                            self.foi_tip_point_dict[f] = list(foi_tip_point)
                        if pass_check == len(self.finger_of_interest):
                            result_list.append(p_dict)
        self.result = result_list
        hand_counts = len(result_list) if len(result_list) > 0 else 0
        self.landmarks_dict['hands_count'] = hand_counts
        if hand_counts <  sum(self.hand_of_interest):
            # print("Not enough hands detected inside box by landmarkmodel!!", hand_counts)
            self.status_message = ("All HOI, FOI not presented in FS", (0, 0, 255))
        elif hand_counts == len(result_list):
            # print("detect quantity correct by landmarkmodel ")
            self.status_message = ("ALL HOI and FOI Present. Proceed to Analyse formation for undesired fingers", (0, 255, 255))
        else:
            print("No comments .. weird.. more than 2 hands in fs")
        # print("returning")
        return result_list

    def firstLoop_Inference(self, f):
        l = self.inferencer(f, batch_size=self.batch_size,
                        kpt_thr=self.threshold)
        for r in l:
            print("Initialization of Inferencer.." )

    def validity_check(self):
        fois = self.finger_of_interest
        results = self.result
        nonfoi_inside_fs_box = [False, False] # Each bool for each hand
        p = False
        fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for f in fingers:
            if f in self.finger_of_interest:
                continue
            else:
                match (self.hand_of_interest, self.finger_of_interest, f, self.cam_index):
                    case ((1, 0), ("Thumb",), "Index", 1): # (If left thumb mode and finger assessing is index, special case
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=False)
                        sum_length = self.finger_vectors_dict['Indexbase0']['length'] + self.finger_vectors_dict['Indextipbase0']['length'] + \
                                     self.finger_vectors_dict['Indextip0']['length']
                        tipendbaseend = self.finger_vectors_dict['Indextipendbaseend0']['length']
                        delta_sum_vs_tipendbaseend= abs(tipendbaseend - sum_length)
                        if delta_sum_vs_tipendbaseend > 0.15 * sum_length:
                            nonfoi_inside_fs_box = [False, False]
                        if nonfoi_inside_fs_box[0] or nonfoi_inside_fs_box[1]:
                            index_tip_end_point_y = self.landmarks_dict_list[0]['Indextip2'][1]
                            thumb_tip_end_point_y = self.landmarks_dict_list[0]['Thumbtip2'][1]
                            if index_tip_end_point_y - thumb_tip_end_point_y > 13:
                                nonfoi_inside_fs_box = [False, False]
                    case((1, 0), ("Thumb",), "Index", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((1, 0), ("Thumb", ), "Middle", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((1, 0), ("Thumb", ), "Ring", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((1, 0), ("Thumb", ), "Pinky", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((0, 1), ("Thumb", ), "Index", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((0, 1), ("Thumb", ), "Middle", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((0, 1), ("Thumb", ), "Ring", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                    case ((0, 1), ("Thumb", ), "Pinky", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)

                    case ((1, 0), ("Index", "Middle", "Ring", "Pinky"), "Thumb", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                        # print("check dict", self.keypoint_score_threshold)
                    case ((0, 1), ("Index", "Middle", "Ring", "Pinky"), "Thumb", 2):
                        nonfoi_inside_fs_box = self.check_entire_finger_tip_inside_fs_box(f, ignore_low_score_tips=True)
                        # print("check dict", self.keypoint_score_threshold)
                    case _:
                        nonfoi_inside_fs_box = self.check_finger_inside_fs_box(f)


                # nonfoi_inside_fs_box = self.check_finger_inside_fs_box(f)
                # print(f"checking nonfoi on 2 hands: {f} .. result {nonfoi_inside_fs_box}")
                if nonfoi_inside_fs_box[0] or nonfoi_inside_fs_box[1]:
                    # print(f"Detect nonfoi, {f} inside box! ")
                    break
            angles_dict = self.finger_angles_dict # To be completed...


        p = not nonfoi_inside_fs_box[0] and not nonfoi_inside_fs_box[1]
        self.status_message = (f"Pass Validity Check", (0, 255, 0)) if p else (f"Failed Validity Check, non FOI {f} in FS", (0, 0, 255))
        return p

    def check_finger_inside_fs_box(self, finger:str):
        '''if len(self.result) == 1:
            check = False
            finger_point = self.landmarks_dict[finger + 'tip2']
            check = is_inside_finger_reader_space(finger_point)
        elif len(self.result) == 2:'''
        check = [False, False]
        for i, landmarks_dict in enumerate(self.landmarks_dict_list):
            finger_point = landmarks_dict[finger + 'tip2']
            check[i] = is_inside_finger_reader_space(finger_point, cam_index=self.cam_index)
        return check

    def check_entire_finger_tip_inside_fs_box(self, finger: str, ignore_low_score_tips = False):
        check = [False, False]
        for i, landmarks_dict in enumerate(self.landmarks_dict_list):
            fingertip1_point = landmarks_dict[finger + 'tip1']
            fingertip2_point = landmarks_dict[finger + 'tip2']
            if not ignore_low_score_tips:
                check[i] = is_inside_finger_reader_space(fingertip1_point, cam_index=self.cam_index) \
                           and is_inside_finger_reader_space(fingertip2_point, cam_index=self.cam_index)
            else:
                fingertip2_score = landmarks_dict[finger + 'tip2 keypoint_scores']
                fingertip2_threshold = self.keypoint_score_threshold[COCO_LABELS_REVERSE[finger + 'tip2']]
                # print('fingertip1score/2score', finger, fingertip2_score, fingertip2_threshold)
                if fingertip2_score < fingertip2_threshold:
                    continue
                else:

                    check[i] = is_inside_finger_reader_space(fingertip2_point, cam_index=self.cam_index)
                    # print('pass status', self.cam_index, check[i])

        return check

    def get_thumb_vectors(self):
        # Get the angle for the first predict
        thumbtip1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Thumbtip1'])
        thumbtip2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Thumbtip2'])
        thumbbase1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Thumbbase1'])
        thumbbase2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Thumbbase2'])
        thumbbase_angle = round(calculate_vector_angle(thumbbase1_xy, thumbbase2_xy), 5)
        thumbtipbase_angle = round(calculate_vector_angle(thumbbase2_xy, thumbtip1_xy), 5)
        thumbtip_angle = round(calculate_vector_angle(thumbtip1_xy, thumbtip2_xy), 5)
        thumbtipendbaseend_angle = round(calculate_vector_angle(thumbbase1_xy, thumbtip2_xy), 5)
        thumbtipendbasehead_angle = round(calculate_vector_angle(thumbbase2_xy, thumbtip2_xy), 5)

        self.finger_angles_dict['Thumbbase'] = thumbbase_angle
        self.finger_angles_dict['Thumbtipbase'] = thumbtipbase_angle
        self.finger_angles_dict['Thumbtip'] = thumbtip_angle
        self.finger_angles_dict['Thumbtipendbaseend'] = thumbtipendbaseend_angle
        self.finger_angles_dict['Thumbtipendbasehead'] = thumbtipendbasehead_angle

        return thumbbase_angle, thumbtipbase_angle, thumbtip_angle, thumbtipendbaseend_angle, thumbtipendbasehead_angle

    def get_thumb_vectors_list(self):
        # Get the angle for the first predict
        for idx, landmarks_dict in enumerate(self.landmarks_dict_list):
            i = str(idx)
            thumbtip1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Thumbtip1'])
            thumbtip2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Thumbtip2'])
            thumbbase1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Thumbbase1'])
            thumbbase2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Thumbbase2'])

            thumbbase_angle = round(calculate_vector_angle(thumbbase1_xy, thumbbase2_xy), 5)
            thumbtipbase_angle = round(calculate_vector_angle(thumbbase2_xy, thumbtip1_xy), 5)
            thumbtip_angle = round(calculate_vector_angle(thumbtip1_xy, thumbtip2_xy), 5)
            thumbtipendbaseend_angle = round(calculate_vector_angle(thumbbase1_xy, thumbtip2_xy), 5)
            thumbtipendbasehead_angle = round(calculate_vector_angle(thumbbase2_xy, thumbtip2_xy), 5)

            thumbbase_length = round(calculate_vector_length(thumbbase1_xy, thumbbase2_xy), 1)
            thumbtipbase_length = round(calculate_vector_length(thumbbase2_xy, thumbtip1_xy), 1)
            thumbtip_length = round(calculate_vector_length(thumbtip1_xy, thumbtip2_xy), 1)
            thumbtipendbaseend_length = round(calculate_vector_length(thumbbase1_xy, thumbtip2_xy), 1)
            thumbtipendbasehead_length = round(calculate_vector_length(thumbbase2_xy, thumbtip2_xy), 1)

            '''self.finger_angles_dict['Thumbbase' + i] = thumbbase_angle
            self.finger_angles_dict['Thumbtipbase' + i] = thumbtipbase_angle
            self.finger_angles_dict['Thumbtip' + i] = thumbtip_angle
            self.finger_angles_dict['Thumbtipendbaseend' + i] = thumbtipendbaseend_angle
            self.finger_angles_dict['Thumbtipendbasehead' + i] = thumbtipendbasehead_angle
            l.append([thumbbase_angle, thumbtipbase_angle, thumbtip_angle,
                      thumbtipendbaseend_angle, thumbtipendbasehead_angle])'''


            self.finger_vectors_dict['Thumbbase' + i] = {'length': thumbbase_length, 'angle': thumbbase_angle}
            self.finger_vectors_dict['Thumbtipbase' + i] = {'length': thumbtipbase_length, 'angle': thumbtipbase_angle}
            self.finger_vectors_dict['Thumbtip' + i] = {'length': thumbtip_length, 'angle': thumbtip_angle}
            self.finger_vectors_dict['Thumbtipendbaseend' + i] = {'length': thumbtipendbaseend_length,
                                                                  'angle': thumbtipendbaseend_angle}
            self.finger_vectors_dict['Thumbtipendbasehead' + i] = {'length': thumbtipendbasehead_length,
                                                                   'angle': thumbtipendbasehead_angle}


    def get_index_vectors(self):
        # Get the angle for the first predict
        tip1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Indextip1'])
        tip2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Indextip2'])
        base1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Indexbase1'])
        base2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Indexbase2'])
        base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
        tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
        tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
        tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
        tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

        # Update the predictor's prediction attributes
        self.finger_angles_dict['Indexbase'] = base_angle
        self.finger_angles_dict['Indextipbase'] = tipbase_angle
        self.finger_angles_dict['Indextip'] = tip_angle
        self.finger_angles_dict['Indextipendbaseend'] = tipendbaseend_angle
        self.finger_angles_dict['Indextipendbasehead'] = tipendbasehead_angle

        return base_angle, tipbase_angle, tip_angle, tipendbaseend_angle, tipendbasehead_angle

    def get_index_vectors_list(self):
        index_dict_list = []
        for idx, landmarks_dict in enumerate(self.landmarks_dict_list):
            i = str(idx)
            # Get the angle for the first predict
            tip1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Indextip1'])
            tip2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Indextip2'])
            base1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Indexbase1'])
            base2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Indexbase2'])
            base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
            tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
            tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
            tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
            tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

            base_length = round(calculate_vector_length(base1_xy, base2_xy), 1)
            tipbase_length = round(calculate_vector_length(base2_xy, tip1_xy), 1)
            tip_length = round(calculate_vector_length(tip1_xy, tip2_xy), 1)
            tipendbaseend_length = round(calculate_vector_length(base1_xy, tip2_xy), 1)
            tipendbasehead_length = round(calculate_vector_length(base2_xy, tip2_xy), 1)


            '''# Update the predictor's prediction attributes
            self.finger_angles_dict['Indexbase' + i] = base_angle
            self.finger_angles_dict['Indextipbase' + i] = tipbase_angle
            self.finger_angles_dict['Indextip' + i] = tip_angle
            self.finger_angles_dict['Indextipendbaseend' + i] = tipendbaseend_angle
            self.finger_angles_dict['Indextipendbasehead' + i] = tipendbasehead_angle'''

            self.finger_vectors_dict['Indexbase' + i] = {'length': base_length, 'angle': base_angle}
            self.finger_vectors_dict['Indextipbase' + i] = {'length': tipbase_length, 'angle': tipbase_angle}
            self.finger_vectors_dict['Indextip' + i] = {'length': tip_length, 'angle': tip_angle}
            self.finger_vectors_dict['Indextipendbaseend' + i] = {'length': tipendbaseend_length, 'angle': tipendbaseend_angle}
            self.finger_vectors_dict['Indextipendbasehead' + i] = {'length': tipendbasehead_length, 'angle': tipendbasehead_angle}

    def get_middle_vectors(self):
        # Get the angle for the first predict
        tip1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Middletip1'])
        tip2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Middletip2'])
        base1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Middlebase1'])
        base2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Middlebase2'])
        base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
        tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
        tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
        tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
        tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

        # Update the predictor's prediction attributes
        self.finger_angles_dict['Middlebase'] = base_angle
        self.finger_angles_dict['Middletipbase'] = tipbase_angle
        self.finger_angles_dict['Middletip'] = tip_angle
        self.finger_angles_dict['Middletipendbaseend'] = tipendbaseend_angle
        self.finger_angles_dict['Middletipendbasehead'] = tipendbasehead_angle

        return base_angle, tipbase_angle, tip_angle, tipendbaseend_angle, tipendbasehead_angle

    def get_middle_vectors_list(self):
        l = []
        for idx, landmarks_dict in enumerate(self.landmarks_dict_list):
            i = str(idx)
            # Get the angle for the first predict
            tip1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Middletip1'])
            tip2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Middletip2'])
            base1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Middlebase1'])
            base2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Middlebase2'])

            base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
            tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
            tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
            tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
            tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

            base_length = round(calculate_vector_length(base1_xy, base2_xy), 1)
            tipbase_length = round(calculate_vector_length(base2_xy, tip1_xy), 1)
            tip_length = round(calculate_vector_length(tip1_xy, tip2_xy), 1)
            tipendbaseend_length = round(calculate_vector_length(base1_xy, tip2_xy), 1)
            tipendbasehead_length = round(calculate_vector_length(base2_xy, tip2_xy), 1)

            '''# Update the predictor's prediction attributes
            self.finger_angles_dict['Middlebase' + i] = base_angle
            self.finger_angles_dict['Middletipbase' + i] = tipbase_angle
            self.finger_angles_dict['Middletip' + i] = tip_angle
            self.finger_angles_dict['Middletipendbaseend' + i] = tipendbaseend_angle
            self.finger_angles_dict['Middletipendbasehead' + i] = tipendbasehead_angle'''

            self.finger_vectors_dict['Middlebase' + i] = {'length': base_length, 'angle': base_angle}
            self.finger_vectors_dict['Middletipbase' + i] = {'length': tipbase_length, 'angle': tipbase_angle}
            self.finger_vectors_dict['Middletip' + i] = {'length': tip_length, 'angle': tip_angle}
            self.finger_vectors_dict['Middletipendbaseend' + i] = {'length': tipendbaseend_length,
                                                                   'angle': tipendbaseend_angle}
            self.finger_vectors_dict['Middletipendbasehead' + i] = {'length': tipendbasehead_length,
                                                                    'angle': tipendbasehead_angle}

    def get_ring_vectors(self):
        # Get the angle for the first predict
        tip1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Ringtip1'])
        tip2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Ringtip2'])
        base1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Ringbase1'])
        base2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Ringbase2'])
        base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
        tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
        tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
        tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
        tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)


        # Update the predictor's prediction attributes
        self.finger_angles_dict['Ringbase'] = base_angle
        self.finger_angles_dict['Ringtipbase'] = tipbase_angle
        self.finger_angles_dict['Ringtip'] = tip_angle
        self.finger_angles_dict['Ringtipendbaseend'] = tipendbaseend_angle
        self.finger_angles_dict['Ringtipendbasehead'] = tipendbasehead_angle

        return base_angle, tipbase_angle, tip_angle, tipendbaseend_angle, tipendbasehead_angle

    def get_ring_vectors_list(self):
        l = []
        for idx, landmarks_dict in enumerate(self.landmarks_dict_list):
            i = str(idx)
            # Get the angle for the first predict
            tip1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Ringtip1'])
            tip2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Ringtip2'])
            base1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Ringbase1'])
            base2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Ringbase2'])
            base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
            tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
            tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
            tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
            tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

            base_length = round(calculate_vector_length(base1_xy, base2_xy), 1)
            tipbase_length = round(calculate_vector_length(base2_xy, tip1_xy), 1)
            tip_length = round(calculate_vector_length(tip1_xy, tip2_xy), 1)
            tipendbaseend_length = round(calculate_vector_length(base1_xy, tip2_xy), 1)
            tipendbasehead_length = round(calculate_vector_length(base2_xy, tip2_xy), 1)

            # Update the predictor's prediction attributes
            '''self.finger_angles_dict['Ringbase' + i] = base_angle
            self.finger_angles_dict['Ringtipbase' + i] = tipbase_angle
            self.finger_angles_dict['Ringtip' + i] = tip_angle
            self.finger_angles_dict['Ringtipendbaseend' + i] = tipendbaseend_angle
            self.finger_angles_dict['Ringtipendbasehead' + i] = tipendbasehead_angle'''

            self.finger_vectors_dict['Ringbase' + i] = {'length': base_length, 'angle': base_angle}
            self.finger_vectors_dict['Ringtipbase' + i] = {'length': tipbase_length, 'angle': tipbase_angle}
            self.finger_vectors_dict['Ringtip' + i] = {'length': tip_length, 'angle': tip_angle}
            self.finger_vectors_dict['Ringtipendbaseend' + i] = {'length': tipendbaseend_length,
                                                                   'angle': tipendbaseend_angle}
            self.finger_vectors_dict['Ringtipendbasehead' + i] = {'length': tipendbasehead_length,
                                                                    'angle': tipendbasehead_angle}

    def get_pinky_vectors(self):
        # Get the angle for the first predict
        tip1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Pinkytip1'])
        tip2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Pinkytip2'])
        base1_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Pinkybase1'])
        base2_xy = convert_landmark_point_to_xy_point(self.landmarks_dict['Pinkybase2'])
        base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
        tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
        tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
        tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
        tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

        # Update the predictor's prediction attributes
        self.finger_angles_dict['Pinkybase'] = base_angle
        self.finger_angles_dict['Pinkytipbase'] = tipbase_angle
        self.finger_angles_dict['Pinkytip'] = tip_angle
        self.finger_angles_dict['Pinkytipendbaseend'] = tipendbaseend_angle
        self.finger_angles_dict['Pinkytipendbasehead'] = tipendbasehead_angle

        return base_angle, tipbase_angle, tip_angle, tipendbaseend_angle, tipendbasehead_angle

    def get_pinky_vectors_list(self):
        l = []
        for idx, landmarks_dict in enumerate(self.landmarks_dict_list):
            i = str(idx)
            # Get the angle for the first predict
            tip1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Pinkytip1'])
            tip2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Pinkytip2'])
            base1_xy = convert_landmark_point_to_xy_point(landmarks_dict['Pinkybase1'])
            base2_xy = convert_landmark_point_to_xy_point(landmarks_dict['Pinkybase2'])
            base_angle = round(calculate_vector_angle(base1_xy, base2_xy), 5)
            tipbase_angle = round(calculate_vector_angle(base2_xy, tip1_xy), 5)
            tip_angle = round(calculate_vector_angle(tip1_xy, tip2_xy), 5)
            tipendbaseend_angle = round(calculate_vector_angle(base1_xy, tip2_xy), 5)
            tipendbasehead_angle = round(calculate_vector_angle(base2_xy, tip2_xy), 5)

            base_length = round(calculate_vector_length(base1_xy, base2_xy), 1)
            tipbase_length = round(calculate_vector_length(base2_xy, tip1_xy), 1)
            tip_length = round(calculate_vector_length(tip1_xy, tip2_xy), 1)
            tipendbaseend_length = round(calculate_vector_length(base1_xy, tip2_xy), 1)
            tipendbasehead_length = round(calculate_vector_length(base2_xy, tip2_xy), 1)

            # Update the predictor's prediction attributes
            '''self.finger_angles_dict['Pinkybase' + i] = base_angle
            self.finger_angles_dict['Pinkytipbase' + i] = tipbase_angle
            self.finger_angles_dict['Pinkytip' + i] = tip_angle
            self.finger_angles_dict['Pinkytipendbaseend' + i] = tipendbaseend_angle
            self.finger_angles_dict['Pinkytipendbasehead' + i] = tipendbasehead_angle'''

            self.finger_vectors_dict['Ringbase' + i] = {'length': base_length, 'angle': base_angle}
            self.finger_vectors_dict['Ringtipbase' + i] = {'length': tipbase_length, 'angle': tipbase_angle}
            self.finger_vectors_dict['Ringtip' + i] = {'length': tip_length, 'angle': tip_angle}
            self.finger_vectors_dict['Ringtipendbaseend' + i] = {'length': tipendbaseend_length,
                                                                 'angle': tipendbaseend_angle}
            self.finger_vectors_dict['Ringtipendbasehead' + i] = {'length': tipendbasehead_length,
                                                                  'angle': tipendbasehead_angle}

    '''def get_all_vectors(self):
        thumb = self.get_thumb_vectors()
        index = self.get_index_vectors()
        middle = self.get_middle_vectors()
        ring = self.get_ring_vectors()
        pinky = self.get_pinky_vectors()
        return thumb, index, middle, ring, pinky'''

    def get_all_vectors_list(self):
        thumb = self.get_thumb_vectors_list()
        index = self.get_index_vectors_list()
        middle = self.get_middle_vectors_list()
        ring = self.get_ring_vectors_list()
        pinky = self.get_pinky_vectors_list()
        foi_tip_dict = self.foi_tip_point_dict
        return thumb, index, middle, ring, pinky, foi_tip_dict

    def draw_foi_vectors_text(self, frame: np.ndarray):
        # Only used for single FOI
        cv2.putText(frame, self.finger_of_interest[0] + "tip angle" + str(self.finger_vectors_dict[self.finger_of_interest[0] + 'tip0']['angle']), (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        cv2.putText(frame, self.finger_of_interest[0] + "tipbaseangle" + str(self.finger_vectors_dict[self.finger_of_interest[0] + 'tipbase0']['angle']), (10, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        cv2.putText(frame, self.finger_of_interest[0] + "base angle" + str(self.finger_vectors_dict[self.finger_of_interest[0] + 'base0']['angle']), (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        cv2.putText(frame, self.finger_of_interest[0] + " tipend, baseend angle" + str(self.finger_vectors_dict[self.finger_of_interest[0] + 'tipendbaseend0']['angle']), (10, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        cv2.putText(frame, self.finger_of_interest[0] + " tipend, basehead angle" + str(self.finger_vectors_dict[self.finger_of_interest[0] + 'tipendbasehead0']['angle']), (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        return frame

    def get_and_update_landmarks_dict(self, r: list, pax_box, use_pax_box_offset=True) -> dict:
        personframe_bounding_box = pax_box
        personframe_min_x = personframe_bounding_box.origin_x
        personframe_min_y = personframe_bounding_box.origin_y
        '''if len(r) <= 2:
            total_hand_in_finger_reader = 0
            for pred in r:
                thumb_tip_point = calibrate_personFramePoint_to_roiFramePoint(np.round(np.array(pred['keypoints'][4]), 1),
                                                                              personframe_bounding_box)
                if is_inside_finger_reader_space(thumb_tip_point):
                    result_dict = pred
                    total_hand_in_finger_reader += 1
            if not total_hand_in_finger_reader == 1:
                print("FAILED TO DETECT ANY HANDS! NO HANDS PRESENTED TO FINGER READER!", total_hand_in_finger_reader)
                return {}, False
        else:
            # print("error the person has more than 2 hands????")'''
        '''if len(r) == 1:
            result_dict = r[-1]
            # Coco Wholebody hand conversion
            landmarks_dict = {}
            for index_keys in COCO_LABELS.keys():
                landmarks_dict[COCO_LABELS[index_keys]] = (
                int(round(personframe_min_x + result_dict['keypoints'][index_keys][0])),
                int(round(personframe_min_y + result_dict['keypoints'][index_keys][1])))
            self.landmarks_dict = landmarks_dict
            self.landmarks_dict_list = [landmarks_dict]
            return landmarks_dict, True
        elif len(r) == 2:'''
        for result_dict in r:
            landmarks_dict = {}
            for index_keys in COCO_LABELS.keys():
                if use_pax_box_offset:
                    landmarks_dict[COCO_LABELS[index_keys]] = (
                        int(round(personframe_min_x + result_dict['keypoints'][index_keys][0])),
                        int(round(personframe_min_y + result_dict['keypoints'][index_keys][1])))
                else:
                    landmarks_dict[COCO_LABELS[index_keys]] = (
                        int(round(result_dict['keypoints'][index_keys][0])),
                        int(round(result_dict['keypoints'][index_keys][1])))
                landmarks_dict[COCO_LABELS[index_keys] + " keypoint_scores"] = result_dict['keypoint_scores'][index_keys]

            self.landmarks_dict_list.append(landmarks_dict)
            # print("current landmark size", len(self.landmarks_dict_list))
            if len(r) == 1:
                self.landmarks_dict = landmarks_dict
        return self.landmarks_dict_list, True

    def draw_status_message(self, frame:np.ndarray)-> np.ndarray:
        #if not self.status_updated:
        message = self.status_message[0] + " Mode:" + str(self.hand_of_interest) + str(self.finger_of_interest)
        color = self.status_message[1]
        rf = cv2.putText(frame, message, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1)
        #self.status_updated = True
        return rf

    def draw_fingers_vectors_list(self, frame: np.ndarray):
        for landmarks_dict in self.landmarks_dict_list:
            frame = self.draw_fingers_vectors(frame, landmarks_dict)
        return frame

    '''def draw_fingers_vectors(self, frame: np.ndarray, landmarks_dict = {} )->np.ndarray:
        # print("Drawing successful vectors") # May be redundant method..
        landmarks_dict = self.landmarks_dict if landmarks_dict == {} else landmarks_dict
        # FOI Drawing
        FOI = self.finger_of_interest
        fingers_to_draw = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for finger in fingers_to_draw:
            if finger in FOI:
                # draw base1 to tip2 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'], landmarks_dict[finger + 'tip2'],
                                        (255, 255, 255), 2)

                # draw base2 to tip2 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'], landmarks_dict[finger + 'tip2'],
                                        (255, 255, 255), 2)

                # draw tip1 to tip2 thumb as green
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'tip1'], landmarks_dict[finger + 'tip2'], (0, 255, 0),
                                        2)

                # draw base2 to tip1 as green
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'], landmarks_dict[finger + 'tip1'], (0, 255, 0),
                                        2)

                # draw base1 to base2 as green
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'], landmarks_dict[finger + 'base2'],
                                        (0, 255, 0), 2)
            else:
                # draw base1 to tip2 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'], landmarks_dict[finger + 'tip2'],
                                        (255, 255, 255), 2)

                # draw base2 to tip2 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger+ 'base2'], landmarks_dict[finger + 'tip2'],
                                        (255, 255, 255), 2)

                # draw tip1 to tip2 thumb as green
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'tip1'], landmarks_dict[finger + 'tip2'],
                                        (0, 255, 0),
                                        2)

                # draw base2 to tip1 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'], landmarks_dict[finger + 'tip1'],
                                        (255, 255, 255),
                                        2)

                # draw base1 to base2 as white
                frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'], landmarks_dict[finger + 'base2'],
                                        (255, 255, 255), 2)
        return frame'''


    def draw_fingers_vectors(self, frame: np.ndarray):
        # FOI and non FOI Drawing
        FOI = self.finger_of_interest
        fingers_to_draw = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        if len(self.landmarks_dict_list) == 2:
            for landmarks_dict in self.landmarks_dict_list:
                for finger in fingers_to_draw:
                    finger_in_box = is_inside_finger_reader_space(landmarks_dict[finger + 'tip2'], cam_index=self.cam_index)
                    if finger in FOI:
                        # draw base1 to tip2 as white
                        tip_end_color = (0, 255, 0) if finger_in_box else (0, 0, 255)
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'],
                                                landmarks_dict[finger + 'tip2'],
                                                (255, 255, 255), 2)

                        # draw base2 to tip2 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'],
                                                landmarks_dict[finger + 'tip2'],
                                                (255, 255, 255), 2)

                        # draw tip1 to tip2 thumb as green
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'tip1'],
                                                landmarks_dict[finger + 'tip2'], tip_end_color,
                                                2)

                        # draw base2 to tip1 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'],
                                                landmarks_dict[finger + 'tip1'], tip_end_color,
                                                2)

                        # draw base1 to base2 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'],
                                                landmarks_dict[finger + 'base2'],
                                                tip_end_color, 2)
                    else:

                        tip_end_color = (0, 0, 255) if finger_in_box else (0, 255, 0)

                        # draw base1 to tip2 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'],
                                                landmarks_dict[finger + 'tip2'],
                                                (255, 255, 255), 2)

                        # draw base2 to tip2 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'],
                                                landmarks_dict[finger + 'tip2'],
                                                (255, 255, 255), 2)

                        # draw tip1 to tip2 thumb as red
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'tip1'],
                                                landmarks_dict[finger + 'tip2'],
                                                tip_end_color,
                                                2)

                        # draw base2 to tip1 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base2'],
                                                landmarks_dict[finger + 'tip1'],
                                                (255, 255, 255),
                                                2)

                        # draw base1 to base2 as white
                        frame = cv2.arrowedLine(frame, landmarks_dict[finger + 'base1'],
                                                landmarks_dict[finger + 'base2'],
                                                (255, 255, 255), 2)
        else:
            for finger in fingers_to_draw:
                fingertip2_score = self.landmarks_dict[finger + 'tip2 keypoint_scores']
                fingertip2_threshold = self.keypoint_score_threshold[COCO_LABELS_REVERSE[finger + 'tip2']]
                if self.cam_index == 2 and fingertip2_score < fingertip2_threshold:
                    finger_in_box = False # Cannot be determined as we dont have good confidence to trace the nonfoi pose of this finger!
                else:
                    finger_in_box = is_inside_finger_reader_space(self.landmarks_dict[finger + 'tip2'], cam_index=self.cam_index)
                if finger in FOI:
                    # draw base1 to tip2 as white
                    tip_end_color = (0, 255, 0) if finger_in_box else (0, 0, 255)
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base1'], self.landmarks_dict[finger + 'tip2'],
                                            (255, 255, 255), 2)

                    # draw base2 to tip2 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base2'], self.landmarks_dict[finger + 'tip2'],
                                            (255, 255, 255), 2)

                    # draw tip1 to tip2 thumb as green
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'tip1'], self.landmarks_dict[finger + 'tip2'], tip_end_color,
                                            2)

                    # draw base2 to tip1 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base2'], self.landmarks_dict[finger + 'tip1'], tip_end_color,
                                            2)

                    # draw base1 to base2 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base1'], self.landmarks_dict[finger + 'base2'],
                                            tip_end_color, 2)
                else:
                    tip_end_color = (0, 0, 255) if finger_in_box else (0, 255, 0)

                    # draw base1 to tip2 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base1'], self.landmarks_dict[finger + 'tip2'],
                                            (255, 255, 255), 2)

                    # draw base2 to tip2 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger+ 'base2'], self.landmarks_dict[finger + 'tip2'],
                                            (255, 255, 255), 2)

                    # draw tip1 to tip2 thumb as red
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'tip1'], self.landmarks_dict[finger + 'tip2'],
                                            tip_end_color,
                                            2)

                    # draw base2 to tip1 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base2'], self.landmarks_dict[finger + 'tip1'],
                                            (255, 255, 255),
                                            2)

                    # draw base1 to base2 as white
                    frame = cv2.arrowedLine(frame, self.landmarks_dict[finger + 'base1'], self.landmarks_dict[finger + 'base2'],
                                            (255, 255, 255), 2)
        return frame


    def clear_cache(self):
        self.landmarks_dict_list = []
        self.landmarks_dict = {}
        self.finger_angles_dict = {}
        self.foi_tip_point_dict = {}
        # self.status_message = ("Reset. Starting new inference..", (0, 255, 255))
        self.status_updated = False
