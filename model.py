import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from metadata import DatasetMeta
import numpy as np
import MediaPipeCallbacks as mpc
import cv2

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


class CCTV_person_modeler(object):
    def __init__(self, detector):
        self.detector = detector

    def persondetector_evaluate_frame(self, input, counter):
        self.detector.detect_async(input, counter)
        result = mpc.persondetector_predictionHandler.person_predictions
        return result


class CCTV_hand_modeler(object):
    def __init__(self, detector):
        self.detector = detector

    def handdetector_evaluate_frame(self, input, counter):
        self.detector.detect_async(input, counter)
        result = mpc.handdetector_predictionHandler.hand_predictions
        return result



class modelloader(object):
    def __init__(self, modelname="IVOD V1", threshold=0.1):
        self.metadata = DatasetMeta()
        models_dir = self.metadata.modelDir

        if modelname == "CCTV_MP_PERSON_PREDICTOR":
            options = vision.ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=models_dir + '/CCTVPredictor/efficientdet.tflite'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                max_results=1,
                category_allowlist=["person"],
                result_callback=mpc.persondetector_print_result,
                score_threshold=self.metadata.person_detector_threshold)
            self.detector = vision.ObjectDetector.create_from_options(options)
            self.CCTV_person_Handler = CCTV_person_modeler(self.detector)
            self.person_modeler = self.CCTV_person_Handler.persondetector_evaluate_frame


            self.modeler_positive_thresold = 0

        elif modelname == "CCTV_MP_HAND_PREDICTOR":
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=models_dir + '/CCTVPredictor/hand_landmarker.task'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=1,
                result_callback=mpc.handdetector_print_result,
                min_hand_detection_confidence=0.0001,
                min_hand_presence_confidence =0.5,
                min_tracking_confidence =0.1
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.CCTV_hand_Handler = CCTV_hand_modeler(self.detector)
            self.hand_modeler = self.CCTV_hand_Handler.handdetector_evaluate_frame

        elif modelname == "CCTV_MP_LRHAND_PREDICTOR":
            mpHands = mp.solutions.hands
            options = mpHands.Hands(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=threshold,
                min_tracking_confidence=threshold,
                max_num_hands=2)
            self.LRhand_modeler = options.process

    def visualize(self, image, detection_result) -> np.ndarray:
            """Draws bounding boxes on the input image and return it.
            Args:
              image: The input RGB image.
              detection_result: The list of all "Detection" entities to be visualize.
            Returns:
              Image with bounding boxes.
            """
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                 MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
            return image
