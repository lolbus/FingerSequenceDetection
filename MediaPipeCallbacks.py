import mediapipe as mp
DetectionResult = mp.tasks.components.containers.DetectionResult
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
class predictionHandler():
    def __init__(self):
        self.person_predictions = [DetectionResult(detections=[]) for _ in range(5)]
        self.hand_predictions = [HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[]) for _ in range(5)]
        self.hand_landmarks_predictions = [HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[]) for _ in range(5)]

persondetector_predictionHandler = predictionHandler()

handdetector_predictionHandler = predictionHandler()
def persondetector_print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    persondetector_predictionHandler.person_predictions = persondetector_predictionHandler.person_predictions[1:] # delete older inference to keep memory in check
    #print("appending: ", result)
    persondetector_predictionHandler.person_predictions.append(result)

def handdetector_print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    handdetector_predictionHandler.hand_predictions = handdetector_predictionHandler.hand_predictions[1:] # delete older inference to keep memory in check
    handdetector_predictionHandler.hand_predictions.append(result)