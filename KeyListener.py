import numpy as np
import pynput
from pynput import keyboard
import cv2
class KeyListener:
    def __init__(self):
        self.record_key_pressed = False
        self.listener = keyboard.Listener(on_press=self.on_press)

    def on_press(self, key):
        try:
            if key == pynput.keyboard.Key.f1:
                self.record_key_pressed = not self.record_key_pressed
                print(f'Key pressed: {self.record_key_pressed}')
        except AttributeError:
            pass

    def start_listener(self):
        self.listener.start()

    def draw_key_status_message(self, frame:np.ndarray)-> np.ndarray:
        message = "Record activated *" if self.record_key_pressed else "Recording not-activated"
        color = (0, 0, 255) if self.record_key_pressed else (0, 255, 255)
        rf = cv2.putText(frame, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return rf