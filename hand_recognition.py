import cv2
import mediapipe as mp
import numpy as np


class HandDetector():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(False, 2, 1, 0.75)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def get_position(self, img, hand_idx = 0, draw = True):
        pos_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_idx]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                pos_list.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
 
        return pos_list


def main():
    detector = HandDetector()
    capture = cv2.VideoCapture(0)

    while(True):
        ret, frame = capture.read()

        img_size = (int(frame.shape[1]/2), int(frame.shape[0]/2))
        frame = cv2.resize(frame, img_size)
        frame = cv2.flip( frame, 1 )

        hand_img = detector.detect( frame )
        positions = detector.get_position( frame )

        if len(positions):
            print(positions[8])

        cv2.imshow('hands',hand_img)
        if cv2.waitKey(1)!=-1:
            break

    capture.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()