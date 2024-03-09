import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np 
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import pygame
import argparse
from threading import Thread
import time

class DrowsinessDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x600')
        self.root.title('Drowsiness Detector')
        self.root.configure(background='#CDCDCD')
        

        self.label1 = Label(self.root, background='#CDCDCD', font=('arial',15,'bold'))
        self.sign_image = Label(self.root)

        self.upload_button = Button(self.root, text="Start Drowsiness Detection", command=self.start_detection, padx=10, pady=5)
        self.upload_button.configure(background="#364156", foreground="white", font=("arial", 20, "bold"))
        self.upload_button.pack(side='bottom', pady=50)
        

        self.sign_image.pack(side='bottom', expand='True')
        self.label1.pack(side='bottom', expand='True')
        self.heading = Label(self.root, text="Drowsiness Detector", pady=20, font=("arial", 25, "bold"))
        self.heading.configure(background="#CDCDCD", foreground="#364156")
        self.heading.pack()
        
    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    @staticmethod
    def sound_alarm(path):
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        time.sleep(1)  # Wait for 1 second
        pygame.mixer.music.stop()


    def start_detection(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-a", "--alarm", type=str, default="alarm.mp3",
            help="path alarm .WAV file")
        args = vars(ap.parse_args())
        
        thresh = 0.25
        frame_check = 20
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        cap = cv2.VideoCapture(0)
        flag = 0
        ALARM_ON = False 
        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < thresh:
                    flag += 1
                    print(flag)
                    if flag >= frame_check:
                        if not ALARM_ON:
                            ALARM_ON = True
                            if args["alarm"] != "":
                                t = Thread(target=self.sound_alarm, args=(args["alarm"],))
                                t.start()
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("Drowsy")



                else:
                    flag = 0
                    ALARM_ON = False

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DrowsinessDetector()
    app.run()

