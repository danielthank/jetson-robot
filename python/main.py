from __future__ import print_function
import cv2
import numpy as np
import tty
import sys
import select
import termios
import os
from time import sleep
from imutils.video import VideoStream
from car.car import Car
from car.blacklane import BlackLaneDetector
from arduino import Arduino
from model import DeepModel

null_dev = os.open('/dev/null', os.O_WRONLY)
os.dup2(null_dev, 2)

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

arduino = Arduino()
car = Car(arduino)

def ttyraw():
    tty.setraw(fd)

def ttydefault():
    termios.tcsetattr(fd, termios.TCSADRAIN, old)

def Usage():
    print('Usage : BlackLaneDetector <cvp> <source>')
    print('    c : read data from camera')
    print('    v : read data from video')
    print('    p : read data from picture')
    terminate()

def key(com):
    if com == '\x1b[A':
        car.action("FORWARD")
    elif com == '\x1b[B':
        car.action("BACKWARD")
    elif com == '\x1b[C':
        car.action("RIGHT")
    elif com == '\x1b[D':
        car.action("LEFT")
    elif com == ' ':
        car.stop()
    elif com == 'q':
        ttydefault()
        car.stop()
        sys.exit()

"""
if len(sys.argv) != 3:
    Usage()
"""

if sys.argv[1] == '1':
    manual = True
else:
    manual = False

detector = BlackLaneDetector()
model = DeepModel()
print(model)
vs = VideoStream(src=0).start()

if manual:
    while True:
        ttyraw()
        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch = ch + sys.stdin.read(2)
            ttydefault()
            key(ch)
        else:
            ttydefault()
            ir = arduino.request('i\n')
            frame = vs.read()
            # angle, state = detector.detect(frame, True, False)
            # ret = car.action(angle, state)
else:
    while True:
        ir = arduino.request('i\n')
        frame = vs.read()
        imshow('test', frame)
        if cv2.waitKey(100)  == ord('q'):
            break


ttydefault()
cap.release()
cv2.destroyAllWindows()
