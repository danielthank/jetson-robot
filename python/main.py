from __future__ import print_function
import cv2
import numpy as np
import tty
import sys
import select
import termios
from time import sleep
from imutils.video import VideoStream
from car.car import Car
from car.blacklane import BlackLaneDetector
from arduino import Arduino
from contextlib import contextmanager

@contextmanager
def redirect_stdout(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

arduino = Arduino()
car = Car(arduino)
# arduino.start()

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
    # up
    if com == '\x1b[A':
        car.setSpeed(40, 40)
    # down
    elif com == '\x1b[B':
        car.setSpeed(-40, -40)
    # right
    elif com == '\x1b[C':
        car.setSpeed(-40, 40)
    # left
    elif com == '\x1b[D':
        car.setSpeed(40, -40)
    elif com == ' ':
        car.stop()
    elif com == 'q':
        ttydefault()
        sys.exit()
    elif com == 'c':
        tty.setcbreak(sys.stdin.fileno())
"""
if len(sys.argv) != 3:
    Usage()
"""

vs = VideoStream(src=0).start()
if sys.argv[1] == '1':
    manual = True
else:
    manual = False

detector = BlackLaneDetector()

while True:
    if manual:
        ttyraw()
        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch = ch + sys.stdin.read(2)
            key(ch)
            ttydefault()
        else:
            ttydefault()
            ir = arduino.request('i\n')
            with redirect_stdout(sys.stderr):
                frame = vs.read()
            # angle, state = detector.detect(frame, True, False)
            # ret = car.action(angle, state)
            # print(ret, end='')
    else:
        ir = arduino.request('i\n')
        # print(ir[:-1])
        frame = vs.read()
        imshow('test', frame)
        if cv2.waitKey(100)  == ord('q'):
            break


ttydefault()
cap.release()
cv2.destroyAllWindows()

"""
elif sys.argv[1] == 'v':
        cap = cv2.VideoCapture(sys.argv[2])
elif sys.argv[1] == 'p':
    print 'Not yet implement'
    terminate()
else:
        Usage()
"""
