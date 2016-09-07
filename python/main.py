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
from arduino import Arduino

class Main:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.arduino = Arduino()
        self.car = Car(self.arduino)
        self.vs = VideoStream(src=0).start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()
        self.ttydefault()
        self.car.stop()
        self.car.model.save_dqn()

    def ttyraw(self):
        tty.setraw(self.fd)

    def ttydefault(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def Usage(self):
        print('Usage : BlackLaneDetector <cvp> <source>')
        print('    c : read data from camera')
        print('    v : read data from video')
        print('    p : read data from picture')
        sys.exit()
        
def key2action(key):
    if key  == '\x1b[A':
        return 0
    elif key == '\x1b[B':
        return 1
    elif key == '\x1b[C':
        return 2
    elif key == '\x1b[D':
        return 3
    elif key == ' ':
        return 4
    else:
        return 5

def redirect_stderr(self,flag):
    if flag:
        null_dev = os.open('/dev/null', os.O_WRONLY)
        os.dup2(null_dev, 2)

with Main() as main:
    main.ttyraw()
    ir = main.arduino.request('i\n')
    img = main.vs.read()
    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 1)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b' or ch == ' ':
                if ch == '\x1b':
                    ch = ch + sys.stdin.read(2)
                idx = key2action(ch)
                main.car.action(idx)
                main.ttydefault()
                main.car.model.push(img, idx)
                print('[Train] ' + str(main.car.model.train()))
                main.ttyraw()
            elif ch == 'q':
                sys.exit()
        else:
            img = main.vs.read()
            ir = main.arduino.request('i\n')
            main.ttydefault()
            prob = main.car.model.predict(img)[0]
            print('[Predict] ' + str(prob))
            main.car.action(np.argmax(prob))
            main.ttyraw()
            """
            cv2.imshow('test', img)
            if cv2.waitKey(100)  == ord('q'):
                break
            """


