from __future__ import print_function
import cv2
import numpy as np
import sys
import select
import termios
import os
import curses

from time import sleep
from imutils.video import VideoStream

from car.car import Car
from arduino import Arduino

class Main:
    def __init__(self):
        self.arduino = Arduino()
        self.car = Car(self.arduino)
        self.vs = VideoStream(src=0).start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()
        self.car.stop()
        self.car.model.save_dqn()
        self.car.model.memory.save()

    def Usage(self):
        print('Usage : BlackLaneDetector <cvp> <source>')
        print('    c : read data from camera')
        print('    v : read data from video')
        print('    p : read data from picture')
        sys.exit()
        
def redirect_stderr(self,flag):
    if flag:
        null_dev = os.open('/dev/null', os.O_WRONLY)
        os.dup2(null_dev, 2)

IMG_SIZE = (640, 480)
def curses_main(stdscr):
    stdscr.nodelay(True)
    stdscr.scrollok(True)
    with Main() as main:
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out = cv2.VideoWriter('test.mp4',fourcc,20,IMG_SIZE)

        while True:
            img = main.vs.read()
            out.write(img)
            try:
                key = stdscr.getkey()
            except:
                key = None
            if key == 'KEY_UP':
                main.car.model.push(img, 0)
                stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            elif key == 'KEY_DOWN':
                main.car.model.push(img, 1)
                stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            elif key == 'KEY_LEFT':
                main.car.model.push(img, 2)
                stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            elif key == 'KEY_RIGHT':
                main.car.model.push(img, 3)
                stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            elif key == 'q':
                break
            elif key == None:
                # ir = main.arduino.request('i\n')
                prob = main.car.model.predict(img)[0]
                stdscr.addstr('[Predict] ' + str(prob) + '\n')
                main.car.action(np.argmax(prob))
                """
                cv2.imshow('test', img)
                if cv2.waitKey(50)  == ord('q'):
                    break
                """
        out.release()

curses.wrapper(curses_main)
