from __future__ import print_function
import cv2
import numpy as np
import sys
import termios
import os
import random
import select

from time import sleep, time
from imutils.video import VideoStream

from car.car import Car
from arduino import Arduino

class Main:
    def __init__(self, argv):
        self.arduino = Arduino()
        self.car = Car(self.arduino)
        self.vs = VideoStream(src=0).start()
        self.initKey()
        self.poller = select.epoll()
        self.poller.register(sys.stdin.fileno(), select.EPOLLIN)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.termKey()
        cv2.destroyAllWindows()
        self.car.stop()
        #self.car.model.save_dqn()
        #self.car.model.memory.save()
        #self.video.release()

    def initKey(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON) # lflags
        new_settings[6][termios.VMIN] = 0  # cc
        new_settings[6][termios.VTIME] = 0 # cc
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

    def termKey(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def getKey(self):
        if self.poller.poll(timeout=0):
            print('Success')
            ch_set = []
            ch = os.read(sys.stdin.fileno(), 1)
            while ch != None and len(ch) > 0:
              ch_set.append(ch[0])
              ch = os.read(sys.stdin.fileno(), 1)
            command = None
            arrow = ['KEY_UP', 'KEY_DOWN', 'KEY_RIGHT', 'KEY_LEFT']
            i = 0
            while i < len(ch_set):
                if ch_set[i] == 27 and ch_set[i+1] == 91:
                    command = arrow[ch_set[i+2]-65]
                    i += 3
                else:
                    command = chr(ch_set[i])
                    i += 1
            return command
        else:
            return None

    def usage(self):
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


with Main(sys.argv) as main:
    """
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    main.video = cv2.VideoWriter('test.mp4',fourcc,20,IMG_SIZE)
    """
    nowimg = main.vs.read()
    nowimg = cv2.resize(nowimg, (100, 100))
    start = time()
    while True:
        end = time()
        start = end
        preimg = nowimg
        nowimg = main.vs.read()
        cv2.imshow('test', nowimg)
        if cv2.waitKey(50)  == ord('q'):
            break
        nowimg = cv2.resize(nowimg, (100, 100))
        # main.video.write(img)
        key = main.getKey()
        if key == 'KEY_UP':
            main.car.action(0)
            """
            motion_feature = main.car.motion.GetFeature(preimg, nowimg)
            main.car.model.push([preimg, nowimg, motion_feature, 0])
            stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            """
        elif key == 'KEY_DOWN':
            main.car.action(1)
            """
            motion_feature = main.car.motion.GetFeature(preimg, nowimg)
            main.car.model.push([preimg, nowimg, motion_feature, 1])
            stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            """
        elif key == 'KEY_LEFT':
            main.car.action(2)
            """
            motion_feature = main.car.motion.GetFeature(preimg, nowimg)
            main.car.model.push([preimg, nowimg, motion_feature, 2])
            stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            """
        elif key == 'KEY_RIGHT':
            main.car.action(3)
            """
            motion_feature = main.car.motion.GetFeature(preimg, nowimg)
            main.car.model.push([preimg, nowimg, motion_feature, 3])
            stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            """
        elif key == ' ':
            main.car.action(4)
            """
            motion_feature = main.car.motion.GetFeature(preimg, nowimg)
            main.car.model.push([preimg, nowimg, motion_feature, 4])
            stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
            """
        elif key == 'q':
            break
        elif key == None:
            # ir = main.arduino.request('i\n')
            """
            prob = main.car.model.predict([preimg, nowimg, main.car.motion.GetFeature(preimg, nowimg)])[0]
            stdscr.addstr('[Predict] ' + str(prob) + '\n')
            main.car.action(np.argmax(prob))
            """
            pass

