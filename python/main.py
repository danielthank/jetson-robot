from __future__ import print_function
import cv2
import numpy as np
import sys
import select
import termios
import os
import sys
import curses
import random

from time import sleep, time
from imutils.video import VideoStream

from car.car import Car
from arduino import Arduino

class Main:
    def __init__(self, argv):
        if argv[1] == '0':
            self.pre_training = False
        else:
            self.pre_training = True
            self.epsilon = 0.1
        self.arduino = Arduino()
        self.car = Car(self.arduino, self.pre_training, 2)
        self.vs = VideoStream(src=0).start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()
        self.car.stop()
        self.car.model.save_dqn()
        self.car.model.memory.save()
        # self.video.release()

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
    with Main(sys.argv) as main:
        """
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        main.video = cv2.VideoWriter('test.mp4',fourcc,20,IMG_SIZE)
        """
        if main.pre_training:
            nowimg = main.vs.read()
            start = time()
            while True:
                end = time()
                stdscr.addstr('[Time] ' + str(end-start) + '\n')
                start = end
                preimg = nowimg
                nowimg = main.vs.read()
                # main.video.write(img)
                try:
                    key = stdscr.getkey()
                except:
                    key = None
                if key == 'KEY_UP':
                    main.car.action(0)
                    main.car.model.push(preimg, 0)
                    main.car.model.push(nowimg, 0)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_DOWN':
                    main.car.action(1)
                    main.car.model.push(preimg, 1)
                    main.car.model.push(nowimg, 1)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_LEFT':
                    main.car.action(2)
                    main.car.model.push(preimg, 2)
                    main.car.model.push(nowimg, 2)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_RIGHT':
                    main.car.action(3)
                    main.car.model.push(preimg, 3)
                    main.car.model.push(nowimg, 3)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == ' ':
                    main.car.action(4)
                    main.car.model.push(preimg, 4)
                    main.car.model.push(nowimg, 4)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'q':
                    break
                elif key == None:
                    # ir = main.arduino.request('i\n')
                    prob = main.car.model.predict([preimg, nowimg])[0]
                    stdscr.addstr('[Predict] ' + str(prob) + '\n')
                    main.car.action(np.argmax(prob))
                    """
                    cv2.imshow('test', img)
                    if cv2.waitKey(50)  == ord('q'):
                        break
                    """
        else:
            nowimg = main.vs.read()
            start = time()
            terminal = False
            explore = True
            action = 4
            while True:
                stdscr.refresh()
                end = time()
                stdscr.addstr('[Time] ' + str(end-start) + '\n')
                start = end
                try:
                    key = stdscr.getkey()
                except:
                    key = None

                action = None
                if key == 'KEY_UP':
                    action = 0
                elif key == 'KEY_DOWN':
                    action = 1
                elif key == 'KEY_LEFT':
                    action = 2
                elif key == 'KEY_RIGHT':
                    action = 3
                elif key == ' ':
                    terminal = not terminal
                    if terminal:
                        main.car.action(4)
                elif key == 'q':
                    break

                if terminal:
                    continue

                preimg = nowimg
                nowimg = main.vs.read()

                if action == None:
                    ret = main.car.model.predict([preimg, nowimg])
                    action = np.argmax(ret[0])
                    stdscr.addstr('[Predict] ' + str(ret) + '\n')

                main.car.action(action)
                ir = main.arduino.request('i\n')
                reward = 0
                """
                if ir == '1\n':
                    reward += -10
                """
                if action == 0:
                    reward += 1
                elif action == 1 or action == 4:
                    reward -= 1
                stdscr.addstr('[Reward] ' + str(reward) + '\n')

                main.car.model.push(nowimg, action, reward, terminal)
                ret = main.car.model.train()
                if ret != None:
                    stdscr.addstr('[Train] ' + str(ret) + '\n')

curses.wrapper(curses_main)
