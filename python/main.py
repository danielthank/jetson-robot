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
                    main.car.model.push(preimg, 0)
                    main.car.model.push(nowimg, 0)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_DOWN':
                    main.car.model.push(preimg, 1)
                    main.car.model.push(nowimg, 1)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_LEFT':
                    main.car.model.push(preimg, 2)
                    main.car.model.push(nowimg, 2)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == 'KEY_RIGHT':
                    main.car.model.push(preimg, 3)
                    main.car.model.push(nowimg, 3)
                    stdscr.addstr('[Train] ' + str(main.car.model.train()) + '\n')
                elif key == ' ':
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
            while True:
                end = time()
                stdscr.addstr('[Time] ' + str(end-start) + '\n')
                start = end
                try:
                    key = stdscr.getkey()
                except:
                    key = None
                if key == 'q':
                    break

                stdscr.refresh()
                preimg = nowimg
                nowimg = main.vs.read()

                rand = random.random()
                if rand < main.car.model.epsilon:
                    action = random.randint(0, 4)
                    stdscr.addstr('[Action(Rand)] ' + str(action) + '\n')
                else:
                    ret = main.car.model.predict([preimg, nowimg])
                    action = np.argmax(ret[0])
                    stdscr.addstr('[Predict] ' + str(ret) + '\n')
                    stdscr.addstr('[Action(Predict)] ' + str(action) + '\n')

                ir = main.arduino.request('i\n')
                reward = 0
                if ir != '00':
                    reward += -10
                if action == 0:
                    reward += 1
                stdscr.addstr('[Reward] ' + str(reward) + '\n')

                main.car.model.push(nowimg, action, reward, False)
                ret = main.car.model.train()
                if ret != None:
                    stdscr.addstr('[Train] ' + str(ret) + '\n')

curses.wrapper(curses_main)
