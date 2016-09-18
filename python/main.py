import cv2
import numpy as np
import sys
import termios
import os
import random
import select
import argparse

from time import sleep, time

class Main:
    IMG_SIZE = (100, 100)
    def __init__(self, argv):
        parser = argparse.ArgumentParser(description='jetson-robot')
        parser.add_argument('-1', action='store_const', dest='level', const='car', default=None)
        parser.add_argument('-2', action='store_const', dest='level', const='arm', default=None)
        parser.add_argument('-t', action='store_const', dest='level', const='train', default=None)
        parser.add_argument('-s', action='store_true', dest='show', default=False)
        parser.add_argument('-nod', action='store_true', dest='nod', default=False)
        args = parser.parse_args()
        self.level = args.level
        self.show = args.show
        if args.nod:
            null_dev = os.open('/dev/null', os.O_WRONLY)
            os.dup2(null_dev, 2)

        self.initKey()
        self.poller = select.epoll()
        self.poller.register(sys.stdin.fileno(), select.EPOLLIN)

        if self.level == 'car':
            from arduino import Arduino
            self.arduino = Arduino()

            from car.car import Car
            self.car = Car(self.arduino)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video = cv2.VideoWriter(self.level +'.avi', fourcc, 20, self.IMG_SIZE)

            from imutils.video import VideoStream
            self.vs = VideoStream(src=0).start()

        elif self.level == 'arm':
            from arduino import Arduino
            self.arduino = Arduino()

            from arm.arm import Arm
            self.arm = Arm(self.arduino)

        print('[Start]')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()
        self.termKey()
        if self.level == 'car':
            self.car.stop()
            self.car.model.save_cnn()
            self.car.model.memory.save()
            self.vs.stop()
            self.video.release()
        elif self.level == 'arm':
            pass

    def initKey(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON) 
        new_settings[6][termios.VMIN] = 0
        new_settings[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)

    def termKey(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def getKey(self):
        if self.poller.poll(timeout=0):
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

with Main(sys.argv) as main:
    if main.level == 'car':
        nowimg = main.vs.read()
        nowimg = cv2.resize(nowimg, main.IMG_SIZE)
        start = time()
        while True:
            end = time()
            print('[Time] ' + str(end - start))
            start = end
            preimg = nowimg
            nowimg = main.vs.read()
            if main.show:
                cv2.imshow('test', nowimg)
                if cv2.waitKey(50)  == ord('q'):
                    break
            nowimg = cv2.resize(nowimg, main.IMG_SIZE)
            main.video.write(nowimg)
            key = main.getKey()
            if key == 'KEY_UP':
                main.car.setAction(0)
                motion_feature = main.car.motion.GetFeature(preimg, nowimg)
                main.car.model.push([preimg, nowimg, motion_feature, 0])
                # print('[Train] ' + str(main.car.model.train()))
            elif key == 'KEY_DOWN':
                main.car.setAction(1)
                motion_feature = main.car.motion.GetFeature(preimg, nowimg)
                main.car.model.push([preimg, nowimg, motion_feature, 1])
                # print('[Train] ' + str(main.car.model.train()))
            elif key == 'KEY_LEFT':
                main.car.setAction(2)
                motion_feature = main.car.motion.GetFeature(preimg, nowimg)
                main.car.model.push([preimg, nowimg, motion_feature, 2])
                # print('[Train] ' + str(main.car.model.train()))
            elif key == 'KEY_RIGHT':
                main.car.setAction(3)
                motion_feature = main.car.motion.GetFeature(preimg, nowimg)
                main.car.model.push([preimg, nowimg, motion_feature, 3])
                # print('[Train] ' + str(main.car.model.train()))
            elif key == ' ':
                main.car.setAction(4)
                motion_feature = main.car.motion.GetFeature(preimg, nowimg)
                main.car.model.push([preimg, nowimg, motion_feature, 4])
                # print('[Train] ' + str(main.car.model.train()))
            elif key == 'q':
                break
            elif key == None:
                # ir = main.arduino.request('i\n')
                prob = main.car.model.predict([preimg, nowimg, main.car.motion.GetFeature(preimg, nowimg)])[0]
                print('[Predict] ' + str(prob))
                # action = np.argmax(prob)
                main.car.setSmooth(prob)
    elif main.level == 'arm':
        print('arm')
    elif main.level == 'train':
        from car.find_motion import FindMotion
        motion = FindMotion(camera_shape=(3,) + main.IMG_SIZE)
        from car.cnn.model import CNN
        model = CNN(camera_shape=(3,) + main.IMG_SIZE, motion_shape=motion.GetFeatureShape(), batch_size=64)
        while True:
            key = main.getKey()
            if key == 'q':
                model.save_cnn()
                break
            print('[Train] ' + str(model.train()))
