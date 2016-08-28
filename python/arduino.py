from __future__ import print_function
# from threading import Thread
# from Queue import Queue
from time import sleep
import serial
import os
import numpy as np

class Arduino():
    def __init__(self):
        # Thread.__init__(self)
        candidates = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/cu.usbmodem14231']
        for candidate in candidates:
            try:
                self.dev = serial.Serial(candidate)
            except:
                pass
            else:
                break
        # print(self.dev)
        # self.q = Queue(maxsize=10)
        # self.dev.setDTR(False)
        sleep(3)
        self.dev.flushInput()
        # self.sleeping = False
        print("[Arduino] ready")

    def write(self, command):
        self.dev.write(command)

    def read(self, cnt):
        return self.dev.read(cnt)

    def readline(self):
        return self.dev.readline()

    def request(self, command):
        self.write(command)
        ret = self.readline()
        print('[Arduino] ' +  command[:-1] + ' ' + ret[:-1])
        return ret

    def terminate(self):
        self.write('s 0,0\n')
        # self.write('q\n')
