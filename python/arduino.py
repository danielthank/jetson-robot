from __future__ import print_function
# from threading import Thread
# from Queue import Queue
from time import sleep
import serial
import os
import numpy as np

class Arduino():
    def __init__(self):
        pass
        """
        candidates = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/cu.usbmodem14231']
        for candidate in candidates:
            try:
                self.dev = serial.Serial(candidate)
            except:
                pass
            else:
                break
        # print(self.dev)
        sleep(3)
        self.dev.flushInput()
        print("[Arduino] ready")
        """

    def write(self, command):
        pass
        """
        self.dev.write(command)
        """

    def read(self, cnt):
        return None
        """
        return self.dev.read(cnt)
        """

    def readline(self):
        return None
        """
        return self.dev.readline()
        """

    def request(self, command):
        return None
        """
        self.write(command)
        ret = self.readline()
        print('[Arduino] ' +  command[:-1] + ' ' + ret[:-1])
        return ret
        """

    def terminate(self):
        return
        """
        self.write('s 0,0\n')
        """
