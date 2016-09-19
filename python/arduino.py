import serial
import numpy as np

from time import sleep

class Arduino():
    def __init__(self):
        candidates = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/cu.usbmodem14231']
        for candidate in candidates:
            try:
                self.dev = serial.Serial(candidate)
            except:
                continue
            else:
                break
        # print(self.dev)
        sleep(3)
        self.dev.flushInput()
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
        print('[Arduino] ' +  command.decode('ascii')[:-1] + ' ' + ret.decode('ascii')[:-1])
        return ret

    def terminate(self):
        self.write('s 0,0\n')
