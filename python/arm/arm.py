from time import sleep
import numpy as np

class Arm():
    def __init__(self, arduino):
        self.arduino = arduino
        self.l1 = 850
        self.l2 = 770

    def degree2Arduino(self, degrees):
        command = 'a '
        for i, degree in enumerate(degrees):
            if degree < 0:
                degree = 0
            elif degree > 180:
                degree = 180
            command += str(int(degree)) + ','
        command = command[:-1] + '\n'
        return self.arduino.request(bytearray(command, 'ascii'))

    def laser2Arduino(self, dev, flag):
        command = 'l ' + str(int(dev)) + ',' + str(int(flag)) + '\n'
        return self.arduino.request(bytearray(command, 'ascii'))

    def rad2deg(self, rad):
        return rad * 180 / np.pi

    def gotoRZ(self, r, z):
        # print(r, z)
        degrees = [90] * 6
        d_square = r**2 + z**2
        d = np.sqrt(d_square)
        # print((self.l1 ** 2 + d_square - self.l2 ** 2) / (2 * self.l1 * d))
        # print((self.l1 ** 2 + self.l2 ** 2 - d_square) / (2 * self.l1 * d))
        degrees[1] = self.rad2deg(np.arctan2(z, r) + np.arccos((self.l1 ** 2 + d_square - self.l2 ** 2) / (2 * self.l1 * d)))
        degrees[2] = self.rad2deg(np.arccos((self.l1 ** 2 + self.l2 ** 2 - d_square) / (2 * self.l1 * self.l2))) - 90
        degrees[3] = 90 - degrees[1] - degrees[2]
        if degrees != self.now:
            self.now = degrees
