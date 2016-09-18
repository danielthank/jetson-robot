import numpy as np

class Car:
    def __init__(self, arduino):
        self.arduino = arduino
        self.speeds = np.array([[50, 50], [-50, -50], [50, -50], [-50, 50], [-25, -25]])

        self.MAX_SPEED = 90

        from .find_motion import FindMotion
        self.motion = FindMotion(camera_shape=(3, 100, 100))
        from .cnn.model import CNN
        self.model = CNN(camera_shape=(3, 100, 100), motion_shape=self.motion.GetFeatureShape(), batch_size=64)

    def setAction(self, idx) :
        speed = self.speeds[idx]
        return self.toArduino(*speed)

    def setSmooth(self, weights):
        speed = np.dot(weights, self.speeds)
        return self.toArduino(*speed)

    def stop(self):
        return self.toArduino(0, 0)

    def toArduino(self, rSpeed, lSpeed):
        print(rSpeed, lSpeed)
        if rSpeed < -self.MAX_SPEED:
            rSpeed = -self.MAX_SPEED
        elif rSpeed > self.MAX_SPEED:
            rSpeed = self.MAX_SPEED
        if lSpeed < -self.MAX_SPEED:
            lSpeed = -self.MAX_SPEED
        elif lSpeed > self.MAX_SPEED:
            lSpeed = self.MAX_SPEED
        command = 'c ' + str(int(rSpeed)) + ',' + str(int(lSpeed)) + '\n'
        return self.arduino.request(bytearray(command, 'ascii'))
