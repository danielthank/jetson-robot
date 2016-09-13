from blacklane import BlackLaneDetector
from find_motion import FindMotion
from dqn.model import DQN

class Car:
    def __init__(self, arduino):
        self.ROTATE_SPEED = 10
        self.BASE_SPEED = 10
        self.MAX_SPEED = 90
        self.arduino = arduino
        self.lastAngle = 0
        self.p = 0.5
        self.i = 0
        self.d = 0
        self.accum = 0
        self.rSpeed = 0
        self.lSpeed = 0
        self.gamma = 0.5
        self.detector = BlackLaneDetector()
        self.motion = FindMotion()
        self.model = DQN(camera_shape=(3, 100, 100), motion_shape=self.motion.GetFeatureShape())

    def action(self, idx) :
        funcs = [self.forward, self.backward, self.left, self.right, self.stop]
        if idx < len(funcs):
            return funcs[idx]()
        return 'fail'

    def setSpeed(self, r, l):
        self.rSpeed = r
        self.lSpeed = l
        return self.toArduino()

    def forward(self):
        return self.setSpeed(self.BASE_SPEED, self.BASE_SPEED)

    def backward(self):
        return self.setSpeed(-self.BASE_SPEED, -self.BASE_SPEED)

    def left(self):
        return self.setSpeed(self.BASE_SPEED, -self.BASE_SPEED)

    def right(self):
        return self.setSpeed(-self.BASE_SPEED, self.BASE_SPEED)

    def stop(self):
        return self.setSpeed(0, 0)

    def toArduino(self):
        if self.rSpeed < -self.MAX_SPEED:
            self.rSpeed = -self.MAX_SPEED
        elif self.rSpeed > self.MAX_SPEED:
            self.rSpeed = self.MAX_SPEED
        if self.lSpeed < -self.MAX_SPEED:
            self.lSpeed = -self.MAX_SPEED
        elif self.lSpeed > self.MAX_SPEED:
            self.lSpeed = self.MAX_SPEED
        command = 's ' + str(int(self.rSpeed)) + ',' + str(int(self.lSpeed)) + '\n'
        return self.arduino.request(command)
