class Car:
    def __init__(self, arduino):
        self.ROTATE_SPEED = 20
        self.BASE_SPEED = 20
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

    def action(self, angle, state) :
        print(angle, state)
        if state == "FORWARD" :
            return self.forward(angle)
        elif state == "ROTATE" :
            return self.rotate(angle)
        elif state == "STOP " :
            return self.stop()
        elif state == "BACKWARD" :
            return 0

    def forward(self, angle):
        self.accum += angle
        self.rSpeed = self.BASE_SPEED - self.p * angle - self.i * self.accum - self.d * (angle-self.lastAngle)
        self.lSpeed = self.BASE_SPEED + self.p * angle + self.i * self.accum + self.d * (angle-self.lastAngle)
        return self.toArduino()

    def setSpeed(self, right, left):
        self.rSpeed = right
        self.lSpeed = left
        """
        self.rSpeed += self.gamma * (right - self.rSpeed)
        self.lSpeed += self.gamma * (left - self.lSpeed)
        """
        return self.toArduino()

    def rotate(self, angle):
        self.rSpeed = self.ROTATE_SPEED
        self.lSpeed = self.ROTATE_SPEED
        if angle > 0:
            self.rSpeed *= -1
        elif angle < 0:
            self.lSpeed *= -1
        else:
            self.rSpeed, self.lSpeed = self.BASE_SPEED, self.BASE_SPEED
        return self.toArduino()

    def stop(self):
        self.rSpeed = 0
        self.lSpeed = 0
        return self.toArduino()

    def toArduino(self):
        if self.rSpeed < -self.MAX_SPEED:
            self.rSpeed = -self.MAX_SPEED
        elif self.rSpeed > self.MAX_SPEED:
            self.rSpeed = self.MAX_SPEED
        if self.lSpeed < -self.MAX_SPEED:
            self.lSpeed = -self.MAX_SPEED
        elif self.lSpeed > self.MAX_SPEED:
            self.lSpeed = sellf.MAX_SPEED
        command = 's ' + str(int(self.rSpeed)) + ',' + str(int(self.lSpeed)) + '\n'
        return self.arduino.request(command)
        """
        if self.arduino.available() == True:
            self.arduino.push(command)
        """
