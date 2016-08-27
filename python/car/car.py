class Car:
    def __init__(self, arduino):
        self.ROTATE_SPEED = 10
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

    def forward(self, angle):
        self.accum += angle
        self.rSpeed = self.BASE_SPEED - self.p * angle - self.i * self.accum - self.d * (angle-self.lastAngle)
        self.lSpeed = self.BASE_SPEED + self.p * angle + self.i * self.accum + self.d * (angle-self.lastAngle)
        self.toArduino()

    def setSpeed(self, right, left):
        self.rSpeed = right
        self.lSpeed = left
        """
        self.rSpeed += self.gamma * (right - self.rSpeed)
        self.lSpeed += self.gamma * (left - self.lSpeed)
        """
        self.toArduino()

    def rotate(self, angle):
        self.rSpeed = self.ROTATE_SPEED
        self.lSpeed = self.ROTATE_SPEED
        if angle > 0:
            self.rSpeed *= -1
        elif angle < 0:
            self.lSPeed *= -1
        else:
            self.rSpeed, self.lSpeed = self.BASE_SPEED, self.BASE_SPEED
        self.toArduino()

    def stop(self):
        self.rSpeed = 0
        self.lSpeed = 0
        self.toArduino()

    def toArduino(self):
        if self.rSpeed < -MAX_SPEED:
            self.rSpeed = -MAX_SPEED
        elif self.rSpeed > MAX_SPEED:
            self.rSpeed = MAX_SPEED
        if self.lSpeed < -MAX_SPEED:
            self.lSpeed = -MAX_SPEED
        elif self.lSpeed > MAX_SPEED:
            self.lSpeed = MAX_SPEED
        command = 's ' + str(int(self.rSpeed)) + ',' + str(int(self.lSpeed)) + '\n'
        if self.arduino.available() == True:
            self.arduino.push(command)
