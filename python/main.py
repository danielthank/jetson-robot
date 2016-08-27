from __future__ import print_function
import cv2
import numpy as np
import tty
import sys
import select
import termios
from time import sleep
from car.car import Car
from car.blacklane import BlackLaneDetector
from arduino import Arduino

arduino = Arduino()
car = Car(arduino)
arduino.start()

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setraw(fd)


def terminate():
    arduino.terminate()
    arduino.join()
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sys.exit()

def Usage():
    print('Usage : BlackLaneDetector <cvp> <source>')
    print('    c : read data from camera')
    print('    v : read data from video')
    print('    p : read data from picture')
    terminate()

def key(com):
    # up
    if com == '\x1b[A':
        car.setSpeed(40, 40)
    # down
    elif com == '\x1b[B':
        car.setSpeed(-40, -40)
    # right
    elif com == '\x1b[C':
        car.setSpeed(-40, 40)
    # left
    elif com == '\x1b[D':
        car.setSpeed(40, -40)
        # print("left")
    elif com == ' ':
        car.stop()
    elif com == 'q':
        terminate()
    elif com == 'c':
        tty.setcbreak(sys.stdin.fileno())

if len(sys.argv) != 3:
	Usage()
if sys.argv[1]== 'c':
	dev = int(sys.argv[2])
	cap = cv2.VideoCapture(dev)

# detector = BlackLaneDetector()

while True:
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch = ch + sys.stdin.read(2)
        key(ch)
    else:
        arduino.write('i\n')
<<<<<<< Updated upstream
        ir = arduino.read(3)
        print(ir[:2])
=======
        ir = arduino.read(2)
>>>>>>> Stashed changes
        """
        ret, frame = cap.read()
        # detector.detect(frame, True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """

cap.release()
cv2.destroyAllWindows()
terminate()

"""
elif sys.argv[1] == 'v':
	cap = cv2.VideoCapture(sys.argv[2])
elif sys.argv[1] == 'p':
    print 'Not yet implement'
    terminate()
else:
	Usage()
"""
