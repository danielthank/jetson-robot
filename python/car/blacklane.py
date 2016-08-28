import cv2
import numpy as np
import math
from scipy import weave 
class BlackLaneDetector :
    def __init__(self):
        self.angle = 0;
        self.state = "STOP";
        """ 
            state value : STOP, FORWARD, ROTATE, BACKWARD
        
        """
        self.PI = 3.14159;
        self.houghVote = 50;
        self.houghMinLen = 40;
        self.houghMaxGap = 10;

    def detect(self, frame, showImg, showInfo) :
        frame = cv2.resize(frame, (640, 480))
        roi = frame[frame.shape[0]/6 : frame.shape[0]*5/6 , 0 : frame.shape[1]]
        #blur = cv2.blur(frame, (3 , 3))
        gray = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 65)
        adapt = cv2.dilate(adapt, kernel, iterations = 3)

        zhang = self.ZhangAlgorithm(adapt, 50)
        zhang = cv2.dilate(zhang, kernel, iterations = 1)

        lines=cv2.HoughLinesP(zhang, 1, self.PI/180, self.houghVote, self.houghMinLen, self.houghMaxGap)

        result = self.findDirection(roi.shape, lines, showInfo)

        if showImg:
            cv2.imshow("original", frame)
            cv2.imshow("adapt", adapt)
            cv2.imshow("Zhang", zhang)
            cv2.imshow("result", result)
    
    def ZhangAlgorithm(self, src, iteration):
        dst = src.copy()/255
        prev = np.zeros(src.shape,np.uint8)
        diff = None
        for n in range(iteration) :
            dst = self.thinning(dst, 0)
            dst = self.thinning(dst, 1)
            diff = np.absolute(dst - prev)
            prev = dst.copy()
            if np.sum(diff) == 0 :
                break
        return dst*255

    def thinning(self, image, mode) :
        I,M = image, np.zeros(image.shape, np.uint8)
        expr = """
        for(int i=1;i < NI[0]-1;++i){
            for(int j=1;j < NI[1]-1;++j){
                int p2 = I2(i-1,j);
                int p3 = I2(i-1,j+1);
                int p4 = I2(i,j+1);
                int p5 = I2(i+1,j+1);
                int p6 = I2(i+1,j);
                int p7 = I2(i+1,j-1);
                int p8 = I2(i,j-1);
                int p9 = I2(i-1,j-1);

                int A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = mode == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = mode == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0) 
                    M2(i,j) = 1;
            }
        }
        """

        weave.inline(expr, ["I","mode","M"])
        return (I & ~M)

    def findDirection(self, size, lines, showInfo) :
        posLines = list()
        negLines = list()
        result = np.zeros(size, np.uint8)
        ### divide lines into two groups by its slope
        if lines != None :
            for line in lines:
                m, n = self.getLine(line[0])
                ### exclude vertical and horizontal lines
                if abs(m) > 0.1 and abs(m) < 100000 :
                    cv2.line(result,(line[0][0],line[0][1]),(line[0][2],line[0][3]),255,8)
                    i = 0
                    ### positive slope
                    if m > 0 :
                        for i in range(len(posLines)) :
                            if self.maxY(line[0]) > self.maxY(posLines[i]) :
                                posLines.insert(i,line[0])
                                break
                        if len(posLines) == 0 or i == len(posLines)-1 :
                            posLines.append(line[0])
                    ### negative slope
                    elif m < 0 :
                        for i in range(len(negLines)) :
                            if self.maxY(line[0]) > self.maxY(negLines[i]) :
                                negLines.insert(i,line[0])
                                break
                        if len(negLines) == 0 or i == len(negLines)-1 :
                            negLines.append(line[0])
        if showInfo:
            print "positive lines : ", len(posLines)
            print "negative lines : ", len(negLines)

        ### Case 1 : 
        changed = False
        if len(posLines) != 0 and len(negLines) != 0 :
            while len(posLines) > 0 and len(negLines) > 0 :
                ### find cross point of two lines
                x,y = self.crossPoint(posLines[0], negLines[0])
                if showInfo :
                    print "(x, y) : (",x," , ",y,")"
                if y > min(self.minY(posLines[0]), self.minY(negLines[0])) :
                    if self.maxY(posLines[0]) < self.maxY(negLines[0]) :
                        posLines.pop(0)
                    else :
                        negLines.pop(0)
                else :
                    cv2.line(result,(posLines[0][0],posLines[0][1]),(posLines[0][2],posLines[0][3]),128,8)
                    cv2.line(result,(negLines[0][0],negLines[0][1]),(negLines[0][2],negLines[0][3]),128,8)
                    
                    m, n = self.getLine(posLines[0])    
                    result = self.drawLine(result, m, n)
                    m, n = self.getLine(negLines[0])
                    result = self.drawLine(result, m, n)
                    
                    self.angle=self.computeThetaByXY(result.shape[1], result.shape[0], x, y)
                    result = self.drawDirection(result, self.angle, showInfo)
                    
                    self.state = "FORWARD"
                    changed = True
                    break
        ### Case 2 :
        if not changed and len(posLines) != 0 :
            self.findDirectionByOneGroup(result, posLines, showInfo)
            changed=True
        if not changed and len(negLines) != 0 :
            self.findDirectionByOneGroup(result, negLines, showInfo)
            changed=True
        ### Case 3 :
        if not changed :
            self.state = "STOP"
            changed = True

        return result

    def findDirectionByOneGroup(self, result, lines, showInfo) :
        twoLine=False
        for i in range(1,len(lines)) :
            m1, n1 = self.getLine(lines[0])
            m2, n2 = self.getLine(lines[i])
            theta = math.fabs(math.atan(m1)-math.atan(m2))*180/self.PI
            theta = min(theta,180-theta)
            if theta > 20 :
                if showInfo :
                    print "angle between two line : ", theta
                result = self.drawLine(result, m1, n1)
                result = self.drawLine(result, m2, n2)
                
                twoLine = True
                break
        
        if twoLine :
            self.state = "FORWARD"
            result = self.drawDirection(result,0,showInfo)
        elif self.state == "FORWARD" :
            self.state = "STOP" 
        else :
            m, n =self.getLine(lines[0])
            result = self.drawLine(result,m,n)
            self.angle = self.computeThetaByM(m)

            y = result.shape[0]
            x = (y-n)/m
            if (m > 0 and x < result.shape[1]/2 ) or (m < 0 and x > result.shape[1]/2) :
                self.state = "FORWARD"
                result = self.drawDirection(result,0,showInfo)
            else :
                self.state = "ROTATE"
                result = self.drawDirection(result,self.angle,showInfo)
            

    def getLine(self, points):
        a1, b1 = float(points[0]), float(points[1])
        a2, b2 = float(points[2]), float(points[3])
        if a1-a2 == 0:
            return 100000, 100000
        m = (b1-b2)/(a1-a2)
        n = (a1*b2 - a2*b1)/(a1-a2)
        
        return m, n

    def getPoints(self, cols, rows, m, n) :
        x1, y1 = 0, int(n)
        x2, y2 = int(cols), int(m*cols + n)
        x3, y3 = int(- n/m), 0
        x4, y4 = int((rows-n)/m), rows
                 
        p = [None] * 4
        if y1 < 0:
            p[0]=x2
            p[1]=y2
        else :
            p[0]=x1
            p[1]=y1
        if x3 < 0 :
            p[2]=x4
            p[3]=y4
        else :
            p[2]=x3
            p[3]=y3
        return p

    def crossPoint(self, l1, l2) :
        m1, n1 = self.getLine(l1)
        m2, n2 = self.getLine(l2)
        if m1 == m2 :
            return 100000,100000
        x = (n2 - n1)/(m1 - m2)
        y = (m1*n2 - m2*n1)/(m1 - m2)
        return x, y


    def drawLine(self, dst, m, n) :
        p = self.getPoints(dst.shape[1], dst.shape[0], m, n)
        cv2.line(dst, (p[0],p[1]),(p[2],p[3]),128,2)
        return dst

    def drawDirection(self, dst, angle, showInfo) :
        oX = dst.shape[1]/2
        oY = dst.shape[0]
        r = 100
        if showInfo :
            print "Direction in degree : ",self.angle
        cv2.line(dst, (oX,oY), (int(oX+r*math.sin(angle*self.PI/180)),int(oY-r*math.cos(angle*self.PI/180))), 200, 2)
        return dst

    def computeThetaByXY(self, cols, rows, x, y) :
        oX=cols/2
        oY=rows
        cosine=(oY-y)/math.sqrt((x-oX)**2 + (y-oY)**2)
        theta=math.acos(cosine)*180/self.PI
        if x < oX:
            theta*=-1
        return theta

    def computeThetaByM(self,m) :
        theta = math.atan(m)*180/self.PI
        if theta < 0 :
            theta = 90 +theta
        else :
            theta = theta - 90
        return theta

    def minY(self, points) :
        return min(points[1], points[3])
    
    def maxY(self, points) :
        return max(points[1], points[3])


import sys
import time
def Usage() :
    print "Usage : python blackLane.py <cv> <source>"
    print "    c : read data from camera"
    print "    v : read data from video" 
    sys.exit()
if __name__ == "__main__" :
    print "Testing the function of class BlackLane"
    if len(sys.argv) != 3 :
        Usage()
    
    if sys.argv[1] == 'c' :
        cap = cv2.VideoCapture(int(sys.argv[2]))
    elif sys.argv[1] == 'v' :
        cap = cv2.VideoCapture(sys.argv[2])
    else :
        Usage()
        sys.exit()
    
    detector = BlackLaneDetector()
    
    key = None
    while key != ord('q') :
        while cap.isOpened() :
            ret, frame = cap.read()
            if ret :
                detector.detect(frame, True, True)
                key = cv2.waitKey(100) & 0xFF 
                if key == ord('q') :
                    break
            else :
                break
            if key == ord('p') :
                print "press \"g\" to continue"
                while cv2.waitKey(1) & 0xFF != ord('g') :
                    pass
        if sys.argv[1] == 'v' and key != ord('q') :
            cap.open(sys.argv[2])
