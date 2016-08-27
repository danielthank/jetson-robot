import cv2
import numpy as np

class BlackLaneDetector:
	def _init_(self, camera):
		self.PI = 3.14159
		self.angle = 0;
		self.houghVote = 50;
		self.houghMinLen = 40;
		self.houghMaxGap = 10;
	def detect(self, frame, showImg, showInfo):
		frame = cv2.resize(frame, (320, 180))
		roi = frame[0 : frame.shape(0)/6 , frame.shape(1) : frame.shape(0)*5/6]
		#blur = cv2.blur(frame, (3 , 3))
		gray = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
		
		kernel = np.ones((3,3), np.uint8)
		adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 65)
		adapt = cv2.dilate(adapt, kernel, iterations = 3)
		
		zhang = self.ZhangAlgorithm(adapt, 50)
		zhang = cv2.dilate(zhang, kernel, iterations = 1)

		lines=cv2.HoughLineP(zhang, 1, self.PI/180, self.houghVote, self.houghMinLen, self.houghMaxGap)

		result = self.findDirection(roi.shape, lines, showInfo)
		
		if showImg:
			cv2.imshow("original", frame)
			cv2.imshow("adapt", adapt)
			cv2.imshow("Zhang", zhang)
			cv2.imshow("result", result)
		

	def ZhangAlgorithm(self, src, iteration) :
		dst = src.copy()/255
		prev = np.zeros(src.shape(:2),np.unit8)
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
		I,M = im, np.zeros(image.shape, np.uint8)
		expr = """
		for(int i=1;i < MI[0]-1;++i){
			for(int j=1;j < NI[1]-1;++j){
				int p2 = I2(i-1,j)
				int p3 = I2(i-1,j+1)
				int p4 = I2(i,j+1)
				int p5 = I2(i+1,j+1)
				int p6 = I2(i+1,j)
				int p7 = I2(i+1,j-1)
				int p8 = I2(i,j-1)
				int p9 = I2(i-1,j-1)

				int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
						(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
						(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
						(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
				int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
				int m1 = mode == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
				int m2 = mode == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

				if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) 
					M2(i,j) = 1;
			}
		}
		"""

		weave.inline(expr, ["I","mode","M"])
		return (I & ~M)

	def findDirection(self, size, lines, showInfo) :
		posLines = list()
		negLines = list()
		result = np.zeros(size, uint8)
		### divide lines into two groups by its slope
		for line in lines
			m, n = self.getLine(line)
			### exclude vertical and horizontal lines
			if abs(m) > 0.1 && abs(m) < 100000 :
				cv2.line(result,Point(line[0],line[1]),Point(line[2],line[3]),255,8)	
				i = 0
				### positive slope
				if m > 0 :	
					for i in range(len(posLines)) :
						if maxY(line) > maxY(posLines[i]) :
							posLines.insert(i,line)
							break
					if len(posLines) == 0 || i == len(posLines)-1 :
						posLines.append(line)
				### negative slope
				elif m < 0 :
					for i in range(len(negLines)) :
						if maxy(line) > maxY(negLines[i]) :
							negLines.insert(i,line)
							break
					if len(negLines) == 0 || i == len(negLines)-1 :
						negLines.append(line)
		if showInfo :
			print "positive lines : ", len(posLines)
			print "negative lines : ", len(negLines)
		
		### Case 1 : 
		changed = False
		if len(posLines) != 0 && len(negLines) != 0 :
			while len(posLines) > 0 && len(negLines) > 0 :
				### find cross point of two lines
				x,y = self.crossPoint(posLines[0], negLines[0])
				if showInfo :
					print "(x, y) : (",x," , ",y,")"
				if y > min(minY(posLines[0]), minY(negLines[0])) :
					if maxY(posLines[0]) < maxY(negLines[0]) :
						posLines.pop(0)
					else :
						negLines.pop(0)
				else :
					m, n = getLine(posLines[0])
					result = self.drawLine(result, m, n)
					m, n = getLine(negLines[0])
					result = self.drawLine(result, m, n)
					
					self.angle=self.computeTheta(result.shape(1), result.shape(0), x, y)
					result = self.drawDirection(result, showInfo)
					
					changed = True
					break
		### Case 2 :
		if !changed && len(posLines) != 0 :
			pass
		if !changed && len(negLines) != 0 :
			pass
		return result

	def getLine(self, points):
		a1 = points[0], b1 = points[1]
		a2 = points[2], b2 = points[3]
		if a1-a2 == 0
			return 100000, 100000

		m = (b1-b2)/(a1-a2)
		n = (a1*b2 - a2*b1)/(a1-a2)
		return m, n

	def getPoints(self, cols, rows, m, n) :
		x1 = 0, y1 = n
		x2 = cols, y2 = m*cols + n
		y3 = 0, x3 = - n/m
		y4 = rows, x4 = (rows-n)/m
		p = [None] * 4
		if y1 < 0 : 
			p[0]=x2
			p[1]=y2
		else :
			p[0]=x1
			p[1]=y1
		 
		if x3 < 0 :
			p[2]=x4
			p[3]=y4
		else ：
			p[2]=x3
			p[3]=y3
		return p;

	def crossPoint(self, l1, l2) :
		m1, n1 = getLine(l1)
		m2, n2 = getLine(l2)
		x = (n2 - n1)/(m1 - m2)
		y = (m1*n2 - m2*n1)/(m1 - m2)
		return x, y


	def drawLine(self, dst, m, n) :
		p = self.getPoints(dst.shape(1), dst.shape(0), m, n)
		line(dst, Point(p[0],p[1]),Point(p[2],p[3]),128,2)
		return dst

	def drawDirection(self, dst, showInfo) :
		oX = dst.shape(1)/2, oY = dst,shape(0)
		r = 100
		if showInfo :
			print "Direction in degree : ",self.angle
		line(dst, Point(oX,oY), Point(oX+r*sin(self.angle*self.PI/180),oY-r*cos(self.angle*self.PI/180)), 200, 2)
		return dst

	def computeTheta(self, cols, rows, x, y) :
		oX=cols/2,oY=rows;
		cosine=(oY-y)/sqrt(pow(x-oX,2)+pow(y-oY,2)); 
		theta=acos(cosine)*180/self.PI;
		if(x < oX)
			theta*=-1;
		return theta;
		

	def minY(self, points) :
		return min(points[1], points[3])
	def maxY(self, points) :
		return max(points[1], points[3])
