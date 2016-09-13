import numpy as np
import cv2
import sys
import timeit
import itertools
import math
from scipy import weave
import random

class FindMotion:
    WINDOW = (5, 5)       # (h/2, w/2)
    BLOCK = (10, 10)        # (h, w)
    MAX_MAG = (WINDOW[0]**2 + WINDOW[1]**2)**0.5
    IMG_SIZE = (100, 100)
    def __init__(self):
        self.yy = xrange(self.WINDOW[0], self.IMG_SIZE[1]-self.BLOCK[0]-self.WINDOW[0]+1, self.BLOCK[0])
        self.xx = xrange(self.WINDOW[1], self.IMG_SIZE[0]-self.BLOCK[1]-self.WINDOW[1]+1, self.BLOCK[1])
        self.feature = np.empty((2, len(self.yy), len(self.xx)))

        self.img_t0 = None
        self.img_t1 = None
        self.win = None
        self.roi = None
        self.show = None
        self.trace = np.zeros((self.WINDOW[0]+1,self.WINDOW[1]+1))
        self.search_iter = [[self.WINDOW[0],self.WINDOW[1]]]
        self.all_cost = np.zeros((11,11),dtype = 'int')

    def ExhaustiveSearch(self):
        cen = np.array(self.WINDOW)
        min_cost = np.inf
        min_len = np.inf
        cord = None
        ref_iter = itertools.product(xrange(0, self.WINDOW[0]*2+1,2),
                                     xrange(0, self.WINDOW[1]*2+1,2))

        for y, x in ref_iter:
            Len = (self.WINDOW[0] - y)**2 + (self.WINDOW[1] - x)**2
            ref = self.win[y:y+self.BLOCK[0],x:x+self.BLOCK[1]]
            cost = np.sum(np.abs(ref - self.roi))
            self.all_cost[y/2,x/2] = cost
            if cost < min_cost or (cost == min_cost and Len < min_len):
                cord = np.array([y, x])
                min_cost = cost
                min_len = Len
        return (cord - cen)[::-1], min_cost

    def DepthFirstSearch(self) :
        cen = np.array(self.WINDOW)
        self.trace.fill(-1)
        min_cost = np.inf
        cord = None
        #self.search_iter = [[random.randrange(0,21,2),random.randrange(0,21,2)]]
        for y,x in self.search_iter :
            cost ,cord_t =self.DFS(x,y)
            if cost < min_cost :
                min_cost = cost
                cord = cord_t
            elif cost == min_cost :
                break
        return (cord - cen)[::-1],min_cost

    def DFS(self,x,y) :
        ref = self.win[y:y+self.BLOCK[0],x:x+self.BLOCK[1]]
        min_cost = np.sum(np.abs(ref - self.roi))
        min_len = (self.WINDOW[0] - y)**2 + (self.WINDOW[1] - x)**2
        cord = np.array([y,x])
        self.trace[y/2,x/2] = min_cost
        update = True
        while update :
            ref_iter = [[y,x-2],[y-2,x],[y,x+2],[y+2,x]]
            update = False
            for ty,tx in ref_iter :
                if ty >= 0 and ty <2*self.WINDOW[0]+1 and tx >= 0 and tx < 2*self.WINDOW[1]+1 :
                    if self.trace[ty/2,tx/2] == -1 :
                        ref = self.win[ty:ty+self.BLOCK[0],tx:tx+self.BLOCK[1]]
                        cost = np.sum(np.abs(ref - self.roi))
                        self.trace[ty/2,tx/2] = cost
                    else :
                        cost = self.trace[ty/2,tx/2]
                    Len = (self.WINDOW[0] - ty)**2 + (self.WINDOW[1] - tx)**2
                    if cost < min_cost or (cost == min_cost and Len < min_len):
                        min_cost = cost
                        cord = np.array([ty,tx])
                        min_len = Len
                        x,y = tx,ty
                        update = True

        return min_cost,cord

    def GetFeatureShape():
        self.feature.shape

    def BlockMatching(self):
        block_iter = itertools.product(),

        for y, block_y in enumerate(self.yy):
            for x, block_x in enumerate(self.xx):
                self.roi = np.array(self.img_t0[block_y:block_y+self.BLOCK[0], block_x:block_x+self.BLOCK[1]],dtype = 'float32')
                self.win = np.array(self.img_t1[block_y-self.WINDOW[0]:block_y+self.BLOCK[0]+self.WINDOW[0],block_x-self.WINDOW[1]:block_x+self.BLOCK[1]+self.WINDOW[1]], dtype = 'float32')
                #print "ex search : ",timeit.default_timer()-start
                #start = timeit.default_timer()
                motion2, min_cost2 = self.DepthFirstSearch()
                #print "df search : ",timeit.default_timer()-start
                motion_n = motion2 / self.MAX_MAG
                self.feature[: , y, x] = motion_n
        return self.feature

    def GetFeature(self, pre, now):
        print pre
        pre = cv2.resize(pre, self.IMG_SIZE)
        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        pre = cv2.blur(pre, (3, 3))

        now = cv2.resize(now, self.IMG_SIZE)
        now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
        now = cv2.blur(now, (3, 3))

        self.img_t0, self.img_t1 = pre, now
        return self.BlockMatching()

"""
if __name__ == '__main__':
    motion = FindMotion()
    cap = cv2.VideoCapture(0)
    ret, pre = cap.read()
    from time import sleep
    sleep(1)
    ret, now = cap.read()
    print motion.GetFeature(pre, now)
"""
