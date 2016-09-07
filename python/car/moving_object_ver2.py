import numpy as np
import cv2
import sys
import timeit
import itertools
#import tensorflow as tf
import theano.tensor as T
from theano import function, scan
import theano

WINDOW = (10, 10)       # (h/2, w/2)
BLOCK = (25, 25)        # (h, w)
MAG = BLOCK[0]/2
MAX_MAG = (WINDOW[0]**2 + WINDOW[1]**2)**0.5
IMG_SIZE = (320, 240)   # (w, h)
THRESHOLD = 10.

'''sess = tf.InteractiveSession()
a = tf.placeholder('float', shape=[21*21, 480, 640])
b = tf.placeholder('float', shape=[21*21, 480, 640])
c = tf.abs(b - a)'''
### theano variable
A = T.tensor3("A")
B = T.tensor3("B")
Diff = abs(A-B)

X = T.iscalar("X")
Y = T.iscalar("Y")
N = T.iscalar("N")
Matrix = T.tensor3("Matrix")
Submatrix = Matrix[:,Y:Y+BLOCK[0],X:X+BLOCK[1]]

Summation = T.sum(Matrix,axis=(1,2))

### theano function
get_diff= function([A, B], Diff)
get_submatrix = function([Matrix,X,Y],Submatrix)
get_sum = function([Matrix],Summation)

###+++++++++++++++++++++++++++++++++++++++++###
def get_motion(index,diff) :
    
    width = IMG_SIZE[0] / BLOCK[1]
    height = IMG_SIZE[1] / BLOCK[0]
    block_x = index % width * width
    block_y = index / width * height
    submatrix = diff[:,block_y:block_y+BLOCK[0],block_x:block_x+BLOCK[1]]
    #submatrix = get_submatrix(diff,block_x,block_y)
            
    Sum = T.sum(submatrix,axis=(1,2))
    #Sum = get_sum(submatrix)
    #Sum = np.sum(submatrix,axis=(1,2))
            
    motion_flat = np.argmin(Sum)
    return motion_flat

Motions ,Updates = scan(fn = get_motion,
                        non_sequences = Matrix,
                        sequences = T.arange(N)
                        )
get_motions = function([Matrix,N],Motions)


### class definition
class FindMotion:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        self.img_tm1 = None
        self.img_t = None
        #self.win_img = None
        self.roi = None

    '''def ExhaustiveSearch(self, win_y_b, win_x_b, win_y_e, win_x_e, roi_pos):
        #win_h = self.win_img.shape[0]
        #win_w = self.win_img.shape[1]
        win_h = win_y_e - win_y_b
        win_w = win_x_e - win_x_b
        assert win_h >= BLOCK[0] and win_w >= BLOCK[1]

        min_cost = np.inf
        cord = None
        ref_iter = itertools.product(xrange(win_h - BLOCK[0] + 1),
                                     xrange(win_w - BLOCK[1] + 1))
        for y, x in ref_iter:
            #ref = self.win_img[y:y+BLOCK[0], x:x+BLOCK[1]]
            ref = self.img_t[win_y_b+y:win_y_b+y+BLOCK[0], win_x_b+x:win_x_b+x+BLOCK[1]]
            cost = np.sum(np.abs(ref - self.roi))
            if cost < min_cost:
                cord = np.array([y, x])
                min_cost = cost
        return (cord - roi_pos)[::-1]'''

    def BlockMatching(self):
        pad_img_t = np.pad(np.array(self.img_t, dtype='float32'),
                           pad_width=([WINDOW[0]]*2, [WINDOW[1]]*2),
                           mode='constant',
                           constant_values=np.inf)
        print '1', timeit.default_timer()
        shift_iter = itertools.product(xrange(2*WINDOW[0]+1), xrange(2*WINDOW[1]+1))
        shift_list = map(lambda (y, x) : pad_img_t[y:y+IMG_SIZE[1], x:x+IMG_SIZE[0]], shift_iter)
        print '2', timeit.default_timer()
        stack_ref = np.stack(shift_list)
        print '3', timeit.default_timer()
        stack_target = np.stack( [self.img_tm1] * ((2*WINDOW[0]+1)*(2*WINDOW[1]+1)) )
        print '4', timeit.default_timer()
        diff = np.abs(stack_target - stack_ref)
        #diff = get_diff(stack_target, stack_ref)
        #diff = sess.run(c, feed_dict={b:stack_target, a:stack_ref})
        print '5', timeit.default_timer()
        
        """
        n=(IMG_SIZE[1]/BLOCK[0])*(IMG_SIZE[0]/BLOCK[1])
        motions_flat =  get_motions(diff,n)
        #motions = np.array([motions_flat / (2*WINDOW[0]+1), motions_flat % (2*WINDOW[1]+1)]) - [np.array(WINDOW)]
        """
        block_iter = itertools.product(xrange(0, IMG_SIZE[1]-BLOCK[0]+1, BLOCK[0]),
                                       xrange(0, IMG_SIZE[0]-BLOCK[1]+1, BLOCK[1]))
        for block_y, block_x in block_iter:
            #submatrix = get_submatrix(diff,block_x,block_y)
            submatrix = diff[:,block_y:block_y+BLOCK[0],block_x:block_x+BLOCK[1]]
            
            #Sum = get_sum(submatrix)
            Sum = np.sum(submatrix,axis=(1,2))
            
            motion_flat = np.argmin(Sum)
            #print motion_flat
            motion = np.array([motion_flat / (2*WINDOW[0]+1), motion_flat % (2*WINDOW[1]+1)]) - np.array(WINDOW)
            
            #print motion
            motion_n = (motion / MAX_MAG)[::-1]
            block_center = np.array([block_x+BLOCK[1]/2, block_y+BLOCK[0]/2])
            cv2.circle(self.img_t, tuple(block_center), 2, 0, -1)
            cv2.line(self.img_t, tuple(block_center), tuple(np.array(block_center+MAG*motion_n, dtype=np.int)), 255, 1)
            #print (block_y, block_x)
        print "6",timeit.default_timer()
    def run(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame, IMG_SIZE)
        self.img_tm1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while True:
            ret, frame = self.video.read()
            frame = cv2.resize(frame, IMG_SIZE)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.img_t = img_gray.copy()
            t_start = timeit.default_timer()
            print 'start', t_start
            self.BlockMatching()
            print timeit.default_timer()
            print "after full:", timeit.default_timer() - t_start
            cv2.imshow('frame', frame)
            cv2.imshow('gray_tm1', self.img_tm1)
            cv2.imshow('gray_t', self.img_t)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break
            self.img_tm1 = img_gray
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    finder = FindMotion('./Turn4.avi')
    finder.run()
