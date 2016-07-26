import cv2
import numpy as np
import sys
import time
from matplotlib import pyplot as plt

def HOGofCells(HOGfeatures, img_size, block_size, cell_size, stride, nbins):
    nblock_y = (img_size[1] - block_size[1]) / stride[1] + 1
    nblock_x = (img_size[0] - block_size[0]) / stride[0] + 1
    ncell_y = block_size[1] / cell_size[1]
    ncell_x = block_size[0] / cell_size[0]
    ##features = HOGfeatures.reshape([nblock_y*ncell_y, nblock_x*ncell_x, nbins])
    cnt_cells = np.zeros([img_size[1]/cell_size[1], img_size[0]/cell_size[0]])
    hog_cells = np.zeros([img_size[1]/cell_size[1], img_size[0]/cell_size[0], nbins])

    feature_num = 0
    for x in xrange(nblock_x):
        for y in xrange(nblock_y):
            cell_start_x = x*stride[0]/cell_size[0]
            cell_start_y = y*stride[1]/cell_size[1]
            for c_x in xrange(ncell_x):
                for c_y in xrange(ncell_y):
                    for b in xrange(nbins):
                        hog_cells[cell_start_y+c_y][cell_start_x+c_x][b] += HOGfeatures[feature_num]
                        feature_num += 1
                    cnt_cells[cell_start_y+c_y][cell_start_x+c_x] += 1
    assert feature_num == HOGfeatures.shape[0]

    ## averaging cell gradient ##
    for y in xrange(img_size[1]/cell_size[1]):
        for x in xrange(img_size[0]/cell_size[0]):
            count = cnt_cells[y][x]
            #assert count <= 4
            for b in xrange(nbins):
                hog_cells[y][x][b] /= count

    return hog_cells

def gradLine(img_out, m_y, m_x, length, rad):
    x_start = int(m_x - length * np.cos(rad))
    y_start = int(m_y - length * np.sin(rad))
    x_end = int(m_x + length * np.cos(rad))
    y_end = int(m_y + length * np.sin(rad))
    cv2.line(img_out, (x_start, y_start), (x_end, y_end), 255, 1)

def drawHOG(hog_cells, img_size, cell_size):
    img_out = np.zeros(list(reversed(img_size)))
    cell_in_x = hog_cells.shape[1]
    cell_in_y = hog_cells.shape[0]
    nbins = hog_cells.shape[2]
    binRadRange = np.pi/nbins
    length_max = min(cell_size)/2.0
    for y in xrange(cell_in_y):
        for x in xrange(cell_in_x):
            grad_max = np.max(hog_cells[y][x])
            if grad_max == 0:
                continue
            m_x = cell_size[0]*x + cell_size[0]/2
            m_y = cell_size[1]*y + cell_size[1]/2
            for b in xrange(nbins):
                rad = b*binRadRange + binRadRange/2.0
                length = length_max * hog_cells[y][x][b] / grad_max
                if length == 0:
                    continue
                gradLine(img_out, m_y, m_x, length, rad)
    return img_out

def arcLength(cnt):
    return cv2.arcLength(cnt, False)

def findConvexHull(img, preprocess,
                   edge_th_min=100, edge_th_max=200,
                   show=True, name=''):
    if preprocess:
        ## blurring image ##
        img_blur = cv2.bilateralFilter(img, 11, 17, 17)

        ## adding image contrast ##
        '''img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
        y, u, v = cv2.split(img_yuv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        y = clahe.apply(y)
        img_yuv = cv2.merge([y, u, v])
        img_addcon = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)'''
    else:
        img_blur = img.copy()

    ## edge detection ##
    img_edge = cv2.Canny(img_blur, edge_th_min, edge_th_max)
    img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, (3,3))
    if show:
        cv2.imshow('img_edge'+name, img_edge)

    ## find contours ##
    _, cnts, hierarchy = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:50]
    '''cnts_approx = []
    for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            cnts_approx.append(approx)
    cnts_approx = sorted(cnts_approx, key = cv2.contourArea, reverse = True)'''
    print "before Hull and area filter:", len(cnts)
    if show:
        img_raw_cnt = img.copy()
        cv2.drawContours(img_raw_cnt, cnts, -1, (0, 0, 0), 1)
        cv2.imshow('image_raw_contour'+name, img_raw_cnt)


    ## Convex Hull ##
    cnts_convex = []
    for c in cnts:
            # using convex hull
            c_hull = cv2.convexHull(c)
            cnts_convex.append(c_hull)
    return cnts_convex

## filter contours by area ##
def AreaFilter(cnts_convex, cnts_convex_exts=None,
               img_area=640*480,
               area_th_min=0.01,
               area_th_max=0.2,
               area_max=0.04):
    if not cnts_convex_exts == None:
        cnt_ext_pairs = zip(cnts_convex, cnts_convex_exts)
        cnt_ext_pairs.sort(key=lambda pair: cv2.contourArea(pair[0]), reverse=True)
        cnt_ext_pairs = zip(*cnt_ext_pairs)
        cnts_convex, cnts_convex_exts = list(cnt_ext_pairs[0]), list(cnt_ext_pairs[1])
    else:
        cnts_convex = sorted(cnts_convex, key=cv2.contourArea, reverse=True)

    max = cv2.contourArea(cnts_convex[0])
    print 'max area:', max
    if max < img_area*0.5:
        area_th_max = 1.0

    new_cnts_convex = []
    new_cnts_convex_exts = []
    for c, i in zip(cnts_convex, range(len(cnts_convex))):
        area = cv2.contourArea(c)
        if area < img_area*area_max:
            if area >= max*area_th_min and area <= max*area_th_max:
                new_cnts_convex.append(c)
                if not cnts_convex_exts == None:
                    new_cnts_convex_exts.append(cnts_convex_exts[i])
            else:
                print "filter out area:", area
        else:
            print "filter out area:", area

    return new_cnts_convex, new_cnts_convex_exts

## calculate external points of contours and filter by aspect if request ##
def CalcExts(cnts, filter=False, th=2.0):
    cnts_filter = []
    cnts_exts = []
    for c in cnts:
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        #print extLeft, extRight, extTop, extBot
        exts = {'left':extLeft, 'right':extRight, 'top':extTop, 'bottom':extBot}
        width = extRight[0] - extLeft[0]
        height = extBot[1] - extTop[1]
        #print width, height
        if filter:
            if float(height)/width < th and float(height)/width > 1/th:
                cnts_exts.append(exts)
                cnts_filter.append(c)
            else:
                print "filter", float(height)/width, "by aspect"
        else:
            cnts_exts.append(exts)
            cnts_filter.append(c)
    return cnts_filter, cnts_exts

## merge overlap contour ##
def MergeOverlap(cnts, cnts_exts):
    cnts_copy = zip(cnts, cnts_exts, range(len(cnts)))
    has_checked = np.zeros(len(cnts))
    last_count = len(cnts) - 1
    for c_i, c_i_exts, i in cnts_copy:
        if has_checked[i] == 0:
            for c_j, c_j_exts, j in cnts_copy:
                if (not i == j) and (has_checked[j] == 0):
                    inside = False
                    for p in c_i_exts:
                        check = cv2.pointPolygonTest(c_j, c_i_exts[p], False)
                        if check == 1:
                            inside = True
                    if inside:
                        new_c = np.asarray((c_j.tolist() + c_i.tolist()), dtype=np.int32)
                        new_c = cv2.convexHull(new_c)
                        extLeft = tuple(new_c[new_c[:, :, 0].argmin()][0])
                        extRight = tuple(new_c[new_c[:, :, 0].argmax()][0])
                        extTop = tuple(new_c[new_c[:, :, 1].argmin()][0])
                        extBot = tuple(new_c[new_c[:, :, 1].argmax()][0])
                        new_exts = {'left':extLeft, 'right':extRight, 'top':extTop, 'bottom':extBot}
                        last_count += 1
                        has_checked[i] = 1.0
                        has_checked[j] = 1.0
                        has_checked = np.append(has_checked, [0.])
                        cnts_copy.append((new_c, new_exts, last_count))
                        break
    assert len(has_checked) == len(cnts_copy)
    cnts_merge = []
    cnts_merge_exts = []
    for i in xrange(len(has_checked)):
        if has_checked[i] == 0:
            cnts_merge.append(cnts_copy[i][0])
            cnts_merge_exts.append(cnts_copy[i][1])

    print "merge from", len(cnts), "to", len(cnts_merge)
    return cnts_merge, cnts_merge_exts

def ModifyMinRect(rect, tx, ty, addLength, new_angle):
    new_center = list(rect[0])
    new_shape = list(rect[1])
    new_center[0] += tx
    new_center[1] += ty
    new_shape[0] += addLength
    new_shape[1] += addLength
    return (tuple(new_center), tuple(new_shape), new_angle)

def TranslateContour(cnt, tx, ty):
    new_cnt = []
    for point in cnt:
        new_x = point[0][0] + tx
        new_y = point[0][1] + ty
        new_cnt.append([[new_x, new_y]])
    return np.asarray(new_cnt, dtype=np.int32)

def RotateContour(cnt, center, angle):
    rad = angle*np.pi/180.0
    new_cnt = []
    for point in cnt:
        ## first translate
        new_x = point[0][0] - center[0]
        new_y = point[0][1] - center[1]
        ## rotate
        new_x_p = new_x*np.cos(rad) - new_y*np.sin(rad)
        new_y_p = new_x*np.sin(rad) + new_y*np.cos(rad)
        ## translate back
        new_x_p += center[0]
        new_y_p += center[1]
        new_cnt.append([[new_x_p, new_y_p]])
    return np.asarray(new_cnt, dtype=np.int32)

def HistEqual(hist, k):
    assert k%2 == 1, 'k of HistEqual() need to be odd number!'
    hist_equal = []
    max = len(hist)
    for bin in xrange(max):
        total = 0
        start = bin - k/2
        valid_k = 0
        for i in xrange(k):
            if (start+i) >= 0 and (start+i) < max:
                total += hist[start+i][0]
                valid_k += 1
        hist_equal.append([total/float(valid_k)])
    return hist_equal

def findPeak(histr, max_num, th):
    pre_val = -1.0
    pre_slope = 1 ## 1: ascend, -1: descend, 0: no change
    peaks = [] ## list of tuple(which bin, number of pixels)
    for bin, bin_val in enumerate(histr):
        curr_slope = int((bin_val[0] - pre_val) > 0) - int((bin_val[0] - pre_val) < 0)
        if (not curr_slope == pre_slope) and (not pre_slope == -1) and (not curr_slope == 1):
            if(histr[bin-1][0] > max_num*th):
                peaks.append((bin-1, histr[bin-1][0]))
        pre_val = bin_val[0]
        pre_slope = curr_slope
    if pre_slope == 1 and (bin_val[0] > max_num*th):
        peaks.append((bin, bin_val[0]))
    peaks.sort(key=lambda peak: peak[1], reverse=True)
    return peaks



if __name__ == '__main__':
    img_name = 'stop_sign3'
    img = cv2.imread('./pictures/'+img_name+'.jpg')
    img = cv2.resize(img, (640, 480))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    start = time.time()
    '''## calc histgram of gradient ##
    winSize = tuple(reversed(img_gray.shape))#(img.shape[0]/16*16, img.shape[1]/16*16)#(208,208)
    blockSize = (16,16)

    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    hist = hog.compute(img_gray)

    hog_cells = HOGofCells(hist, winSize, blockSize, cellSize, blockStride, nbins)
    img_hog = drawHOG(hog_cells, winSize, cellSize)
    #output_img = img.copy()
    #output_img[np.where(mask==0)] = 0'''

    ## image histogram equalization(first add bilateral blur) ##
    cnts_convex = findConvexHull(img,
                                 preprocess=True,
                                 edge_th_min=100,
                                 edge_th_max=200,
                                 show=False,
                                 name='1st')
    cnts_convex, _ = AreaFilter(cnts_convex,
                                area_th_min=0.001,
                                area_th_max=0.2)
    print "number of cnts_convex:", len(cnts_convex), '\n'
    #cnts_filter, cnts_exts = CalcExts(cnts_convex)
    img_convex = img.copy()
    #print "after aspect filtered:", len(cnts_filter), '\n'
    cv2.drawContours(img_convex, cnts_convex, -1, (255, 255, 255), 1)

    cnts_convex_2nd = findConvexHull(img_convex,
                                     preprocess=False,
                                     edge_th_min=100,
                                     edge_th_max=200,
                                     show=False,
                                     name='2nd')
    cnts_convex_2nd, _ = AreaFilter(cnts_convex_2nd,
                                    area_th_min=0.001,
                                    area_th_max=0.2)
    print "number of cnts_convex_2nd:", len(cnts_convex_2nd)
    cnts_filter_2nd, cnts_exts_2nd = CalcExts(cnts_convex_2nd, filter=True, th=5.0)
    img_convex_2nd = img.copy()
    print "after aspect filtered:", len(cnts_filter_2nd), '\n'
    cv2.drawContours(img_convex_2nd, cnts_filter_2nd, -1, (0, 255, 0), 1)

    img_merge = img.copy()
    cnts_merge, cnts_exts_merge = MergeOverlap(cnts_filter_2nd, cnts_exts_2nd)
    cnts_merge_f, cnts_merge_f_exts = CalcExts(cnts_merge, filter=True, th=5.0)
    print "after aspect filtered:", len(cnts_merge_f)

    cnts_merge_f, cnts_merge_f_exts = AreaFilter(cnts_merge_f, cnts_merge_f_exts,
                                                 area_th_min=0.4,
                                                 area_th_max=1.0)
    cv2.drawContours(img_merge, cnts_merge_f, -1, (0, 255, 0), 1)
    '''## draw external points ##
    for ext in cnts_merge_f_exts:
        for p in ext:
            cv2.circle(img_merge, ext[p], 3, (255, 0, 0), -1)'''

    ## get individual letter crop image ##
    img_minbox = img.copy()
    img_crops = []
    cnts_minRect = []
    cnts_fit = []
    num = 0
    for c in cnts_merge_f:
    #if True:
    #    c = cnts_merge_f[num]
        ## find min Rect ##
        rect = cv2.minAreaRect(c)
        ## convert minRect to drawable box points ##
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ## draw min Rect ##
        cv2.drawContours(img_minbox ,[box], 0, (0,0,255), 2)
        ## crop letter ##
        img_crop = img.copy()
        l = np.min(box[:,0])
        r = np.max(box[:,0])
        t = np.min(box[:,1])
        b = np.max(box[:,1])
        buffer = 30
        length = max((r-l+buffer), (b-t+buffer), int((max(rect[1])+buffer)))
        center = np.asarray(rect[0], dtype=np.int)
        # crop x #
        if center[0] - length/2 < 0:
            img_crop = img_crop[:, 0:2*center[0]]
            tx = 0
        elif center[0] + length/2 >= 640:
            img_crop = img_crop[:, center[0]-(640-center[0]):640]
            tx = -(center[0]-(640-center[0]))
        else:
            img_crop = img_crop[:, center[0]-length/2:center[0]+length/2]
            tx = -(center[0]-length/2)
        # crop y ##
        if center[1] - length/2 < 0:
            img_crop = img_crop[0:2*center[1], :]
            ty = 0
        elif center[1] + length/2 >= 480:
            img_crop = img_crop[center[1]-(480-center[1]):480, :]
            ty = -(center[1]-(480-center[1]))
        else:
            img_crop = img_crop[center[1]-length/2:center[1]+length/2, :]
            ty = -(center[1]-length/2)
        img_crops.append(img_crop)
        cnts_minRect.append(ModifyMinRect(rect, tx, ty, 0, rect[2]))
        cnts_fit.append(TranslateContour(c, tx, ty))
        ## rotate letter image ##
        rows = img_crops[-1].shape[0]
        cols = img_crops[-1].shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2), cnts_minRect[-1][2], 1)
        img_crops[-1] = cv2.warpAffine(img_crops[-1], M, (cols, rows))
        cnts_fit[-1] = RotateContour(cnts_fit[-1], (cols/2, rows/2), -cnts_minRect[-1][2])
        ## second crop ##
        center = np.asarray(cnts_minRect[-1][0], dtype=np.int)
        width = int(cnts_minRect[-1][1][0])
        height = int(cnts_minRect[-1][1][1])
        extra = 5
        try:
            x_start = center[0] - width/2 - extra
            y_start = center[1] - height/2 - extra
            img_crops[-1] = img_crops[-1][y_start:y_start+height+2*extra,
                                          x_start:x_start+width+2*extra]
            cnts_minRect[-1] = ModifyMinRect(cnts_minRect[-1],
                                             -x_start,
                                             -y_start,
                                             0, 0)
            cnts_fit[-1] = TranslateContour(cnts_fit[-1], -x_start, -y_start)
        except:
            type, msg, _ = sys.exc_info()
            print type+'\n'+msg
        #box = cv2.boxPoints(cnts_minRect[-1])
        #box = np.int0(box)
        #cv2.drawContours(img_crops[-1] ,[box], -1, (0,0,255), 2)
        #cv2.imshow(img_name+'letter'+str(num), img_crops[-1])
        ## calc hue histogram ##
        '''mask = np.zeros(img_crops[-1].shape[:2], np.uint8)
        mask[extra:extra+height,extra:extra+width] = 255'''
        mask = np.zeros(img_crops[-1].shape[:2], np.uint8)
        cv2.drawContours(mask, [cnts_fit[-1]], -1, 255, -1)
        img_crops[-1] = cv2.GaussianBlur(img_crops[-1],(3,3), 0)
        img_crop_hsv = cv2.cvtColor(img_crops[-1], cv2.COLOR_BGR2HSV)
        first = np.array(img_crop_hsv[:,:,0], dtype=np.int)
        second = np.array(img_crop_hsv[:,:,1], dtype=np.int)
        third = np.array(img_crop_hsv[:,:,2], dtype=np.int)
        img_mul_2 = np.asarray((1*first+0*third)/1, dtype=np.uint8)
        img_mul_1 = np.asarray((2*first+1*third)/3, dtype=np.uint8)
            hist_2 = cv2.calcHist([img_mul_2], [0], mask, [180], [0,180])
        hist_1 = cv2.calcHist([img_mul_1], [0], mask, [205], [0,205])
        hist_equal_2 = HistEqual(hist_2, 3)
        hist_equal_1 = HistEqual(hist_1, 7)

        ## find histogram peaks and filter out small peak values ##
        shape = img_crop_hsv.shape
        peaks_2 = findPeak(hist_equal_2, shape[0]*shape[1], 0)
        peaks_1 = findPeak(hist_equal_1, shape[0]*shape[1], 0)

        ltr_mask2 = np.bitwise_and((img_mul_2 >= peaks_2[0][0] - 4), (img_mul_2 <= peaks_2[0][0] + 4))
        ltr_mask1 = np.bitwise_and((img_mul_1 >= peaks_1[0][0] - 4), (img_mul_1 <= peaks_1[0][0] + 4))
        ltr_mask_comb = np.bitwise_and(ltr_mask1, ltr_mask2)
        ltr_mask2 = np.array(ltr_mask2, dtype=np.int)
        ltr_mask1 = np.array(ltr_mask1, dtype=np.int)
        ltr_mask_comb = np.array(ltr_mask_comb, dtype=np.int)
        '''if False:#len(peaks) > 1:
            mask2 = np.bitwise_and((img_mul >= peaks[1][0] - 0), (img_mul <= peaks[1][0] + 0))
            mask2 = np.array(mask2, dtype=np.int)
            mask1 = mask1+mask2'''
        foreground = np.array([0, 255], dtype=np.uint8) ## letter:white(255), background:black(0)
        img_black_2 = foreground[ltr_mask2]
        img_black_1 = foreground[ltr_mask1]
        img_black_comb = foreground[ltr_mask_comb]
        if num == 27:
            ## first letter mask
            plt.plot(hist_2, color = 'r')
            plt.plot(hist_equal_2, color = 'g')
            plt.xlim([0,180])
            ## second letter mask
            plt.figure()
            plt.plot(hist_1, color = 'r')
            plt.plot(hist_equal_1, color = 'g')
            plt.xlim([0,204])
            plt.ion()
            plt.show()
            #print peaks_1
            '''pix_num = 0
            pix_count = 0
            f = open('pix.txt', 'w')
            f_m = open('mask.txt', 'w')
            for i in xrange(len(img_mul)):
                for j in xrange(len(img_mul[i])):
                    f.write(str(img_mul[i][j]).rjust(3)+' ')
                    f_m.write(str(mask[i][j]).rjust(3)+' ')
                    if mask[i][j] == 255:
                        if img_mul[i][j] == 84:
                            pix_count+=1
                        pix_num+=1
                f.write('\n')
                f_m.write('\n')
            #print pix_num, pix_count, img_mul.shape
            f.close()
            f_m.close()'''
            '''cv2.imshow('mask', mask)
            mask_fit = np.zeros(img_crops[-1].shape[:2], np.uint8)
            cv2.drawContours(mask_fit, [cnts_fit[-1]], -1, 255, -1)
            cv2.imshow('mask_fit', mask_fit)'''

        '''img_blur = cv2.GaussianBlur(img_crops[-1],(3,3), 0)
        img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
        y, u, v = cv2.split(img_yuv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        img_yuv = cv2.merge([y, u, v])
        img_addcon = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        img_edge = cv2.Canny(img_addcon, 100, 200)
        img_edge = cv2.dilate(img_edge, (3,3), 2)
        #_, cnts, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img_edge, cnts, -1, (255, 255, 255), 2)'''

        #img_crop_gray = cv2.cvtColor(img_crops[-1], cv2.COLOR_BGR2GRAY)

        img_black_2 = np.bitwise_and(img_black_2, mask)
        #img_black_2 = cv2.morphologyEx(img_black_2, cv2.MORPH_CLOSE, (9,9))
        cv2.imshow(img_name+'letter_hue'+str(num), img_black_2)
        #cv2.imwrite('./letters/'+img_name+'letter_hue'+str(num)+'.jpg', img_black_2)

        img_black_1 = np.bitwise_and(img_black_1, mask)
        #img_black = cv2.morphologyEx(img_black, cv2.MORPH_CLOSE, (9,9))
        cv2.imshow(img_name+'letter_mix'+str(num), img_black_1)
        #cv2.imwrite('./letters/'+img_name+'letter_mix'+str(num)+'.jpg', img_black_1)

        img_black_comb = np.bitwise_and(img_black_comb, mask)
        #img_black_comb = cv2.morphologyEx(img_black_comb, cv2.MORPH_CLOSE, (9,9))
        cv2.imshow(img_name+'letter_comb'+str(num), img_black_comb)
        cv2.imwrite('./letters2/'+img_name+'letter_comb'+str(num)+'.jpg', img_black_comb)
        num += 1


    print '\nTotal crop letters:', num
    print 'Total Time elapsed: ' + '{:.6f}'.format((time.time() - start)) + ' secs'
    cv2.imshow('image', img_gray)
    #cv2.imshow('image_hog', img_hog)
    cv2.imshow('image_convex', img_convex)
    cv2.imshow('image_convex_2nd', img_convex_2nd)
    cv2.imshow('image_merge', img_merge)
    cv2.imshow('image_minbox', img_minbox)
    #cv2.imshow('image_addcon', img_addcon)
    cv2.imwrite('result.jpg', img_minbox)
    #cv2.imwrite('crop.jpg', img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
