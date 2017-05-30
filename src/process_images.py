# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:15:01 2017

@author: Dhruv Kaushik

A module to process images and convert them to conveniently store the data  

"""

import cv2
import numpy as np
from itertools import izip

'''DEFINING PARAMETERS FOR THE FILE'''
IMAGE_URL = 'IMages/3.jpeg'
RESCALING_RES = (150,150)
FINAL_RESCALING_RES = (30,30)
BILATERAL_FILTER_PARAMS = (30,30)


def pairwise(iterable):
    '''A helper function to process elements pairwise in lists
    s -> (s0, s1), (s2, s3), (s4, s5), ...'''
    a = iter(iterable)
    return izip(a, a)

'''Reading in image as a numpy array '''
img = cv2.imread(IMAGE_URL,0)

'''Rescaling the image to 150 x 150'''
res = cv2.resize(img,RESCALING_RES, interpolation = cv2.INTER_CUBIC)

'''Denoising the image'''
blur = cv2.bilateralFilter(res,9,BILATERAL_FILTER_PARAMS[0],BILATERAL_FILTER_PARAMS[1])

'''Binarizing the image'''
(thresh, im_bw) = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

'''Remvoing the black padding'''
mask = im_bw>0
final = im_bw[np.ix_(mask.any(1),mask.any(0))]

'''Inverting the image'''
final = cv2.bitwise_not(final)

'''Finding grid lines to determine the cell edges'''
a = np.array([ np.count_nonzero(final[i,:]) for i in range(final.shape[0])]) < 10
grid = np.ones(final.shape)*255
grid[a,:] = 0           
a = np.array([ np.count_nonzero(final[:,i]) for i in range(final.shape[1])]) < 10
grid[:,a] = 0

'''Writing the final processed array to an image '''
cv2.imwrite('processed.png',final)           

'''Cell edges for columns and rows of cells'''    
col = [0]
for i in range(1,grid.shape[0]):
    if grid[i-1,0] != grid[i,0]:
        col.append(i)
col.append(-1)
row = [0]
for i in range(1,grid.shape[1]):
    if grid[0,i-1] != grid[0,i]:
        row.append(i)
row.append(-1)

'''Creating individual images for each cell'''
i=1
for x1, x2 in pairwise(col):
    j=1
    for y1,y2 in pairwise(row):
        m = final[x1+2:x2-2,y1+2:y2-2]
        '''Rescaling the image to 150 x 150'''
        res = cv2.resize(m,FINAL_RESCALING_RES, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite( str(i)+str(j)+'.png', res )
        j += 1
    i += 1

    
'''Writing the extracted grid line to an image'''
cv2.imwrite('grid.png',grid)

