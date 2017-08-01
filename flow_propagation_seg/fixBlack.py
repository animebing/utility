#!/usr/bin/python
import numpy as np

def fixBlack(img):
    new_img = np.copy(img)
    h, w = img.shape
    for i in xrange(h):
        for j in xrange(w):
            if img[i, j] != 19:
                continue

            size = 1
            while size <= 2:
                left = j-size
                if left < 0:
                    left = 0
                right = j+size
                if right > w-1:
                    right = w-1

                up = i-size
                if up < 0:
                    up = 0

                bottom = i+size
                if bottom > h-1:
                    bottom = h-1

                left = int(left)
                right = int(right)
                up = int(up)
                bottom = int(bottom)

                box = img[up:bottom+1, left:right+1]
                other = box[box!=19]
                if other.size != 0:
                    val, cnt = np.unique(other, return_counts=True)
                    major = val[np.argmax(cnt)]
                    new_img[i, j] = major
                    break
                size += 1
    return new_img

