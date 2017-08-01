# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 20:03:58 2016
@author: ziheng
"""
from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def imgRead(img_dir, channel='rgb'):
  '''
  Read an image and store it into a ch*h*w ndarray
  The image will be in rgb space by default.

  Args
  -------
  img_dir: str
    Path to the image to be read
  channel: str
    'rgb' for rgb color space and 'bgr' for gbr color space

  Returns
  -------
  A processed ndarray image
  '''
  img = cv2.imread(img_dir)
  if channel == 'rgb':
    img = img[:,:,[2,1,0]]

  return img

def imgNorm(img, mean = 0, std = 1):
  '''
  Normalize the image with mean and std

  Args
  -------
  img: numpy ndarray
    An image stored in a h*w*ch numpy ndarray
  mean
    Specify the desire mean to be substract from img
  std
    Specify the desire standard deviation

  Returns
  -------
  A normalized image
  '''
  output =  (img - mean) / std

  return output


def img2Blob(img):
  '''
  Rearranges the dimensions of img to match caffe's requirement
  The dimentions of the image will be rearranged as ch*h*w.

  Args
  -------
  img: numpy ndarray image
    Input image(s) as ndarray
  bs: uint
    Batch size

  Returns
  -------
  A processed ndarray image
  '''

  output = img.transpose(2, 0, 1)

  return output

def blob2Img(blob, idx = 0):
  '''
  Read an image from the caffe blob

  Args
  -------
  blob: numpy ndarray caffe blob
    Caffe data blob
  idx
    Which image in the batch will be read

  Returns
  -------
  A ndarray image
  '''

  if blob.ndim == 4:
    img = blob[idx,...]
  else: img = blob
  img = img.transpose(1, 2, 0)

  return img

def imgScale(img, f = (1, 1), d = None):
  '''
  Scale an image by a factor or to a desired size

  Args
  -------
  img: numpy ndarray image
    The input image
  f: turple (float, float)
    Scale the image by factor f = (fh, fw)
  d: turple (uint, uint)
    Scale the image to a desired size (h, w)

  Returns
  -------
  A processed ndarray image
  '''

  if f != None:
    dh, dw = int(img.shape[0]*f[0]), int(img.shape[1]*f[1])
  if d != None:
    dh, dw = d

  output = cv2.resize(img, (dw, dh))

  return output

def imgCrop(img, roi):
  '''
  Crop an image, only reserve the roi region of the whole image

  Args
  -------
  img: numpy ndarray image
    The input image
  roi: array like [uint, uint, uint, uint]
    The region of interest, i.e the reserved region, as a rectangular
    [x_start, y_start, x_end, y_end]

  Returns
  -------
  The Cropped image
  '''
  output = img[roi[1]:roi[3]+1,roi[0]:roi[2]+1,:]

def imgPad(img, l, r, t, b, val = 0):
  '''
  Pad an image

  Args
  -------
  img: numpy ndarray image
    The input image
  l, r, t, b: uint
    Left, Right, Top, Bottom padding size
  val: uint8
    Pad the image with the value val

  Returns
  -------
  Padded image
  '''

  p_l = np.ones((img.shape[0], l, img.shape[2]))*val
  p_r = np.ones((img.shape[0], r, img.shape[2]))*val
  p_t = np.ones((t, img.shape[1]+l+r, img.shape[2]))*val
  p_b = np.ones((b, img.shape[1]+l+r, img.shape[2]))*val

  output = np.uint8(np.concatenate((p_l, img, p_r), axis=1))
  output = np.uint8(np.concatenate((p_t, output, p_b), axis=0))

  return output

def imgCropas(img, d):
  ''' Crop the center area of shape d

  Args
  -------
  img: numpy ndarray image
    The input image
  d: turple (uint, uint)
    The desired output shape as (h, w)

  Returns
  -------
  The Cropped image
  '''

  c_err = max(0, ((img.shape[0] - d[0])) / 2.0), max(0, ((img.shape[1] - d[1]) / 2.0))
  x_start = int(np.floor(c_err[1]))
  x_end = int(max(0, img.shape[1] - np.ceil(c_err[1]) - 1))
  y_start = int(np.floor(c_err[0]))
  y_end = int(max(0, img.shape[0] - np.ceil(c_err[0]) - 1))

  return imgCrop(img, [x_start, y_start, x_end, y_end])

def imgPadas(img, d, val = 0):
  '''
  Pad around the image to shape d

  Args
  -------
  img: numpy ndarray image
    The input image
  d: turple (uint, uint)
    Desired output shape d = (h, w)

  Returns
  -------
  Padded image
  '''

  c_err = max(0, ((d[0] - img.shape[0]) / 2.0)), max(0, ((d[1] - img.shape[1]) / 2.0))
  l = int(np.floor(c_err[1]))
  r = int(np.ceil(c_err[1]))
  t = int(np.floor(c_err[0]))
  b = int(np.ceil(c_err[0]))

  return imgPad(img, l, r, t, b, val = val)

def imgReshape(img, d, val = 0, mode = 'crop'):
  '''
  Pad or Crop the image to the desired size

  Args
  -------
  img: numpy ndarray image
    The input image
  d: turple (uint, uint)
    Desired output shape d = (h, w)
  val: uint8
    Pad the image with value val if necessary
  mode: str
    alternatives are 'crop' and 'scale'.
    In 'crop' mode, the area beyond the desired size will be cropped, and the
    center area will be reserved, the blank area will be padded with value val.

    In 'scale' mode, the image will be scaled so that the image is no larger
    than the desired size, the blank area will be padded with value val.

  Returns
  -------
  Processed image
  '''

  if mode == 'crop':
    output = img_cropas(img, d)
  elif mode == 'scale':
    f = min(float(d[0])/img.shape[0], float(d[1])/img.shape[1])
    output = img_scale(img, (f, f))
  output = img_padas(output, d)

  return output

def imgCropROIas(img, roi, d = None, val = 0):
  '''
  Crop the roi from img and scale to the desired shape d, blank area will
  be padded with value val

  Args
  -------
  img: numpu ndarray image
    The input image
  roi: turple (uint, uint, uint, uint)
    The region of interest as (x0, y0, x1, y1)
  d: turple (uint, uint)
    The desired output size as (h, w)
  val: uint8
    The padding value
  '''
  output = imgCrop(img, roi)
  output = imgReshape(output, d, val, mode = 'scale')

  return output

def imgRot(img, angle, d = None, val = 0)
  '''
  Rotate and scal the image

  Args
  -------
  img: numpy ndarray image
    Input image
  angle: float
    Rotation angle in rad
  d: turple (uint, uint)
    Desired output shape in a tuple (h, w)
  val: uint8
    The padding value

  Returns
  -------
  The Rotated image
  '''

  w = img.shape[1]
  h = img.shape[0]
  rangle = np.deg2rad(angle) # angle in radians
  if d == None:
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))
  else:
    nh, nw = d

  scale = min(nw / (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w)),
              nh / (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w)))
  # ask OpenCV for the rotation matrix
  rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
  # calculate the move from the old center to the new center combined
  # with the rotation
  rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
  # the move only affects the translation, so update the translation
  # part of the transform
  rot_mat[0,2] += rot_move[0]
  rot_mat[1,2] += rot_move[1]
  return cv2.warpAffine(img, rot_mat, (int(np.ceil(nw)), int(np.ceil(nh))),
                        borderValue = (val, val, val),flags=cv2.INTER_LANCZOS4)

def imgFlip(img, direction = 'h'):
  '''
  Flip an image

  Args
  -------
  img: numpy ndarray image
    The input image
  direction: str
    Along which direction to flip, 'h' for horizontal flip and 'v' to vertical
    flip

  Returns
  -------
  The flipped image

  '''
  if direction == 'h':
    return cv2.flip(img, 1)
  elif direction == 'v':
    return cv2.flip(img, 0)
  else:
    raise ValueError('Invalid direction value!')

def imgShift(img, c):
  '''
  Circular shift a image so that c=(h, w) become the center of the image

  Args
  -------
  img : numpy ndarray image
    Input image
  c : tuple as (h, w)
    The pixel willing to put in the center

  Returns
  -------
  A shifted image with point c at the center
  '''

  ori_c = img.shape / 2
  th, tw = ori_c[0] - c[0], ori_c[1] - c[1]
  output = np.roll(img, th, axis=0)
  output = np.roll(img, tw, axis=1)

  return output

class imgShow(object):
  ''' Display an image using matplotlib

  Args
  -------
  img: numpy ndarray image
    The input image
  '''
  figno = 0
  def __init__(self, img):
    import matplotlib.pyplot as plt
    plt.figure(img_show.figno)
    plt.imshow(img)
    img_show.figno += 1

class imgiShow(object):
  '''
  Interactive edition of img_show

  Allow user draw a box on the image and get the box parameter. After called
  the method img_ishow.connect, one can interactivly draw a box on the image,
  and get box parameter from the .rect attribute as a tuple (x, y, w, h)


  Args
  -------
  img: numpy ndarray image
    The input image

  Methods
  -------
  connect()
    Start to draw box

  disconnect()
    Stop to draw box
  '''

  def __init__(self, img):
    plt.ion()
    self.fig = plt.subplot(111)
    self.img = self.fig.imshow(img)
    self.press = None
    self.rect = None
    self.key = None
    self.box = self.fig.add_patch(patches.Rectangle(
                                  (0, 0), #(x, y)
                                  0,  #width
                                  0,  #height
                                  fill=False,      # remove background
                                  edgecolor="red",
                                  linewidth=3
                                              ))
    self.connect()
  def connect(self):
    self.img.figure.canvas.mpl_connect('button_press_event', self.OnPress)
    self.img.figure.canvas.mpl_connect('button_release_event', self.OnRelease)
    self.img.figure.canvas.mpl_connect('motion_notify_event', self.OnMotion)
    self.img.figure.canvas.mpl_connect('key_press_event', self.OnKeyPress)
    self.img.figure.canvas.mpl_connect('key_release_event', self.OnKeyRelease)
    #raw_input("Press Enter to continue...")
    self.img.figure.canvas.start_event_loop(timeout=-1)
  def OnPress(self, event):
    if event.inaxes!=self.img.axes: return
    #print('On Press: x=%d, y=%d, xdata=%f, ydata=%f' %
    #    (event.x, event.y, event.xdata, event.ydata))
    self.press = event.xdata, event.ydata
    self.box.set_width(0)
    self.box.set_height(0)
    self.box.figure.canvas.draw()
  def OnMotion(self, event):
    if event.inaxes!=self.img.axes: return
    #print('On Motion: x=%d, y=%d, xdata=%f, ydata=%f' %
    #    (event.x, event.y, event.xdata, event.ydata))
    if self.press == None:
      return
    self.motion = event.xdata, event.ydata
    self.x0 = min((self.press[0], self.motion[0]))
    self.y0 = min((self.press[1], self.motion[1]))
    self.w = abs(self.press[0]-self.motion[0])
    self.h = abs(self.press[1]-self.motion[1])
    self.box.set_x(self.x0)
    self.box.set_y(self.y0)
    self.box.set_width(self.w)
    self.box.set_height(self.h)
    self.box.figure.canvas.draw()
  def OnRelease(self, event):
    if event.inaxes!=self.img.axes: return
    'on release we reset the press data'
    self.press = None
    self.rect = int(self.x0), int(self.y0), int(self.x0+self.w), int(self.y0+self.h)
    self.box.figure.canvas.draw()
  def OnKeyPress(self, event):
    pass
  def OnKeyRelease(self, event):
    self.img.figure.canvas.stop_event_loop()
    self.disconnect()
    self.key = event.key
    pass
#    self.key = event.key
#    if self.handler(fig_obj = self):
#      self.fig.canvas.stop_event_loop()
  def disconnect(self):
    'disconnect all the stored connection ids'
    self.box.figure.canvas.mpl_disconnect(self.OnPress)
    self.box.figure.canvas.mpl_disconnect(self.OnRelease)
    self.box.figure.canvas.mpl_disconnect(self.OnMotion)
    self.box.figure.canvas.mpl_disconnect(self.OnKeyPress)
    self.box.figure.canvas.mpl_disconnect(self.OnKeyRelease)

if __name__ == '__main__':
## Test all functions
  # Read an image
#  img = img_read('/home/ziheng/git/cpm/dataset/MPI/images/033473533.jpg')
#  fig = img_ishow(img)
#  roi = img_croproias(img, fig.rect, (64, 128), val = 128)
#  print('roi shape: %s'%(str(roi.shape)))
#  img_show(roi)
  pass
