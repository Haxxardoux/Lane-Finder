#!/usr/bin/env python3
import time
import cv2
from display import Display
from frame import Frame, denormalize, match_frames, IRt
import numpy as np
#import g2o

# camera intrinsics
W, H = 1920, 1080

F = 270
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])


class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []

  def display(self):
    for f in self.frames:
      print(f.id)
      print(f.pose)
      print()

# main classes
disp = Display(W, H)
mapp = Map()

class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

  def __init__(self, mapp, loc):
    self.xyz = loc
    self.frames = []
    self.idxs = []

    self.id = len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
  return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def display_lines(img, original_image):
    lines = cv2.HoughLinesP(img, 10, np.pi/180, 100, np.array([]), minLineLength = 25, maxLineGap = 3)
    #lines = cv2.HoughLines(img, 5, np.pi/180, 150, )
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if (0.2 < (y2-y1)/(x2-x1) < 100000) or (-10000 < (y2-y1)/(x2-x1) < -0.2):
                cv2.line(original_image, (x1, y1), (x2, y2), (255,0,0), 10)

    return original_image

def process_frame(img):
  #img = cv2.resize(img, (W,H))
  frame = Frame(mapp, img, K)
  if frame.id == 0:
    return

  f1 = mapp.frames[-1]
  f2 = mapp.frames[-2]

  idx1, idx2, Rt = match_frames(f1, f2)
  f1.pose = np.dot(Rt, f2.pose)

  # homogeneous 3-D coords
  pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
  pts4d /= pts4d[:, 3:]

  # reject pts without enough "parallax" (this right?)
  # reject points behind the camera
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

  for i,p in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    pt = Point(mapp, p)
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
    u1, v1 = denormalize(K, pt1)
    u2, v2 = denormalize(K, pt2)
    cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
    cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))




  Grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(Grayscale, (5,5), 0)
  thresh = cv2.Canny(blur, 100, 200)
  roi_vertices = np.array([[525,300],[900,530],[50,530],[425,300]], dtype=np.int32)
  verts=roi_vertices*2
  mask = np.zeros_like(thresh)
  cv2.fillPoly(mask, [verts], 1)


  display_lines(thresh*mask,img)


  disp.paint(img)

  # 3-D
  mapp.display()

if __name__ == "__main__":
  cap = cv2.VideoCapture("/Users/pcuser/SelfDrivingCar/copy2/test.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
