import os
from skimage import io
import face_alignment
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def fan_func(image_path=None, landmark_path=None, fileList_path=None):
  if not os.path.exists(landmark_path):
    os.mkdir(landmark_path)

  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,face_detector='sfd')
  filenames = os.listdir(image_path)
  print("Number of images: ", len(filenames))

  detected = 0
  not_detected = []
  for filename in filenames:
    if filename[-3:] != 'png' and filename[-3:] != 'jpg':
      continue
    
    img = io.imread(os.path.join(image_path, filename))
    print(filename+' ')
    l_pos = fa.get_landmarks(img)
    if l_pos == None:
      print('not detected\n')
      not_detected.append(filename)
      continue
     
    with open(os.path.join(landmark_path, filename[:-4]+'.txt'), 'w') as f:
      for i in range(68):
        f.write(str(l_pos[0][i,0])+' '+str(l_pos[0][i,1])+' ')
      f.write('\n')
      detected += 1

  print("Detected landmarks: ", detected)
  ext = ['.jpg', '.png', '.txt']

  images = []
  for root, dirs, files in os.walk(landmark_path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
          images.append(os.path.join(root, file))

  images = sorted(images)
  np.savetxt(fileList_path, images, fmt='%s')

  return not_detected