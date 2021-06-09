import logging
import os
import time
from operator import itemgetter
import numpy as np
import cv2
import project_root_dir

log_file_root_path = os.path.join(project_root_dir.project_dir, 'logs')
log_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))


def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def save_to_file(root_dic, tracker):
    filter_face_addtional_attribute_list = []
    for item in tracker.face_addtional_attribute:
        if item[2] < 1.4 and item[4] < 1:  # recommended thresold value
            filter_face_addtional_attribute_list.append(item)
    if len(filter_face_addtional_attribute_list) > 0:
        score_reverse_sorted_list = sorted(filter_face_addtional_attribute_list, key=itemgetter(4))
        mkdir(root_dic)
        trackIdDir = os.path.join(root_dic, str(tracker.id))
        if not os.path.exists(trackIdDir):
            mkdir(trackIdDir)
        for i in range(len(score_reverse_sorted_list)):
            faceScore = round(score_reverse_sorted_list[i][1], 3)
            fileName = os.path.join(trackIdDir,str(faceScore)+'.png')
            cv2.imwrite(fileName, score_reverse_sorted_list[i][0])


def saveMatchedFace(rootDir, trackId, addtionalAttribute):
    if not os.path.exists(rootDir):
        mkdir(rootDir)
    trackDir = os.path.join(rootDir,str(trackId))
    if not os.path.exists(trackDir):
        mkdir(trackDir)
    cropped, score, dist_rate, high_ratio_variance, width_rate = addtionalAttribute
    faceScore = round(score,4)
    fileName = os.path.join(trackDir,str(faceScore) + '.png')
    cv2.imwrite(fileName,cropped)


def saveFaceInTracker(rootDir, frameId, tracker):
    if not os.path.exists(rootDir):
        mkdir(rootDir)
    frameDir = os.path.join(rootDir,str(frameId+1))
    if not os.path.exists(frameDir):
        mkdir(frameDir)
    assert len(tracker.face_addtional_attribute) > 0, 'There must be 1 face in track'
    cropped, score = tracker.face_addtional_attribute[-1]
    faceScore = round(score,4)
    fileName = f'{tracker.id}_{faceScore}.png'
    fileName = os.path.join(frameDir,fileName)
    cv2.imwrite(fileName,cropped)
    

def detect_blur_fft(image, size = 60, thresh = 10):
  # NOTE: size: The size of the radius around the centerpoint of
  #             the image for which we will zero out the FFT shift


  # grab the dimensions of the image and use the dimensions to
	# derive the center  (x, y)-coordinates
  (h, w) = image.shape
  (cX, cY) = (int(w / 2.0), int(h / 2.0))

  # compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
  fft = np.fft.fft2(image)
  fftShift= np.fft.fftshift(fft)

  # zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
  fftShift[cY - size:cY + size, cX - size:cX + size] = 0
  fftShift = np.fft.ifftshift(fftShift)
  recon = np.fft.ifft2(fftShift)
 
  # compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
  magnitude = 20 * np.log(np.abs(recon))
  mean = np.mean(magnitude)
 
  # the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
  return mean, mean <= thresh

class Logger:

    def __init__(self, module_name="MOT"):
        super().__init__()
        path_join = os.path.join(log_file_root_path, module_name)
        mkdir(path_join)

        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(path_join, '{}.log'.format(log_time))
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s -  %(threadName)s - %(process)d ")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def error(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        if self.logger is not None:
            self.logger.exception(msg, *args, exc_info=True, **kwargs)
