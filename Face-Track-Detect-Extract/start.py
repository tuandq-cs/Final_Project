import argparse
import os
from time import time

import align.detect_face as detect_face
import cv2
import numpy as np
# import tensorflow as tf
# from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
# from project_root_dir import project_dir
from src.sort import Sort
from utils import *


logger = Logger()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def getVideoName(filePath):
    filePath = os.path.splitext(filePath)[0]
    return filePath.split('/')[-1]

def getFullPath(path):
    return os.path.join(ROOT_DIR,path)

def __getFPS(cam):
    # Find OpenCV version
    (major_ver, _, _) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    if int(major_ver)  < 3 :
        return cam.get(cv2.cv.CV_CAP_PROP_FPS)
    return cam.get(cv2.CAP_PROP_FPS)

def initVideoWriter(cam, scale_rate, filename, output_videos_dir):
    if not os.path.exists(output_videos_dir):
        mkdir(output_videos_dir)
    # We convert the resolutions from float to integer.
    frame_width = int(cam.get(3) * scale_rate) 
    frame_height = int(cam.get(4) * scale_rate)
    FPS = __getFPS(cam)
    logger.info('Video has {FPS}fps')
    outVideoName = f'{getVideoName(filename)}_out_{FPS}fps.avi'
    # Define the codec and create VideoWriter object.The output is stored in '{}.avi' file.
    return cv2.VideoWriter(os.path.join(output_videos_dir,outVideoName),cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))

def main():
    global colours, img_size
    args = parse_args()
    videos_dir = getFullPath(args.videos_dir)
    output_images_dir = getFullPath(args.output_images_dir)
    output_videos_dir = getFullPath(args.output_videos_dir)
    isSaveVideos = args.save_videos
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    # Make directory for output images
    if not os.path.exists(output_images_dir):
        mkdir(output_images_dir)

    # For display
    if not no_display:
        colours = np.random.rand(32, 3)

    # Init tracker
    tracker = Sort()  # create instance of the SORT tracker
    logger.info(args)
    logger.info('Start track and extract......')
    # with tf.Graph().as_default():
    #     with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
    #                                           log_device_placement=False)) as sess:
    #         pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
    net = cv2.dnn.readNetFromDarknet('yoloface/yolov3-face.cfg', 'yoloface/yolov3-wider_16000.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # TODO consider these configs
    minsize = 40  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold 
    factor = 0.709  # scale factor

    for filename in os.listdir(videos_dir):
        logger.info('All files:{}'.format(filename))
    for filename in os.listdir(videos_dir):
        start = time()
        suffix = filename.split('.')[1]
        if suffix != 'mp4' and suffix != 'avi':  # you can specify more video formats if you need
            continue
        video_name = os.path.join(videos_dir, filename)
        directoryName = os.path.join(output_images_dir, filename.split('.')[0])
        logger.info('Video_name:{}'.format(video_name))
        cam = cv2.VideoCapture(video_name)
        c = 0

        videoWriter = None
        if isSaveVideos:
            videoWriter = initVideoWriter(cam, scale_rate, filename, output_videos_dir)

        while True:
            final_faces = []
            addtional_attribute_list = []
            ret, frame = cam.read()
            if not ret:
                logger.warning("ret false")
                break
            if frame is None:
                logger.warning("frame drop")
                break

            # frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
            # r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
            if c % detect_interval == 0:
                img_size = np.asarray(frame.shape)[0:2]
                mtcnn_starttime = time()
                # faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                #                                         factor)
                # Sets the input to the network
                net.setInput(blob)
                # Runs the forward pass to get output of the output layers
                outs = net.forward(get_outputs_names(net))
                # Remove the bounding boxes with low confidence
                faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                logger.info("Yolov3 detect face cost time : {} s".format(
                    round(time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                face_sums = faces.shape[0]
                if face_sums > 0:
                    face_list = []
                    for i, item in enumerate(faces):
                        score = round(faces[i, 4], 6)
                        if score > face_score_threshold:
                            det = np.squeeze(faces[i, 0:4])

                            # face rectangle
                            det[0] = np.maximum(det[0] - margin, 0)
                            det[1] = np.maximum(det[1] - margin, 0)
                            det[2] = np.minimum(det[2] + margin, img_size[1])
                            det[3] = np.minimum(det[3] + margin, img_size[0])
                            face_list.append(item)

                            # face cropped
                            bb = np.array(det, dtype=np.int32)

                            # use 5 face landmarks  to judge the face is front or side
                            # squeeze_points = np.squeeze(points[:, i])
                            # tolist = squeeze_points.tolist()
                            # facial_landmarks = []
                            # for j in range(5):
                            #     item = [tolist[j], tolist[(j + 5)]]
                            #     facial_landmarks.append(item)
                            # if args.face_landmarks:
                            #     for (x, y) in facial_landmarks:
                            #         cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                            # dist_rate, high_ratio_variance, width_rate = judge_side_face(
                            #     np.array(facial_landmarks))

                            # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                            # item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                            item_list = [cropped, score]
                            addtional_attribute_list.append(item_list)

                    final_faces = np.array(face_list)

            trackers = tracker.update(final_faces, img_size, directoryName, addtional_attribute_list, detect_interval)
            tracker.saveFaceImages(rootDir=directoryName,frameId=c)

            for d in trackers:
                if not no_display:
                    d = d.astype(np.int32)
                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                    if final_faces != []:
                        cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    colours[d[4] % 32, :] * 255, 2)
                        cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (1, 1, 1), 2)
                    else:
                        cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75,
                                    colours[d[4] % 32, :] * 255, 2)
            c += 1
            # Write the frame into the file '{}.avi'
            if isSaveVideos:
                videoWriter.write(frame)

            if not no_display:
                showFrame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                cv2.imshow("Frame", showFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        logger.info(f'Total compute time: {time() - start}')
        logger.info('Prepare to release video ...')
        if isSaveVideos:
            videoWriter.release()
        logger.info('---> Finish release video <---')
        cv2.destroyAllWindows()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_images_dir', type=str,
                        help='Path to save face images',
                        default='facepics')
    parser.add_argument('--output_videos_dir', type=str,
                        help='Path to save inference video',
                        default='extracted_videos')
    parser.add_argument('--save_videos',
                        help='Save inference video or not',
                        action='store_true')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=1)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
