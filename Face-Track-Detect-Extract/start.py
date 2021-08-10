import argparse
import json
import os
from re import L
from time import time
from numpy.core.defchararray import index

from numpy.lib.utils import source

import align.detect_face as detect_face
import cv2
import numpy as np
import dlib
# import tensorflow as tf
# from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir, extract_infomation, extract_index_nparray
# from project_root_dir import project_dir
from src.sort import Sort
from utils import *
from keras.models import load_model

logger = Logger()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_INFER_DIR = 'inferences'
FACE_SCORE_WEIGHT = 1
DICE_SIMILARITY_WEIGHT = 0.5

def getFileName(filePath):
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
    outVideoName = f'{getFileName(filename)}_out_{FPS}fps.avi'
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

    mkdir(BASE_INFER_DIR)
    # Make directory for output images
    if not os.path.exists(output_images_dir):
        mkdir(output_images_dir)


    shape_predictor = dlib.shape_predictor(os.path.join(ROOT_DIR, args.landmarks_model))
    smile_model = load_model(os.path.join(ROOT_DIR, args.smile_model))
    EYE_THRESH = args.eye_thresh
    BLUR_THRESH = args.blur_thresh

    best_face_info = {}
    best_frame_info = {}

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
        if suffix != 'mp4' and suffix != 'avi' and suffix != 'mov':  # you can specify more video formats if you need
            continue
        
        inference_dir = os.path.join(BASE_INFER_DIR, getFileName(filename))
        mkdir(inference_dir)
        video_name = os.path.join(videos_dir, filename)
        directoryName = os.path.join(output_images_dir, filename.split('.')[0])
        logger.info('Video_name:{}'.format(video_name))
        cam = cv2.VideoCapture(video_name)
        c = 0

        videoWriter = None
        if isSaveVideos:
            videoWriter = initVideoWriter(cam, scale_rate, filename, output_videos_dir)
        
        list_frame_info = []
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
            frame_id = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
            trackers = tracker.update(final_faces, img_size, directoryName, addtional_attribute_list, detect_interval)
            tracker.saveFaceImages(rootDir=directoryName,frameId=frame_id)

            frame_info = extract_infomation(tracker, frame_id, frame, smile_model, shape_predictor, BLUR_THRESH, EYE_THRESH)
            list_frame_info.append(frame_info)
            if best_frame_info.get('frame_id'):
                # for face_id in frame_info['faces']:
                #     if not best_frame_info['faces'].get(face_id) or frame_info['faces'][face_id]['face_score'] > best_face_info[face_id]['face_score']:
                #         best_face_info[face_id] = frame_info['faces'][face_id]
                if frame_info['avg_score'] > best_frame_info['avg_score'] and len(frame_info['faces']) >= len(best_frame_info['faces']):
                    best_frame_info = frame_info
            else:
                best_frame_info = frame_info
                # best_face_info = frame_info['faces']

            # for d in trackers:
            #     if not no_display:
            #         d = d.astype(np.int32)
            #         cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
            #         if final_faces != []:
            #             cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.75,
            #                         colours[d[4] % 32, :] * 255, 2)
            #             cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #                         (1, 1, 1), 2)
            #         else:
            #             cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.75,
            #                         colours[d[4] % 32, :] * 255, 2)
            c += 1
            # Write the frame into the file '{}.avi'
            if isSaveVideos:
                videoWriter.write(frame)

            if not no_display:
                pass
                # showFrame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                # cv2.imshow("Frame", showFrame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        logger.info(f'Total compute time: {time() - start}')
        logger.info('Prepare to release video ...')
        if isSaveVideos:
            videoWriter.release()
        logger.info('---> Finish release video <---')

        # Post process
        print(f'Key frame at frame {best_frame_info["frame_id"]}')
        # for face_id in best_face_info.keys():
        #     key_frame_score = 0
        #     if best_frame_info["faces"].get(face_id):
        #         key_frame_score = best_frame_info["faces"][face_id]["face_score"]
        #     print(f'Id: {face_id}, Face score of key frame: {key_frame_score}, Best face score: {best_face_info[face_id]["face_score"]} at frame {best_face_info[face_id]["frame_id"]}')
        # key_frame_cam = cv2.VideoCapture(video_name)
        # key_frame_cam.set(cv2.CAP_PROP_POS_FRAMES,best_frame_info['frame_id'])
        # _, key_frame =key_frame_cam.read()

        key_frame = best_frame_info['frame_image']
        key_frame_gray = cv2.cvtColor(key_frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(inference_dir,'key_frame.png'), key_frame)

        for face_id in best_frame_info['faces']:
            best_frame_face_mask = np.zeros_like(key_frame_gray)
            best_frame_face_info = best_frame_info['faces'][face_id]
            best_frame_face_bbox = best_frame_face_info['bbox']
            best_frame_face_landmarks = shape_predictor(key_frame, dlib.rectangle(best_frame_face_bbox[0], best_frame_face_bbox[1], best_frame_face_bbox[2], best_frame_face_bbox[3]))
            best_frame_face_landmarks_points = []
            for point in best_frame_face_landmarks.parts():
                best_frame_face_landmarks_points.append((point.x, point.y))
            best_frame_face_points = np.array(best_frame_face_landmarks_points, dtype=np.int32)
            best_frame_face_convexhull = cv2.convexHull(best_frame_face_points)
            best_frame_face_info["landmarks"] = best_frame_face_points
            best_frame_face_info["convexhull"] = best_frame_face_convexhull
            cv2.fillConvexPoly(best_frame_face_mask, best_frame_face_convexhull, 255)
            # cv2.imshow("Best_frame_face_mask", best_frame_face_mask)
            # cv2.waitKey(0)
            for frame_info in list_frame_info:
                # if frame_info['frame_id'] == best_frame_face_info['frame_id']:
                #     continue
                face_info = frame_info['faces'].get(face_id)
                if not face_info:
                    continue
                face_bbox = face_info['bbox']
                frame_image = frame_info['frame_image']
                frame_image_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
                face_mask = np.zeros_like(frame_image_gray)
                face_landmarks = shape_predictor(frame_image, dlib.rectangle(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]))
                face_landmarks_points = []
                for point in face_landmarks.parts():
                    face_landmarks_points.append((point.x, point.y)) 
                face_points = np.array(face_landmarks_points, dtype=np.int32)
                face_convexhull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(face_mask, face_convexhull, 255)
                # cv2.imshow("Face_mask", face_mask)
                # cv2.waitKey(0)
                dice_similarity_score = np.sum(best_frame_face_mask[face_mask == 255]) * 2.0 / (np.sum(best_frame_face_mask) + np.sum(face_mask))
                face_score = face_info['face_score']
                if dice_similarity_score < 0.9:
                    continue
                total_score = (FACE_SCORE_WEIGHT * face_score + DICE_SIMILARITY_WEIGHT * dice_similarity_score) / (FACE_SCORE_WEIGHT + DICE_SIMILARITY_WEIGHT)
                total_score = face_score
                if (best_face_info.get(face_id) and total_score > best_face_info[face_id]['total_score']) or not best_face_info.get(face_id):
                    best_face_info[face_id] = face_info
                    best_face_info[face_id]['total_score'] = total_score
                    best_face_info[face_id]['frame_image'] = frame_image
                    best_face_info[face_id]['convexhull'] = face_convexhull                        
                    best_face_info[face_id]['landmarks'] = face_points                        
        
        for face_id in best_face_info.keys():
            if best_frame_info["faces"].get(face_id):
                print(f'Id: {face_id}, Best face score: {best_face_info[face_id]["total_score"]} at frame {best_face_info[face_id]["frame_id"]}')            
                face_id_dir = os.path.join(inference_dir, str(face_id))
                mkdir(face_id_dir)
                # Get target face with convexhull
                key_frame_copy = np.copy(key_frame)
                target_convexhull = best_frame_info["faces"][face_id]['convexhull']
                target_bbox = best_frame_info['faces'][face_id]['bbox']
                l, t, r, b = target_bbox
                cv2.imwrite(os.path.join(face_id_dir, 'target.png'), key_frame_copy[t:b, l:r])
                cv2.polylines(key_frame_copy, [target_convexhull], True, (255,0,0), 3)
                cv2.imwrite(os.path.join(face_id_dir, 'target_convexhull.png'), key_frame_copy[t:b, l:r])
                for x, y in best_frame_info["faces"][face_id]['landmarks']:
                    cv2.circle(key_frame_copy, (x, y), 3, (0,0,255), -1)
                cv2.imwrite(os.path.join(face_id_dir, 'target_convexhull_with_landmarks.png'), key_frame_copy[t:b, l:r])
                
                # Get source face with convexhull
                source_image_copy = np.copy(best_face_info[face_id]['frame_image'])
                source_convexhull = best_face_info[face_id]['convexhull']
                source_bbox = best_face_info[face_id]['bbox']
                l, t, r, b = source_bbox
                cv2.imwrite(os.path.join(face_id_dir, 'source.png'), source_image_copy[t:b, l:r])
                cv2.polylines(source_image_copy, [source_convexhull], True, (255,0,0), 3)
                cv2.imwrite(os.path.join(face_id_dir, 'source_convexhull.png'), source_image_copy[t:b, l:r])
                for x, y in best_face_info[face_id]['landmarks']:
                    cv2.circle(source_image_copy, (x, y), 3, (0,0,255), -1)
                cv2.imwrite(os.path.join(face_id_dir, 'source_convexhull_with_landmarks.png'), source_image_copy[t:b, l:r])

        for face_id in best_frame_info['faces']: 
            new_key_frame = np.zeros_like(key_frame)
            # Get landmarks points of source face
            target_bbox = best_frame_info['faces'][face_id]['bbox']
            target_landmarks = shape_predictor(key_frame, dlib.rectangle(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))
            target_landmarks_points = []
            # key_frame_copy = key_frame.copy() # FOR TEST ONLY
            for point in target_landmarks.parts():
                target_landmarks_points.append((point.x, point.y))
                # cv2.circle(key_frame_copy, (point.x, point.y), 3, (0,0,255), -1)
            target_points = np.array(target_landmarks_points, dtype=np.int32)
            target_convexhull = cv2.convexHull(target_points)

            # Test
            # cv2.polylines(key_frame_copy, [target_convexhull], True, (255,0,0), 3)
            # l, t, r, b = target_bbox
            # target_crop_test = key_frame_copy[t:b, l:r]
            # cv2.imshow('Test', target_crop_test)
            # cv2.imwrite('target_crop_problem.png', target_crop_test)
            # cv2.waitKey(0)

            #Delaunay triangulation
            rect = cv2.boundingRect(target_convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(target_landmarks_points)
            target_triangles = subdiv.getTriangleList()
            target_triangles = np.array(target_triangles, dtype=np.int32)
            triangle_indexes = []
            for t in target_triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = np.where((target_points == pt1).all(axis=1))
                index_pt1 = extract_index_nparray(index_pt1)

                index_pt2 = np.where((target_points == pt2).all(axis=1))
                index_pt2 = extract_index_nparray(index_pt2)

                index_pt3 = np.where((target_points == pt3).all(axis=1))
                index_pt3 = extract_index_nparray(index_pt3)

                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle_index = [index_pt1, index_pt2, index_pt3]
                    triangle_indexes.append(triangle_index)

            # Get source image
            source_face_info = best_face_info[face_id]
            # face_cam = cv2.VideoCapture(video_name)
            # face_cam.set(cv2.CAP_PROP_POS_FRAMES,source_face_info['frame_id'])
            # _, source_image = face_cam.read()
            source_image = source_face_info['frame_image']
            # Get landmarks points of source face
            source_bbox = source_face_info['bbox']
            source_landmarks = shape_predictor(source_image, dlib.rectangle(source_bbox[0], source_bbox[1], source_bbox[2], source_bbox[3]))
            source_landmarks_points = []
            for point in source_landmarks.parts():
                source_landmarks_points.append((point.x, point.y))
                # cv2.circle(source_image, (point.x, point.y), 3, (0,0,255), -1)
            source_landmarks_points = np.array(source_landmarks_points, dtype=np.int32)
            
            # Test 
            # convexhull_test = cv2.convexHull(source_landmarks_points)
            # l, t, r, b = source_bbox
            # source_crop_test = source_image[t:b, l:r]
            # cv2.imshow(f'Frame {source_face_info["frame_id"]}', source_crop_test)
            # cv2.imwrite('source_crop_problem.png', source_crop_test)
            # cv2.waitKey(0)
            # cv2.polylines(source_image, [convexhull_test], True, (255,0,0), 3)
            # cv2.imwrite('source_crop_landmarks_problem.png', source_crop_test)
            # cv2.imshow(f'Frame {source_face_info["frame_id"]}', source_crop_test)
            # cv2.waitKey(0)

            # Triangulation of both faces
            for triangle_index in triangle_indexes:
                # Triangulation of source face
                tr1_p1 = source_landmarks_points[triangle_index[0]]
                tr1_p2 = source_landmarks_points[triangle_index[1]]
                tr1_p3 = source_landmarks_points[triangle_index[2]]

                source_triangle = np.array([tr1_p1, tr1_p2, tr1_p3], dtype=np.int32)
                source_rect = cv2.boundingRect(source_triangle)
                (x, y , w, h) = source_rect
                source_cropped_triangle = source_image[y : y + h , x : x + w]
                source_cropped_mask = np.zeros((h, w), dtype=np.uint8)

                source_cropped_points = np.array([[tr1_p1[0] - x, tr1_p1[1] - y],
                                          [tr1_p2[0] - x, tr1_p2[1] - y],
                                          [tr1_p3[0] - x, tr1_p3[1] - y]], dtype=np.int32)
                cv2.fillConvexPoly(source_cropped_mask, source_cropped_points, 255)
                #source_cropped_triangle = cv2.bitwise_and(source_cropped_triangle, source_cropped_triangle, mask=source_cropped_mask)

                # Triangulation of target face
                tr2_p1 = target_landmarks_points[triangle_index[0]]
                tr2_p2 = target_landmarks_points[triangle_index[1]]
                tr2_p3 = target_landmarks_points[triangle_index[2]]

                target_triangle = np.array([tr2_p1, tr2_p2, tr2_p3], dtype=np.int32)
                target_rect = cv2.boundingRect(target_triangle)
                (x, y , w, h) = target_rect
                # target_cropped_triangle = key_frame[y : y + h , x : x + w]
                target_cropped_mask = np.zeros((h, w), dtype=np.uint8)

                target_cropped_points = np.array([[tr2_p1[0] - x, tr2_p1[1] - y],
                                          [tr2_p2[0] - x, tr2_p2[1] - y],
                                          [tr2_p3[0] - x, tr2_p3[1] - y]], dtype=np.int32)
                cv2.fillConvexPoly(target_cropped_mask, target_cropped_points, 255)
                # target_cropped_triangle = cv2.bitwise_and(target_cropped_triangle, target_cropped_triangle, mask=target_cropped_mask)

                # Warp triangles
                source_cropped_points = np.float32(source_cropped_points)
                target_cropped_points = np.float32(target_cropped_points)
                M = cv2.getAffineTransform(source_cropped_points, target_cropped_points)
                warped_triangle = cv2.warpAffine(source_cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=target_cropped_mask)
                
                # Reconstruct target face
                triangle_area = new_key_frame[y : y + h, x : x + w]
                triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)

                # Let's create a mask to remove the lines between the triangles
                _, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                triangle_area = cv2.add(triangle_area, warped_triangle)
                new_key_frame[y : y + h, x : x + w] = triangle_area

            # Face swapped
            key_frame_face_mask = np.zeros_like(key_frame_gray)
            key_frame_head_mask = cv2.fillConvexPoly(key_frame_face_mask, target_convexhull, 255)
            key_frame_face_mask = cv2.bitwise_not(key_frame_head_mask)

            key_frame_head_noface = cv2.bitwise_and(key_frame, key_frame, mask=key_frame_face_mask)
            result = cv2.add(key_frame_head_noface, new_key_frame)

            (x, y, w, h) = cv2.boundingRect(target_convexhull)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
            # test = cv2.seamlessClone(result, key_frame, key_frame_head_mask, center_face2, cv2.MIXED_CLONE)
            key_frame = cv2.seamlessClone(result, key_frame, key_frame_head_mask, center_face2, cv2.MIXED_CLONE)
            # cv2.imwrite('test.png', test)
            # cv2.imshow('Test', test)
            # cv2.waitKey(0)
        # cv2.imshow('Result', key_frame)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(inference_dir,'result.png'), key_frame)

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
    parser.add_argument("--landmarks_model", type=str,
                        help='Path to facial landmarks model', default='models/shape_predictor_68_face_landmarks.dat')                  
    parser.add_argument("--smile_model", type=str,
                        help='Path to smile model', default='models/lenet_smiles.hdf5')
    parser.add_argument('--eye_thresh',
                        help='Threshold for whether eye open',
                        type=float, default=0.1)
    parser.add_argument('--blur_thresh',
                        help='Threshold for whether eye open',
                        type=float, default=-15)        
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
