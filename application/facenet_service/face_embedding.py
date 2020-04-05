# Compute the 128D vector that describes the face in img identified by
# shape.  In general, if two face descriptor vectors have a Euclidean
# distance between them less than 0.6 then they are from the same
# person, otherwise they are from different people. 
from __future__ import division
from __future__ import print_function

import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from application.facenet_src import facenet
from application.align_mtcnn_service.align_mtcnn import AlignMTCNN
from application.rest.request_mapper.faces_request import FacesRequest

CUR_DIR = Path(__file__).resolve().parent


class FaceEmbedding:

    def __init__(self):
        self.data_path = CUR_DIR / 'people'
        self.model_dir = CUR_DIR / 'model/20180402-114759'
        self.alignMTCNN = AlignMTCNN()
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.embedding_size = None
        self.image_size = 160
        self.threshold = [0.8, 0.8, 0.8]
        self.factor = 0.7
        self.minsize = 20
        self.margin = 44
        self.detect_multiple_faces = False
        self.sess = None

    def compare(self, to_compare_photos: List[FacesRequest]):
        try:
            os.makedirs(self.data_path / to_compare_photos[0].chat_id, exist_ok=True)
            file_pathes = []
            for request_photo in to_compare_photos:
                photo_bytes = base64.b64decode(request_photo.base64_string)
                photo_file_path = self.data_path / request_photo.chat_id / f'{request_photo.type}.jpg'
                file_pathes.append(photo_file_path)
                with open(photo_file_path, 'wb') as f:
                    f.write(photo_bytes)
            return self.compare_two_photos(file_pathes)
        except TypeError:
            return -1.0

    def compare_two_photos(self, file_pathes: List[Path]):
        photo1_filename = file_pathes[0]
        photo2_filename = file_pathes[1]
        photo1 = self.convert_compared_photo_to_embedding(photo1_filename)
        photo2 = self.convert_compared_photo_to_embedding(photo2_filename)
        dist = self.ecuclidean_distance(photo1, photo2)
        percent = (1 - (dist / 4)) * 100
        return percent

    def convert_compared_photo_to_embedding(self, filename: Path):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.sess = sess
                # Load the model
                facenet.load_model(self.model_dir)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.images_placeholder = tf.image.resize_images(images_placeholder, (self.image_size, self.image_size))
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]
                img = cv2.imread(str(filename), 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bounding_boxes, points = self.alignMTCNN.get_bounding_boxes(image=img)
                faces = self.get_faces(img, bounding_boxes, points, str(filename))
                return faces

    def get_faces(self, img, bounding_boxes, points, filename):
        faces = []
        nrof_faces = bounding_boxes.shape[0]
        print("No. of faces detected: {}".format(nrof_faces))
        if nrof_faces == 0:
            raise TypeError("No face found")

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'name': filename, 'rect': [bb[0], bb[1], bb[2], bb[3]],
                              'embedding': self.get_embedding(prewhitened)})

        return faces

    def get_embedding(self, processed_img):
        reshaped = processed_img.reshape(-1, self.image_size, self.image_size, 3)
        feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
        feature_vector = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return feature_vector

    @staticmethod
    def ecuclidean_distance(grandpa, son):
        dist = np.sqrt(np.sum(np.square(np.subtract(grandpa[0]['embedding'], son[0]['embedding']))))
        return dist
