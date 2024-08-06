import tensorflow as tf
import numpy as np
from util import get_midpoint, get_distance
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class DetectorTF:
    def __init__(self, path_to_ckpt, memory_fraction=0.45):
        try:
            self.path_to_ckpt = path_to_ckpt
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options), graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        except:
            raise Exception("Could not load the model.")

    def detect(self, image, threshold=0.2, dist_threshold=50, pixel_threshold=-1, size_count=-1):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
        im_height, im_width = image.shape[:2]
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
                int(boxes[0, i, 0] * im_height),
                int(boxes[0, i, 1] * im_width),
                int(boxes[0, i, 2] * im_height),
                int(boxes[0, i, 3] * im_width)
            )
        scores = scores[0].tolist()
        classes = [int(x) for x in classes[0].tolist()]
        foundDefect = False
        nDefects = 0
        defect_count = 0
        for idx, obj_class in enumerate(classes):
            if obj_class > 0:
                if scores[idx] >= threshold:
                    nDefects += 1
                    box = boxes_list[idx]
                    if not foundDefect:
                        mid_p = get_midpoint(box)
                        dist = get_distance(mid_p, (int(im_width/2), int(im_height/2)))
                        if int(dist) <= dist_threshold:
                            foundDefect = True
                            area = (box[3] - box[1])*(box[2] - box[0])
                            if int(area) <= int(pixel_threshold):
                                defect_count += 1
        if defect_count == nDefects:
            if defect_count <= size_count:
                foundDefect = False
        if not foundDefect:
            main_img = np.zeros((im_width, im_width*2, 3), dtype=np.uint8)
            main_img[0:im_height, 0:im_width] = image
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                "defect",
                use_normalized_coordinates=True,
                line_thickness=1,
                max_boxes_to_draw=100,
                min_score_thresh=0.2)
            main_img[0:im_height, im_width:im_width*2] = image
        else:
            main_img = np.zeros((im_width, im_width*2, 3), dtype=np.uint8)
            main_img[0:im_height, 0:im_width] = image
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                "defect",
                max_boxes_to_draw=100,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=0.2)
            main_img[0:im_height, im_width:im_width*2] = image

        return foundDefect, main_img


