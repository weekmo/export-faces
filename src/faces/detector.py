import cv2
from cv2.typing import MatLike
import numpy as np
from numpy import ndarray
import logging
from typing import List, Union, Optional, Dict, Tuple

class FacesDetector:
    CONF_THRESHOLD = 0.4
    def __init__(self, model_path: Union[str, bytearray] = None, conf_thres: float =CONF_THRESHOLD, iou_thres: float = 0.5) -> None:
        self.conf_threshold = self._validate_threshold(conf_thres, "conf_thres")
        self.iou_threshold = self._validate_threshold(iou_thres, "iou_thres")
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        self.net: Optional[cv2.dnn.Net] = None        
        # Initialize model
        if model_path is not None:
            self._read_model(model_path)

        self.INPUT_HEIGHT = 640
        self.INPUT_WIDTH = 640
        self.REG_MAX = 16

        self.project = np.arange(self.REG_MAX)
        self.strides = (8, 16, 32)
        self.feats_hw = [(self.INPUT_HEIGHT // stride, self.INPUT_WIDTH // stride) for stride in self.strides]
        self.anchors = self._make_anchors(self.feats_hw)
    
    def _validate_threshold(self, threshold: float, name: str) -> float:
        """Validates confidence or IOU threshold."""
        if not isinstance(threshold, (float, int)):
            raise TypeError(f"{name} should be a float or int")

        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} should be between 0 and 1")

        return float(threshold)
    
    def _read_model(self, model_path: Union[str, bytearray]) -> None:
        try:
            if isinstance(model_path, str):
                self.net = cv2.dnn.readNet(model_path)
            elif isinstance(model_path, bytearray):
                self.net = cv2.dnn.readNetFromONNX(buffer=model_path)
            else:
                raise TypeError("model_path must be a string or bytearray")
        except Exception as e:
            raise ValueError(f"Error reading model: {e}")

    def set_conf_threshold(self, conf_threshold: float = CONF_THRESHOLD) -> None:
        self.conf_threshold = self._validate_threshold(conf_threshold, "conf_thres")

    def set_iou_threshold(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = self._validate_threshold(iou_threshold, "iou_thres")
    
    def _make_anchors(self, feats_hw: list, grid_cell_offset: float = 0.5) -> dict:
        anchor_points: Dict[int, ndarray] = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def _softmax(self, x: ndarray, axis: int = 1) -> ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum
    
    def _resize_image(self, srcimg: ndarray, keep_ratio: bool = True) -> tuple[ndarray, int, int, int, int]:
        padding_top, padding_left, new_width, new_height,  = 0, 0, self.INPUT_WIDTH, self.INPUT_HEIGHT
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                new_height, new_width = self.INPUT_HEIGHT, int(self.INPUT_WIDTH / hw_scale)
                img = cv2.resize(srcimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
                padding_left = int((self.INPUT_WIDTH - new_width) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padding_left, self.INPUT_WIDTH - new_width - padding_left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                new_height, new_width = int(self.INPUT_HEIGHT * hw_scale), self.INPUT_WIDTH
                img = cv2.resize(srcimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
                padding_top = int((self.INPUT_HEIGHT - new_height) * 0.5)
                img = cv2.copyMakeBorder(img, padding_top, self.INPUT_HEIGHT - new_height - padding_top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.INPUT_WIDTH, self.INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        return img, new_height, new_width, padding_top, padding_left
    
    def _distance2bbox(self, points: ndarray, distance: ndarray, max_shape: tuple[int, int] = None) -> ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            y_max, x_max = max_shape
            x1 = np.clip(x1, 0, x_max)
            y1 = np.clip(y1, 0, y_max)
            x2 = np.clip(x2, 0, x_max)
            y2 = np.clip(y2, 0, y_max)
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _post_process(self, predictions: List[ndarray], scale_h: float, scale_w: float, padh: int, padw: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        bboxes: List[ndarray] = []
        scores: List[ndarray] = []
        landmarks: List[ndarray] = []

        for i, pred in enumerate(predictions):
            # print(f"pred {i} is {pred.shape}")
            stride = int(self.INPUT_HEIGHT/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.REG_MAX * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.REG_MAX * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.REG_MAX)
            bbox_pred = self._softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self._distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3])) # Sigmoid function

            bbox -= np.array([[padw, padh, padw, padh]]) 
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask] 
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        landmarks = landmarks[mask]

        if bboxes_wh.size > 0:
            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                mlvl_bboxes = bboxes_wh[indices]
                confidences = confidences[indices]
                class_ids = class_ids[indices]
                landmarks = landmarks[indices]
                return mlvl_bboxes, confidences, class_ids, landmarks
            else:
                logging.info('No face detected after NMS')
                return np.array([]), np.array([]), np.array([]), np.array([])
        else:
            logging.info('No face detected after confidence thresholding')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def detect(self, srcimg: Union[MatLike, ndarray]) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        if self.net is None:
            raise RuntimeError("Model must be loaded before detection.")
        
        input_img, new_height, new_width, padding_top, padding_left = self._resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/new_height, srcimg.shape[1]/new_width
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self._post_process(outputs, scale_h, scale_w, padding_top, padding_left)
        return det_bboxes, det_conf, det_classid, landmarks
    
    @staticmethod
    def draw_detections(srcimg: Union[ndarray, MatLike], boxes: ndarray, scores: ndarray, kpts: ndarray) -> Union[ndarray, MatLike]:
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(srcimg, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(srcimg, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(srcimg, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                cv2.putText(srcimg, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        return srcimg
    
    @staticmethod
    def pixelate(srcimg: Union[MatLike, ndarray], boxes: ndarray) -> Union[MatLike, ndarray]:
        for box in boxes:
            x, y, w, h = box.astype(int)
            # Draw pixels
            f = srcimg[y:y + h, x:x + w]
            f = cv2.resize(f, (7, 7), interpolation=cv2.INTER_NEAREST)
            try:
                srcimg[y:y + h, x:x + w] = cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                logging.error(f"Found error: {e} !")
                continue
        return srcimg

    @staticmethod
    def get_faces(srcimg: Union[MatLike, ndarray], boxes: ndarray, padding: int = 50):
        faces: List[MatLike] = []
        for bbox in boxes:
            (x, y, w, h) = bbox.astype(int)
            x -= padding
            y -= padding
            w += padding
            h += padding
            face = srcimg[y:y + h, x:x + w]
            if face.size > 0:
                faces.append(face)
        return faces
