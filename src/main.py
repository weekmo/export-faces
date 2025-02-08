import cv2
import uuid
from cv2.typing import MatLike
from faces.detector import FacesDetector
import requests
import numpy as np

URL='http://127.0.0.1:8000/detect'

def try_images(img:MatLike):
    bboxes, _, _, _ = MODEL.detect(img)
    dsimg = FacesDetector.get_faces(img, bboxes,20)
    for img in dsimg:
        cv2.imwrite(f'data/output/{uuid.uuid4()}.jpg', img)

def try_videos(video_path:str):
    skip_frams = 100
    frame_count = 50
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_frams == 0:
            try_images(frame)
            frame_count = 0
        frame_count +=1

if __name__ == "__main__":
    img_path = 'data/img1.png'
    with open(img_path, "rb") as file:
        files = {"file": (file.name, file, "image/jpeg")}
        response = requests.post(URL, files=files)
        file.seek(0)
        np_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        print(response.status_code)
        boxes = np.frombuffer(response.content).reshape(-1, 4)
        print(boxes)
        faces = FacesDetector.get_faces(image, boxes)
        for img in faces:
            cv2.imwrite(f'data/output/{uuid.uuid4()}.jpg', img)

        