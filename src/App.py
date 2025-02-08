import os
import logging
import sys

import numpy as np
from numpy import ndarray
import cv2
from functools import cache

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import  Response, JSONResponse
from typing import Union

from faces.detector import FacesDetector

# Model path for face detection (adjustable via environment variable)
MODEL_PATH: str = os.getenv("MODEL_PATH", "face_model.onnx")

# Expected media type for uploaded images
MEDIA_TYPE: str = "image/jpg"

# Image extension used for encoding
IMG_EXT: str = ".jpg"

DEFAULT_CONFIDENCE_THRESHOLD: int  = 0.3

app = FastAPI()

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(stream=sys.stdout, level=LOGLEVEL)
logging.info("loaded new version with acr")

async def get_boxes(file: bytes, conf_threshold:float = DEFAULT_CONFIDENCE_THRESHOLD) -> ndarray:
    img_np = np.frombuffer(file, np.uint8)
    srcimg = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if srcimg is None:
        raise TypeError("Unsupported data type.")
    MODEL.set_conf_threshold = conf_threshold
    bboxes, _, _, _ = MODEL.detect(srcimg)
    print(bboxes)
    return bboxes.tobytes()

@cache
def load_cached_model(model_path: str, conf_threshold:float = DEFAULT_CONFIDENCE_THRESHOLD) -> FacesDetector:
    return FacesDetector(model_path, conf_thres=conf_threshold)

MODEL = load_cached_model(MODEL_PATH)

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Unsupported file type. Only images allowed."}, status_code=415)
    try:
        # Read image data from the uploaded file
        data = await file.read()
        boxes = await get_boxes(data, conf_threshold)
        return Response(content=boxes, media_type='application/octet-stream')
    except Exception as e:
        logging.exception(f"Error handling image upload: {e}")
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)