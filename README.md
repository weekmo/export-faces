# Face Detection API

## Description
This project provides a FastAPI-based API for detecting faces in images using a pre-trained ONNX model. It includes endpoints for uploading images and receiving detected face bounding boxes.

## Features
- Detect faces in images
- Adjustable confidence threshold for detection
- Supports image uploads via FastAPI
- Provides bounding boxes for detected faces

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:weekmo/export-faces.git
    ```
2. Navigate to the project directory:
    ```bash
    cd export-faces
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:
    ```bash
    hypercorn src.App:app --reload
    ```
2. Use the `/detect` endpoint to upload an image and get face detections:
    ```bash
    curl -X POST "http://127.0.0.1:8000/detect" -F "file=@path/to/your/image.jpg"
    ```

## Running Tests

To run tests, use the following command:

```bash
python -m unittest discover tests
