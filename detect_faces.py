# import the necessary packages
import numpy as np
import argparse
import cv2
import os

def verify_model_files(prototxt_path, model_path):
    """Verify model files exist and are not empty"""
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if os.path.getsize(model_path) < 1000:  # Arbitrary small size
        raise ValueError(f"Model file seems too small: {model_path}")

def load_model(prototxt_path, model_path):
    """Load the Caffe model with error handling"""
    print("[INFO] Loading model...")
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net
    except cv2.error as e:
        if "BatchNormLayerImpl" in str(e):
            print("[ERROR] BatchNorm layer error. This typically means the model file is corrupted or incomplete.")
            print("[ERROR] Please ensure you're using the correct model file and it was downloaded completely.")
        raise

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Verify files exist
    verify_model_files(args["prototxt"], args["model"])

    # Load the model
    net = load_model(args["prototxt"], args["model"])

    # load the input image
    image = cv2.imread(args["image"])
    if image is None:
        raise ValueError(f"Could not load image: {args['image']}")

    # get dimensions and create a blob
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections
    print("[INFO] computing object detections...")
    net.setInput(blob)
    try:
        detections = net.forward()
    except cv2.error as e:
        print(f"[ERROR] Detection failed: {str(e)}")
        raise

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                         (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {str(e)}")