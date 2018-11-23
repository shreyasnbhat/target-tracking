import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from depthmap.monodepth_simple import generate_depth_map_frame, params, init
from tracker.pyimagesearch.centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--file", required=True,
                help="path to video")
ap.add_argument("-d", "--fcap", type=int, default=50,
                help="Frame cap")
args = vars(ap.parse_args())

ct = CentroidTracker(maxDisappeared=args["fcap"])
(H, W) = (None, None)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["file"])
time.sleep(2.0)
fps = FPS().start()
init()
framecnt = 0
depth_map = None

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    _, frame = vs.read()
    frame = cv2.resize(frame,(512,256))

    if framecnt % 20 == 0:
        depth_map = generate_depth_map_frame(params, frame, args["file"])
        print("Depth Map Generated for frame", framecnt)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append(box.astype("int"))

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            y = startY - 15 if startY - 15 > 15 else startY + 15

    # update our centroid tracker using the computed set of bounding
    # box rectangles

    objects = ct.update(rects, depth_map)

    for objectID in ct.history:
        prev = None
        for centroid in ct.history[objectID]:
            cv2.circle(frame, (centroid[0], centroid[1]), 2, ct.colours[objectID])
            if prev is not None:
                cv2.line(frame, (prev[0], prev[1]), (centroid[0], centroid[1]), ct.colours[objectID], 1)
            prev = centroid



    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = str(ct.depth[objectID][-1])
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

    framecnt += 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()
