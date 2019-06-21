from picamera.array import PiRGBArray
from picamera import PiCamera
import requests
import json
import cv2
import time
import numpy as np

addr = 'http://10.0.0.191:5000'
test_url = addr + '/yolov3'

# prepare headers for http request
content_type = 'image/type'
headers = {'content-type': content_type}

# setup the Pi camera
# initialize the HOG descriptor/person detector
IM_WIDTH = 256
IM_HEIGHT = 256
camera = PiCamera()
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 32

# allow the camera to warm up
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# Retrieve frame as numpy array
	image = frame.array
	
	# encode and post frame to the server
	_, img_encoded = cv2.imencode('.jpg', image)
	# send http request with image and receive response
	response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
	# response = requests.get(test_url)
	
	# decode response
	list_of_coords = np.array(json.loads(response.text)['coords'])
	print(list_of_coords)
	
	for (topX, leftY, bottomX, rightY) in list_of_coords:
		cv2.rectangle(image, (topX, leftY), (bottomX, rightY), (0, 0, 255), 2)
	
	# show the output images
	cv2.imshow("Video Feed", image)
    
    # clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	# Press 'q' to quit
	if cv2.waitKey(1) == ord('q'):
		break
