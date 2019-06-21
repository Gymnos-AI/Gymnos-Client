import time
import cv2
import numpy as np
import json
import requests

addr = 'http://10.0.0.245:5000'
test_url = addr + '/yolov3'

# prepare headers for http request
content_type = 'image/type'
headers = {'content-type': content_type}

camera = cv2.VideoCapture(0)
camera_height = 256
camera_width = 256

# allow the camera to warm up
time.sleep(0.1)

FRAMES_TO_DROP = 0
frames_dropped = 0

while True:
	ret, image = camera.read()
	if frames_dropped < FRAMES_TO_DROP:
		frames_dropped += 1
		continue
	frames_dropped = 0

	image = cv2.resize(image, (camera_height, camera_width))

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

	# Press 'q' to quit
	if cv2.waitKey(1) == ord('q'):
		break
