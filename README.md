# COVIDCam
COVIDCam is a tool which integrates a variety of image analysis techniques in order to monitor and enforce social distancing. It was built using two pre-trained darknet models and OpenCV, to identify the distance between individuals and whether or not they are wearing a mask.

It implements real-time video stream analysis, and a logging feature to log both mask and social distancing violations. These logs are saved and stored, so can be analysed at any time.

Note: The "person-detector" folder would contain the darknet installation with the YOLOv3 object detection model. This is not present here due to size constraints. Therefore, in order to run this project, darknet will need to be built in this location, and the YOLOv3 weights imported.

Made with darknet and YOLOv3, and with the help of the following, which provided the pre-trained models: 
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/
- https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
