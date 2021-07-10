# Social-distancing-detection-YOLOv3
### Quick overview
![YOLO v3 project workflow image](https://user-images.githubusercontent.com/62443378/125161183-7f320900-e19e-11eb-9b44-b6ec30f2f7f9.png){:height="36px" width="36px"}.


### YOLOv3 explained
#### ![YOLOv3_ Real-Time Object Detection Algorithm (What's New_) _ viso ai - Brave 7_4_2021 7_57_48 PM](https://user-images.githubusercontent.com/62443378/124389295-19f49880-dd04-11eb-928e-3f555d7633a5.png)


#### YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi.

#### After a frame is read from the input image or video stream, it is passed through the blobFromImage function to convert it to an input blob for the neural network. In this process, it scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255. It also resizes the image to the given size of (416, 416) without cropping. Note that we do not perform any mean subtraction here, hence pass [0,0,0] to the mean parameter of the function and keep the swapRB parameter to its default value of 1.
```python
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
```
#### The output blob is then passed in to the network as its input and a forward pass is run to get a list of predicted bounding boxes as the network’s output. These boxes go through a post-processing step in order to filter out the ones with low confidence scores and boex containing objects other than Person.
#### The forward function in OpenCV’s Net class needs the ending layer till which it should run in the network. Since we want to run through the whole network, we need to identify the last layer of the network. We do that by using the function getUnconnectedOutLayers() that gives the names of the unconnected output layers, which are essentially the last layers of the network which we pass to the function as ln
```python
ln = net.getLayerNames()                                    
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  
```
``` python
layerOutputs = net.forward(ln)   
```


### References -: 
#### https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
#### https://viso.ai/deep-learning/yolov3-overview/
