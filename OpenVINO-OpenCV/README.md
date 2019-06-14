# Using OpenVINO with OpenCV

This directory contains code for image classification using OpenVINO with OpenCV.

## For C++

### How to compile the code

If you don't have OpenCV installed globally, then Specify the OpenCV_DIR in CMakeLists.txt file. Then,

```
cmake .
make
```
## How to Run the code

### C++

`./image-classification`

### Python

`python image-classification.py`

## Tham khảo model tại:
https://github.com/HoldenCaulfieldRye/caffe

Download model tại: https://github.com/HoldenCaulfieldRye/caffe/tree/master/models/bvlc_reference_rcnn_ilsvrc13 

More SSD caffe model: https://github.com/weiliu89/caffe/tree/ssd



## Tham khảo:
[1] https://davidmatablog.wordpress.com/2017/12/05/real-time-object-recognition-with-opencv-python-deep-learning-caffe-model/  
[2] https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

## Các vấn đề:
1. `blob = blobFromImage(frame, 0.007843, Size(300, 300));`  
- frame là ảnh đầu vào, không yêu cầu kích thước
- 0.007843 là tỉ số scale, không hiểu sao khi để là 1 thì k nhận dạng được, để 0.007843 thì nhận dạng được --> **Ý nghĩa và ảnh hưởng của thông số này là gì??**
- Size(300,300) là kích thước ảnh đầu vào của mạng

## Update 130619
* Vẽ boudingbox đang bị lệch với kích thước frame thật
* Cần bổ sung thêm về các chế độ input