#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#include <iostream>
 
using namespace std;
using namespace cv;
using namespace cv::dnn;

void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
      cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

bool operator ! (const Mat&m) { return m.empty(); }

int main() {
  string root = "/media/huynv/Data/14.ComputerVision/1.IntelRealsense/realsense_dev/OpenVINO-OpenCV/";

  Mat image = imread(root + "images/aeroplane.jpg", IMREAD_COLOR);

  string model = root + "caffemodel2/MobileNetSSD/MobileNetSSD_deploy.caffemodel";
  string prototxt = root + "caffemodel2/MobileNetSSD/MobileNetSSD_deploy.prototxt";

  Mat blob;

  string labels_file = "/media/huynv/Data/14.ComputerVision/1.IntelRealsense/realsense_dev/OpenVINO-OpenCV/caffemodel2/MobileNetSSD/labels.txt";
  std::vector<string> classes;

  std::ifstream ifs(labels_file.c_str());
  if (!ifs.is_open())
  {
      cout << "error in open file!";
  }else{
      string line;
      while (std::getline(ifs, line))
      {
          classes.push_back(line);
      }
  }
  cv::resize(image, image, Size(300, 300));
  cout<<"image size: " << image.size<<endl;

  // cout<<"blob row before: " << blob.rows <<endl;
  // cout << "blob col before: " << blob.cols << endl;

  // blobFromImage(image, blob, 1, Size(300, 301
  blob = blobFromImage(image, 1, Size(300, 300));
  // cout<<"blob row: " << blob.rows <<endl;
  // cout << "blob col: " << blob.cols << endl;1
  cout << "blob size: " << blob.size << endl;
  // cout << "blob col: " << blob.cols << endl;1


  std::cout << "[INFO] loading model..." << std::endl;
  Net net = readNetFromCaffe(prototxt, model);
  // net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
  net.setPreferableTarget(DNN_TARGET_CPU);
  
  // set the blob as input to the network and perform a forward-pass to
  // obtain our output classification
  net.setInput(blob);
  Mat detection = net.forward();
  
  // auto detection1 = detection.size[2];

  double confidenceThreshold = 0.6;
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
  cout<<"ashhuu: "<< detection.size[3]<<endl;

  // ghi ra file txt
  const char* filename = "output.txt";
  writeMatToFile(detectionMat,filename);
  cout<<"hang"<<detectionMat.rows<<endl;
  for(int i = 0; i < detectionMat.rows; i++)
  {
      float confidence = detectionMat.at<float>(i, 2);
      if(confidence > confidenceThreshold)
      {
        cout<<"confidence: "<<confidence<<endl;
          // int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
          // int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
          // int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
          // int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
  
          // cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
      }
  }

  double freq = getTickFrequency() / 1000;
  std::vector<double> layersTimes;
  double t = net.getPerfProfile(layersTimes) / freq;
  cout << "[INFO] classification took " << t << " ms" << endl;

  // imshow("image", image);
  // waitKey(0);
  return 0; 
}
