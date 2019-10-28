#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES\n"
              << "Option: \n"
              << "\t-h, --help\t\tShow this help message\n"
              << "\t-i, --image\t\tPath to image\n"
              << "\t-w, --webcam\t\tdevice|default /dev/video0\n"
              << "default: use camera in /dev/video0"
              << endl;
}

void writeMatToFile(cv::Mat &m, const char *filename)
{
    ofstream fout(filename);
    if (!fout)
    {
        std::cout << "File Not Opened" << endl;
        return;
    }

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            fout << m.at<float>(i, j) << "\t";
        }
        fout << endl;
    }

    fout.close();
}

cv::Mat faceDetection(cv::Mat frame, cv::dnn::Net net);

int main(int argc, char *argv[])
{
    // debug = 0 in debug mode;
    bool debug = 0;
#pragma region load
    // load file model
    // Thay doi duong dan root toi thu muc trong pc ban
    string root = "../";
    // string labels_file = root + "caffemodel2/MobileNetSSD/labels.txt";
    string model = root + "caffemodel2/Facedetection/fullfacedetection.caffemodel";
    string prototxt = root + "caffemodel2/Facedetection/fullface_deploy.prototxt";
    // load names of classes
    // std::vector<string> classes;
    // std::ifstream ifs(labels_file.c_str());
    // if (!ifs.is_open())
    // {
    //     std::cout << "error in open file!";
    // }
    // else
    // {
    //     string line;
    //     while (std::getline(ifs, line))
    //     {
    //         classes.push_back(line);
    //     }
    // }

    std::cout << "[INFO] loading model..." << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt, model);
    // net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#pragma endregion load

    cv::VideoCapture cap;
    string str, device = "/dev/video0";
    string rtsplink = "http://192.168.17.100:8080/playlist.m3u";
    cv::Mat frame;
    int mode = 0;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if ((arg == "-H") || (arg == "--help"))
        {
            show_usage(argv[0]);
            return 0;
        }
        else if ((arg == "-I") || (arg == "--image"))
        {
            str = argv[i + 1];
            ifstream ifile(str);
            if (!ifile)
                throw("error");
            mode = 1;
            break;
        }
        else if((arg == "-W") || (arg == "--webcam"))
        {
            if (argc > i+1)
            {
                device = argv[i+1];
            }
            cap.open(device);
            if (!cap.isOpened())
            {
                std::cout<<"Cannot open video from "<<device<<endl;
                return 0;
            }
            mode = 2;
            break;
        }
        else if((arg == "-R") || (arg == "--rtsp"))
        {
            if (argc > i+1)
            {
                rtsplink = argv[i+1];
            }
            try
            {
                cap.open(rtsplink, cv::CAP_FFMPEG);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            
            if (!cap.isOpened())
            {
                std::cout<<"Cannot open video from " << rtsplink <<endl;
                return 0;
            }
            mode = 2;
            break;
        }
    }

    switch (mode)
    {
    case 1: // detection in image
        frame = cv::imread(str, cv::IMREAD_COLOR);
        frame = faceDetection(frame, net);   
        cv::namedWindow("frame", cv::WINDOW_NORMAL);
        cv::imshow("frame", frame);
        cv::waitKey(0);
        return 0;
    case 2: // detection in webcam
        while (cv::waitKey(1) < 0)
        {
            // get frame from video
            cap >> frame;
            frame = faceDetection(frame, net);   
            // namedWindow("frame", CV_WINDOW_NORMAL);
            imshow("frame", frame);
            // imshow("image", image);
        }
    default:
        break;
    }
    return 0;

}

cv::Mat faceDetection(cv::Mat frame, cv::dnn::Net net)
{
    cv::Mat frame_resized, blob;
    cv::resize(frame, frame, cv::Size(300, 300),0,0, cv::INTER_LINEAR);
    cv::resize(frame, frame_resized, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);
    // frame_resized = frame;
    blob = cv::dnn::blobFromImage(frame_resized, 0.007843, cv::Size(300, 300), (127.5, 127.5, 127.5));

    // set the blob as input to the network and perform a forward-pass to
    // obtain our output classification
    net.setInput(blob);
    cv::Mat detection = net.forward();

    double confidenceThreshold = 0.6;
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    // ghi ra file txt
    // Declare what you need
    cv::FileStorage file("detectionMat.xml", cv::FileStorage::WRITE);
    file << "detectionMatrix" << detectionMat;
    writeMatToFile(detectionMat, "detectionMat.txt");


    // size of frame resized
    int frame_Width = frame.size[0];
    int frame_Height = frame.size[1];
    cout << "frame_size: " << frame_Width << ":" << frame_Height << endl;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold)
        {
            int class_id = int(detectionMat.at<float>(i, 1));
            string label = to_string(confidence * 100) + "%";
            cout << label << endl;

            int xLeftTop = int(detectionMat.at<float>(i, 3) * frame_Width);
            int yLeftTop = int(detectionMat.at<float>(i, 4) * frame_Height);
            int xRightBottom = int(detectionMat.at<float>(i, 5) * frame_Width);
            int yRightBottom = int(detectionMat.at<float>(i, 6) * frame_Height);
            cv::rectangle(frame, cv::Point(xLeftTop, yLeftTop), cv::Point(xRightBottom, yRightBottom), cv::Scalar(0, 255, 0), 2, 4);
            // cv::circle(frame, cv::Point(xLeftBottom,yLeftBottom), 2, Scalar(255,255,255), 3);
            // cv::circle(frame, cv::Point(xRightTop,yRightTop), 2, Scalar(255,255,255), 3);
            cv::putText(frame, label, cv::Point(xLeftTop, yLeftTop), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            cout << detectionMat.at<float>(i, 3) << ":" << detectionMat.at<float>(i, 4) << endl;
        }
    }

    double freq = cv::getTickFrequency() / 1000;
    std::vector<double> layersTimes;
    double t = net.getPerfProfile(layersTimes) / freq;
    cout << "[INFO] classification took " << t << " ms" << endl;
    return frame;
}