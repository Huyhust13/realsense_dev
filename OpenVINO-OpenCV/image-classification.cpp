#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static void show_usage(std::string name){
    std::cerr << "Usage: " << name << " <option(s)> SOURCES\n"
              << "Option: \n"
              << "\t-h, --help\t\tShow this help message\n"
              << "\t-i, --image\t\tPath to image\n"
              << "default: use camera in /dev/video0"
              <<endl;
}

void writeMatToFile(cv::Mat &m, const char *filename)
{
    ofstream fout(filename);
    if (!fout)
    {
        cout << "File Not Opened" << endl;
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

int main(int argc, char* argv[])
{
    // debug = 0 in debug mode;
    bool debug = 0;
    #pragma region load
    // load file model
    string root = "/media/huynv/Data/14.ComputerVision/1.IntelRealsense/realsense_dev/OpenVINO-OpenCV/";
    string labels_file = root + "caffemodel2/MobileNetSSD/labels.txt";
    string model = root + "caffemodel2/MobileNetSSD/MobileNetSSD_deploy.caffemodel";
    string prototxt = root + "caffemodel2/MobileNetSSD/MobileNetSSD_deploy.prototxt";   
    // load names of classes 
    std::vector<string> classes;
    std::ifstream ifs(labels_file.c_str());
    if (!ifs.is_open())
    {
        cout << "error in open file!";
    }
    else
    {
        string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    std::cout << "[INFO] loading model..." << std::endl;
    Net net = readNetFromCaffe(prototxt, model);
    // net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(DNN_TARGET_CPU);

    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, image, blob;

    #pragma endregion load
    
    for (int i = 0; i < argc; i++)
    {std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        }
        if ((arg == "-i") || (arg == "--image"))
        {
            str = argv[i+1];
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            // cout<<str<<endl;
            // return 0; 
        }
        else cap.open("/dev/video0");

    }

    while (waitKey(1) < 0)
    {
        // get frame from video
        cap >> frame;
        // Mat image = imread(root + "images/cat.jpg", IMREAD_COLOR);
        // cv::resize(frame, image, Size(300, 300));
        blob = blobFromImage(frame, 0.007843, Size(300, 300));

        // set the blob as input to the network and perform a forward-pass to
        // obtain our output classification
        net.setInput(blob);
        Mat detection = net.forward();

        double confidenceThreshold = 0.6;
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        if (debug)
        {
            // ghi ra file txt
            // Declare what you need
            cv::FileStorage file("detectionMat.xml", cv::FileStorage::WRITE);
            file << "detectionMatrix" << detectionMat;
            writeMatToFile(detectionMat, "detectionMat.txt");
        }

        int frameWidth = frame.size[0];
        int frameHeight = frame.size[1];

        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidenceThreshold)
            {
                int class_id = int(detectionMat.at<float>(i, 1));
                string label = classes[class_id] + ": ";
                label += to_string(confidence*100);
                cout << label<< endl;

                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) *  frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) *  frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) *  frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
                cv::putText(frame, label, Point(x1, y1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
            }
        }

        double freq = getTickFrequency() / 1000;
        std::vector<double> layersTimes;
        double t = net.getPerfProfile(layersTimes) / freq;
        cout << "[INFO] classification took " << t << " ms" << endl;

        imshow("frame", frame);
    }
    return 0;
}
