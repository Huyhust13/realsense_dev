// Caption image from Intel realsense 
// Huynv 040619

#include <librealsense2/rs.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
    String mode;
    if (argc<2 || (String)argv[1]=="color")
    {
        mode = "color";
    }else if ((String)argv[1]=="infrared")
    {
        mode = "infrared";
    }else if ((String)argv[1]=="depth")
    {
        mode = "depth";
    }
    std::cout<<"mode: "<<mode<<std::endl;
    
    // cout<<CV_VERSION<<endl;
    // Khoi tao pipeline, goi du lieu tu camera
    rs2::pipeline pipe;
    // Tao mot cấu hình cho pipline để bắt đầu lấy dữ liệu,
    // không dùng cấu hình mặc định
    rs2::config cfg;
    // Cau hinh cho pipeline

    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    pipe.start(cfg);
    while(true)
    {
        rs2::frameset frames;
        // Cho toi tap du lieu tiep theo
        frames = pipe.wait_for_frames();
        if (mode == "color")
        {
            rs2::frame color_frame = frames.get_color_frame();
            if(!color_frame){
                std::cout<<"Khong co color frame!!!"<<std::endl;
                break;
            }else
            {
            Mat color(Size(640, 480),
                        CV_8UC3, 
                        (void*)color_frame.get_data(), 
                        Mat::AUTO_STEP);

            namedWindow("Hien thi hinh anh tu Realsense", WINDOW_AUTOSIZE);
            imshow("Hien thi hinh anh tu Realsense", color);
            if (waitKey(1) == 27){
                break;
            }
            }
        }else if (mode == "depth")
        {
            rs2::frame depth_frame = frames.get_depth_frame();
            if(!depth_frame){
                std::cout<<"Khong co depth frame!!!"<<std::endl;
                break;
            }else
            {
            Mat depth(Size(640, 480),
                        CV_8UC1, 
                        (void*)depth_frame.get_data(), 
                        Mat::AUTO_STEP);

            namedWindow("Hien thi hinh anh tu Realsense", WINDOW_AUTOSIZE);
            imshow("Hien thi hinh anh tu Realsense", depth);
            if (waitKey(1) == 27){
                break;
            }
            }
        }else if (mode == "infrared")
        {
            rs2::frame infrared_frame = frames.get_infrared_frame();
            if(!infrared_frame){
                std::cout<<"Khong co infrared frame!!!"<<std::endl;
                break;
            }else
            {
            Mat ir(Size(640, 480),
                        CV_8UC1, 
                        (void*)infrared_frame.get_data(), 
                        Mat::AUTO_STEP);

            namedWindow("Hien thi hinh anh tu Realsense", WINDOW_AUTOSIZE);
            imshow("Hien thi hinh anh tu Realsense", ir);
            if (waitKey(1) == 27){
                break;
            }
            }
        }
    }
    // return 0;
}