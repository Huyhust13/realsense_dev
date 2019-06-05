// Caption image from Intel realsense 
// Huynv 040619

#include <librealsense2/rs.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(){
    // cout<<CV_VERSION<<endl;
    // Khoi tao pipeline, goi du lieu tu camera
    rs2::pipeline pipe;
    // Tao mot cấu hình cho pipline để bắt đầu lấy dữ liệu,
    // không dùng cấu hình mặc định
    rs2::config cfg;
    // Cau hinh cho pipeline
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(cfg);
    while(true)
    {
        rs2::frameset frames;
        // Cho toi tap du lieu tiep theo
        frames = pipe.wait_for_frames();
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
    }
    return 0;
}