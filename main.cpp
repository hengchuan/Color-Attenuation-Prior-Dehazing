//
// Created by liuyang on 17-8-27.
//

#include "CAP.h"
#include "opencv2/opencv.hpp"

int main(){
    cv::Mat I = cv::imread("tree2.png", -1);
    cv::Mat J;

    dehazing_CAP(I, J);

    cv::imshow("haze", I);
    cv::imshow("dehaze", J);
    cv::waitKey(0);
    return 0;
}