#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <guidedfilter.h>
#include <ctime>
#include "CAP.h"

void calVSMap(const cv::Mat &I, int r, cv::Mat &dR, cv::Mat &dP){
    cv::Mat hsvI, fI;
    I.convertTo(fI, CV_32FC3);
    cv::cvtColor(fI/255.0, hsvI, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_vec;
    cv::split(hsvI, hsv_vec);
    cv::Mat output;
    cv::addWeighted(hsv_vec[1], -0.780245, hsv_vec[2], 0.959710, 0.121779, output);     //TODO
//    cv::addWeighted(hsv_vec[1], -1.2966, hsv_vec[2], 1.0267, 0.1893, output);
    dP = output;
    cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(r, r));
    cv::Mat outputRegion;
    cv::erode(output, outputRegion, se, cv::Point(-1,-1), 1, cv::BORDER_REFLECT);
    dR = outputRegion;
}

std::vector<double> estA(const cv::Mat &img, cv::Mat &Jdark){
    cv::Mat img_norm = img/255.0;
    double n_bright = ceil(img_norm.rows * img_norm.cols * 0.001);
    cv::Mat Jdark_val = Jdark.reshape(1, 1);
    cv::Mat Loc;
    cv::sortIdx(Jdark_val, Loc, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

    cv::Mat Ics = img_norm.reshape(3, 1);
    cv::Mat Acand(1, n_bright, CV_32FC3);
    cv::Mat Amag(1, n_bright, CV_32F);
    for(int i = 0; i < n_bright; i++){
        float b = Ics.at<cv::Vec3f>(0, Loc.at<int>(0, i))[0];
        float g = Ics.at<cv::Vec3f>(0, Loc.at<int>(0, i))[1];
        float r = Ics.at<cv::Vec3f>(0, Loc.at<int>(0, i))[2];

        Acand.at<cv::Vec3f>(0, i)[0] = b;
        Acand.at<cv::Vec3f>(0, i)[1] = g;
        Acand.at<cv::Vec3f>(0, i)[2] = r;
        Amag.at<float>(0, i) = b*b + g*g + r*r;
    }

    cv::Mat Loc2;
    cv::sortIdx(Amag, Loc2, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
    cv::Mat A_arr(1, std::min(20.0, n_bright), CV_32FC3);
    for(int i = 0; i < std::min(20.0, n_bright); i++){
        A_arr.at<cv::Vec3f>(0, i)[0] = Acand.at<cv::Vec3f>(0, Loc2.at<int>(0, i))[0];
        A_arr.at<cv::Vec3f>(0, i)[1] = Acand.at<cv::Vec3f>(0, Loc2.at<int>(0, i))[1];
        A_arr.at<cv::Vec3f>(0, i)[2] = Acand.at<cv::Vec3f>(0, Loc2.at<int>(0, i))[2];
    }

    std::vector<cv::Mat> A_vec;
    cv::split(A_arr, A_vec);
    double max1, max2, max3;
    cv::minMaxLoc(A_vec[0], NULL, &max1);
    cv::minMaxLoc(A_vec[1], NULL, &max2);
    cv::minMaxLoc(A_vec[2], NULL, &max3);

    std::vector<double> A(3);
    A[0] = max1;
    A[1] = max2;
    A[2] = max3;

    return A;
}

void dehazing_CAP(const cv::Mat &Img, cv::Mat &J){
    clock_t start = clock();

    cv::Mat I = Img;
    I.convertTo(I, CV_32FC3);
    cv::Mat dR(I.rows, I.cols, CV_32FC3);
    cv::Mat dP;
    int r = 15;
    double beta = 1.0;

    calVSMap(I, r, dR, dP);

    cv::Mat p = dP.clone();     //TODO
    double eps = 0.001;
    I.convertTo(I, CV_32FC3);
    cv::Mat refineDR = guidedFilter(I/255.0, p, r, eps);

    cv::Mat tR1, tR;
    cv::exp(-beta * refineDR, tR1);
    double t0 = 0.05;
    double t1 = 1;
    cv::Mat t = tR1.clone();
    for(int h = 0; h < t.rows; h++){
        for(int w = 0; w < t.cols; w++){
            if(t.at<float>(h, w) < t0)
                t.at<float>(h, w) = t0;
            else if(t.at<float>(h, w) > t1)
                t.at<float>(h, w) = t1;
        }
    }
    std::vector<cv::Mat> tR_vec;
    tR_vec.push_back(t);
    tR_vec.push_back(t);
    tR_vec.push_back(t);
    cv::merge(tR_vec, tR);

    std::vector<double> a;
    a = estA(I, dR);
    cv::Mat A(I.rows, I.cols, CV_32FC3, cv::Scalar(a[0], a[1], a[2]));

    cv::Mat J1, J2;
    I = I/255.0;
    cv::scaleAdd(A, -1, I, J1);
    cv::divide(J1, tR, J2);
    cv::add(J2, A, J);

    clock_t end = clock();
    std::cout << "Time is " << (float)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

//    cv::imshow("haze", I);
//    cv::imshow("dR", dR);
//    cv::imshow("refineDR", refineDR);
//    cv::imshow("transmission", tR);
//    cv::imshow("dehaze", J);
//    cv::waitKey(0);
}