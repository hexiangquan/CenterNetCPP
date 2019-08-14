//
// Created by he on 19-6-8.
//

#ifndef YOLOV3_DETECT_CENTERNET_H
#define YOLOV3_DETECT_CENTERNET_H


#include <caffe/blob.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

using namespace caffe;  // NOLINT(build/namespaces)


class CenterNet_Detector {
public:
    CenterNet_Detector(const string& model_file,
                    const string& weights_file,
                    const string& mean_file,
                    const string& mean_value);
    CenterNet_Detector(const string& model_file,
                    const string& weights_file );

    std::vector<vector<float> > Detect(const cv::Mat& img);

    vector<vector<float> >  apply_nms(vector<vector<float> > &box,float  thres);
    float  overlap(float x1, float w1, float x2, float w2);
    float  cal_iou(vector<float> &box, vector<float> &truth);


    float biases[18]={12.9866,27.5164, 15.4678,33.0332, 17.7598,38.2623, 21.0792,44.2199, 23.7056,50.8429, 27.3932,58.8125, 31.5247,68.0211, 34.6541,78.5717, 48.3255,102.7408};

    int mask1[3]={6,7,8};
    int mask2[3]={3,4,5};
    int mask3[3]={0,1,2};


    float thresh =0.5;

private:
    void SetMean(const string& mean_file, const string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);


    cv::Mat Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

    double sigmoid(double p);




private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    cv::Mat scalemat;
};










#endif //YOLOV3_DETECT_CENTERNET_H
