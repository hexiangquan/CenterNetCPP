//
// Created by he on 19-6-8.
//

#include "centernet.h"


CenterNet_Detector::CenterNet_Detector(const string& model_file,
                                 const string& weights_file,
                                 const string& mean_file,
                                 const string& mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 3) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}


CenterNet_Detector::CenterNet_Detector(const string& model_file,
                                 const string& weights_file
) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
   // Caffe::set_mode(Caffe::GPU);
#endif
    Caffe::set_mode(Caffe::CPU);
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 3) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());


}

std::vector<vector<float> > CenterNet_Detector::Detect(const cv::Mat& img) {
    std::vector<vector<float> > rlt;
    Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    cv::Mat tm=Preprocess(img, &input_channels);

    net_->Forward();


    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob0 = net_->output_blobs()[0];
    const float* result0 = result_blob0->cpu_data();
    std::cout<<result_blob0->shape_string()<<std::endl;
    printf("%f %f %f %f %f %f\n",result0[0],result0[1],result0[2],result0[3],result0[4],result0[5]);
    vector<int> s_sz_shape=result_blob0->shape();

//    vector<float>   hm_sigmod_rlt;
//    for(int i=0;i<s_sz_shape[0]*s_sz_shape[1]*s_sz_shape[2]*s_sz_shape[3];i++)
//    {
//        hm_sigmod_rlt.push_back(sigmoid(result0[i]));
//    }
//    //_nms_





    Blob<float>* result_blob1 = net_->output_blobs()[1];
    const float* result1 = result_blob1->cpu_data();
    std::cout<<result_blob1->shape_string()<<std::endl;
    printf("%f %f %f %f %f %f\n",result1[0],result1[1],result1[2],result1[3],result1[4],result1[5]);

    vector<int> m_sz_shape=result_blob1->shape();





//hm layer
    Blob<float>* result_blob2 = net_->output_blobs()[2];
    const float* result2 = result_blob2->cpu_data();
    std::cout<<result_blob2->shape_string()<<std::endl;
    printf("sp:%f %f %f %f %f %f\n",result2[0],result2[1],result2[2],result2[3],result2[4],result2[5]);
    std::cout<<"fuck\n"<<std::endl;

    boost::shared_ptr<caffe::Blob<float>> layerData = net_->blob_by_name("conv_blob53");  // 获得指定层的输出
    const float* pstart = layerData->cpu_data(); // res5_6->cpu_data()返回的是多维数据（数组）

    vector<int> hm_shape=result_blob2->shape();


    printf("sp:%f %f %f %f %f %f\n",pstart[0],pstart[1],pstart[2],pstart[3],pstart[4],pstart[5]);
    //get max index
//
//    std::vector<int> pindex;

    //vector<int>  index_max;
    vector<vector<float>>  fscore_max;
    for(int i=0;i<hm_shape[0];i++)
        for(int j=0;j<hm_shape[1];j++)//class
        {
            for(int k=0;k<hm_shape[2]*hm_shape[3];k++)
                if(pstart[i*hm_shape[1]*hm_shape[2]*hm_shape[3]+j*hm_shape[2]*hm_shape[3]+k]==result2[i*hm_shape[1]*hm_shape[2]*hm_shape[3]+j*hm_shape[2]*hm_shape[3]+k])
                {
                    vector<float> inx;

                    inx.push_back(i*hm_shape[1]*hm_shape[2]*hm_shape[3]+j*hm_shape[2]*hm_shape[3]+k);
                    inx.push_back(pstart[i*hm_shape[1]*hm_shape[2]*hm_shape[3]+j*hm_shape[2]*hm_shape[3]+k]);
                    fscore_max.push_back(inx);

                }

        }
    std::sort(fscore_max.begin(), fscore_max.end(),[](const std::vector<float>& a, const std::vector<float>& b){ return a[1] > b[1];});
    // get top 100
    int iters=std::min<int>(fscore_max.size(),100);
    int only_threshbox=0;
    for(int i=0;i<iters;i++)
    {
        fscore_max[i][1]= sigmoid(fscore_max[i][1]);

        if(fscore_max[i][1]<thresh)
        {
            break;
        }
        only_threshbox++;
    }
    // batch =1
    vector<vector<float>> boxes;
    for(int i=0;i<only_threshbox;i++)
    {
        vector<float> box;
        int index=((int)fscore_max[i][0])/(hm_shape[2]*hm_shape[3]);
        int center_index=((int)fscore_max[i][0])%(hm_shape[2]*hm_shape[3])-hm_shape[3];
        int cls=index;

        float xs=center_index%hm_shape[3];
        float ys=center_index/hm_shape[2];
        //reg batch 1
        xs+=result0[(int)(((int)ys)*hm_shape[3]+xs)];
        ys+=result0[(int)(hm_shape[3]*hm_shape[2]+((int)ys)*hm_shape[3]+xs)];

        float w= result1[(int)(((int)ys)*hm_shape[3]+xs)];
        float h= result1[(int)(hm_shape[3]*hm_shape[2]+((int)ys)*hm_shape[3]+xs)];


        box.push_back((float)cls);
        box.push_back((float)fscore_max[i][1]);

        box.push_back((float)(xs-w/2.0));
        box.push_back((float)(ys-h/2.0));
        box.push_back((float)(xs+w/2.0));
        box.push_back((float)(ys+h/2.0));

        boxes.push_back(box);

    }

    for(int i=0;i<boxes.size();i++)
    {
        cv::rectangle(tm,cv::Point((int)(boxes[i][2]*4),(int)(boxes[i][3]*4)),cv::Point((int)(boxes[i][4]*4),(int)(boxes[i][5]*4)),cv::Scalar(0,0,255),1,1,0);
    }

    cv::imshow("image", tm);
    cv::waitKey(0);


    std::cout<<"fuck line:"<<__LINE__<<std::endl;

    return rlt;
}

/* Load the mean file in binaryproto format. */
void CenterNet_Detector::SetMean(const string& mean_file, const string& mean_value) {

    cv::Scalar channel_mean;

    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
                                  "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
        scalemat = cv::Mat(input_geometry_, mean.type(), 1/127.5);
    }

    if (!mean_value.empty()) {

        CHECK(mean_file.empty()) <<"Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {

            float value = std::atof(item.c_str());
            values.push_back(value);

        }
        CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }

        cv::merge(channels, mean_);
        scalemat = cv::Mat(input_geometry_, mean_.type(), 1/127.5);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CenterNet_Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

}


cv::Mat CenterNet_Detector::Preprocess(const cv::Mat& img,
                                 std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
//  if (img.channels() == 3 && num_channels_ == 1)
//    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 1)
//    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 3)
//    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//  else if (img.channels() == 1 && num_channels_ == 3)
//    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//  else
//    sample = img;
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
//  Mat(int rows, int cols, int type);
//  Mat(Size size, int type);
//  //! constucts 2D matrix and fills it with the specified value _s.
//  Mat(int rows, int cols, int type, const Scalar& s);

    cv::Mat sample_resizeds =cv::Mat((int)(input_geometry_.height),input_geometry_.width,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat sample_resized;
    cv::Mat sample_resized_raw;
    int maxv=std::max<int>(img.cols,img.rows);
    float scale=maxv/(float)input_geometry_.width;

    if (sample.size() != input_geometry_)
    {
        cv::resize(img, sample_resized, cv::Size( input_geometry_.width,(int)412));
        cv::resize(img, sample_resized_raw, cv::Size( input_geometry_.width,(int)(input_geometry_.height)));
    }else
        sample_resized = sample;

    cv::Mat imageROI = sample_resizeds(cv::Rect( 0,(input_geometry_.height-(int)412)/2, sample_resized.cols, sample_resized.rows));	//450，20为自定义起始点坐标
    //【3】加载掩模（必须是灰度图）
    cv::Mat mask;
    cv::Mat img11(sample_resized.rows,sample_resized.cols,CV_8UC1,cv::Scalar(255));
    cvtColor(sample_resized,mask,CV_BGR2GRAY);

    //【4】将掩模复制到ROI
    sample_resized.copyTo(imageROI, img11);

    cv::Mat samplesss=sample_resizeds.clone();


  //cv::imshow("test",img);
//    cv::imshow("test",sample_resizeds);
//   cv::waitKey(0);


//    cv::cvtColor(sample_resizeds, sample_resizeds, cv::COLOR_BGR2RGB);
//    cv::cvtColor(sample_resized_raw, sample_resized_raw, cv::COLOR_BGR2RGB);

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resizeds.convertTo(sample_float, CV_32FC3);
    else
        sample_resizeds.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    // cv::subtract(sample_float, mean_, sample_normalized);
    //cv::multiply(sample_normalized, scalemat, sample_normalized);
    sample_normalized=sample_float*1/255.0;
//    cv::Mat sample_normalized1=sample_normalized-0.225;
    // std::cout << sample_normalized << std::endl;
// exit(0);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */

    std::vector<cv::Mat> tem_input_channels;

//    cv::Mat sp=cv::Mat::ones(cv::Size(512,512),CV_32FC3);
    cv::split(sample_normalized,tem_input_channels);
    cv::Mat op0=(tem_input_channels[0]-0.485)/0.229;
    cv::Mat op1=(tem_input_channels[0]-0.456)/0.224;
    cv::Mat op2=(tem_input_channels[0]-0.406)/0.225;
    memcpy(input_channels->at(0).data,op0.data,input_geometry_.height*input_geometry_.width*4);
    memcpy(input_channels->at(1).data,op1.data,input_geometry_.height*input_geometry_.width*4);
    memcpy(input_channels->at(2).data,op2.data,input_geometry_.height*input_geometry_.width*4);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";


    return samplesss;
}

double CenterNet_Detector::sigmoid(double p){
    return 1.0 / (1 + exp(-p * 1.0));
}
// vector<vector<float> > detections;

float CenterNet_Detector::overlap(float x1, float w1, float x2, float w2)
{
    float left = std::max(x1 - w1 / 2.0, x2 - w2 / 2.0);
    float right = std::min(x1 + w1 / 2.0, x2 + w2 / 2.0);
    return right - left;
}

float CenterNet_Detector::cal_iou(vector<float> &box, vector<float> &truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if (w < 0 || h < 0)
        return 0;

    float inter_area = w * h;
    float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
    return inter_area * 1.0 / union_area;
}





vector<vector<float> > CenterNet_Detector::apply_nms(vector<vector<float> > &box,float  thres)
{
    vector<vector<float> > rlt;
    if (box.empty())
        return vector<vector<float> >();

    std::sort(box.begin(), box.end(),[](const std::vector<float>& a, const std::vector<float>& b){ return a[7] > b[7];});
    std::vector<int> pindex;

    for(int i=0;i<box.size();i++)
    {

        if(std::find(pindex.begin(),pindex.end(),i)!=pindex.end())
        {
            continue;
            //yes
        }
        vector<float> truth =   box[i];
        for(int j=i+1;j<box.size();j++)
        {

            if(std::find(pindex.begin(),pindex.end(),j)!=pindex.end())
            {
                continue;
                //yes
            }

            vector<float> lbox =   box[j];

            float iou = cal_iou(lbox, truth);
            if(iou >= thres)
                pindex.push_back(j);//p[j] = 1

        }
    }

    for(int i=0;i<box.size();i++)
    {
        if(std::find(pindex.begin(),pindex.end(),i)==pindex.end())
        {
            rlt.push_back(box[i]);
        }
    }

    return rlt;
}

/*
 *
 * def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue

        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1

    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res



def get_yolo_detections2(box,thresh,classes,n,biases,mask,w1,h1,det):
    bk=box.shape[0]
    ck=box.shape[1]
    wk=box.shape[2]
    hk=box.shape[3]
    boxes = list()
    for bi in range(bk):
        for wi in range(wk):
            for hi in range(hk):
                for ni in range(n):
                    boxone = list();
                    objectness = sigmoid(box[bi,4+(4+ classes+1)*ni,wi,hi])
                    if objectness  > thresh:
                        claess=[]
                        for class_i in range(classes):
                            prob  = objectness*sigmoid(box[bi,class_i+5+ni*(classes+1+4),wi,hi]);
                            if(prob>thresh):
                                claess.append(prob)
                            else:
                                claess.append(0)

                        x = (hi + sigmoid(box[bi,0+ni*(classes+1+4),wi,hi])) / float(box.shape[3])
                        y = (wi + sigmoid(box[bi,1+ni*(classes+1+4),wi,hi])) / float(box.shape[2])

                        w2=  math.exp(box[bi,2+ni*(classes+1+4),wi,hi]) * biases[2*int(mask[ni])]/ float(w1)
                        h2 = math.exp(box[bi,3+ni*(classes+1+4),wi,hi]) * biases[2*int(mask[ni])+1] / float(h1)
                        boxone.append(x)                 #objectness prob
                        boxone.append(y)                 #objectness prob
                        boxone.append(w2)                 #objectness prob
                        boxone.append(h2)                 #objectness prob
                        #boxone.append(max(claess))                 #objectness prob
                        boxone.append(claess.index(max(claess)))   #objectness prob
                        boxone.append(objectness)
                        boxone.append(max(claess))                 #objectness prob
                        boxone.append(max(claess)*objectness)      #objectness prob


                        det.append(boxone)
                        boxes.append(boxone)
    return 1,boxes
*/

