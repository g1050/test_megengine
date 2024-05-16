#include <cstdlib>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include "lite/network.h"
#include "lite/tensor.h"
#include "lite/global.h"
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace lite;

cv::Mat process(cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(32, 32));
    cv::Mat temp = cv::Mat::ones(image.size(), image.type()) * 255 - image; // 反色处理
    // cv::imwrite("processed_image.jpg", temp);
    return temp;
}

int main(int argc,char *argv[]){
    if(argc != 3){
        std::cout << "Usage: ./lite_infer <model_path> <image_path> " << std::endl;
        return -1; 
    }
    int major,minor,patch;
    get_version(major,minor,patch);
    std::cout << "version " << major << "." << minor << "." << patch << std::endl;
    std::string model_path = argv[1];
    // cuda配置
    lite::Config config{LiteDeviceType::LITE_CUDA};
    NetworkIO network_io;
    IO device_input{"data",false};
    network_io.inputs.push_back(device_input);

    //加载模型
    std::shared_ptr<Network> network = std::make_shared<Network>(config, network_io);
    // std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(model_path);
    //读入图像，前处理
    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    auto processed_image =  process(image);
    std::cout << "element size " << processed_image.elemSize() << std::endl;
    std::cout << "total size " << processed_image.total() << std::endl;
    int dataType = processed_image.type();
    if(dataType == CV_8U) {
        std::cout << "datatype is CV_8U" << std::endl;
    } else if(dataType == CV_32F) {
        std::cout << "datatype is CV_32F" << std::endl;
    } else {
        std::cout << "datatyp is " << dataType << std::endl;
    }
    // to cpu tensor
    std::shared_ptr<Tensor> host_tensor = std::make_shared<Tensor>();
    Layout layout;
    layout.ndim = 4;
    layout.shapes[0] = 1;
    layout.shapes[1] = 1;
    layout.shapes[2] = 32;
    layout.shapes[3] = 32;
    layout.data_type = LiteDataType::LITE_UINT8;
    host_tensor->set_layout(layout);
    size_t length = host_tensor->get_tensor_total_size_in_byte();
    std::cout << "Host tensor length is " << length << std::endl;
    void* host = host_tensor->get_memory_ptr();
    memcpy(host,processed_image.data,length);
    // host to device tensor
    auto tensor_device = Tensor(LiteDeviceType::LITE_CUDA, layout);
    tensor_device.copy_from(*host_tensor);

    //设置input tensor
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor("data");
    srand(static_cast<unsigned>(time(NULL)));
    length = input_tensor->get_tensor_total_size_in_byte();
    std::cout << "host tensor's length " << length << std::endl;
    void* in_data_ptr = input_tensor->get_memory_ptr();
    memcpy(in_data_ptr,processed_image.data,length);
    
    // reset inputtensor
    // input_tensor->reset(tensor_device.get_memory_ptr(), layout);

    // 推理
    network->forward();
    network->wait();

    // 获取推理结果
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    int32_t* predict_ptr = static_cast<int32_t*>(output_tensor->get_memory_ptr());
    size_t output_length = output_tensor->get_tensor_total_size_in_byte() / sizeof(int32_t);
    for(size_t i = 0;i<output_length;i++){
        std::cout << " output[" << i << "] " << static_cast<int32_t>(predict_ptr[i]) << std::endl;
    }

    return 0;
}