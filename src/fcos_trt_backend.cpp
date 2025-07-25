#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

struct Detection {
    cv::Rect bbox;
    float score;
    int class_id;
};

struct TRTEngine {
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    std::vector<void*> buffers;
    cudaStream_t stream;
    std::vector<int> input_dims;
    std::vector<int> output_dims[6];
};

bool loadEngine(const std::string& engineFile, TRTEngine& trt) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Unable to read engine file: " << engineFile << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    trt.runtime = createInferRuntime(gLogger);
    if (!trt.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    trt.engine = trt.runtime->deserializeCudaEngine(engineData.data(), size);
    if (!trt.engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    trt.context = trt.engine->createExecutionContext();
    if (!trt.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    CHECK(cudaStreamCreate(&trt.stream));

    // Get input and output dimensions using TensorRT 10.x API
    int num_bindings = trt.engine->getNbIOTensors();
    trt.buffers.resize(num_bindings);

    for (int i = 0; i < num_bindings; ++i) {
        const char* name = trt.engine->getIOTensorName(i);
        Dims dims = trt.engine->getTensorShape(name);
        std::vector<int> dims_vec(dims.d, dims.d + dims.nbDims);
        
        if (trt.engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            trt.input_dims = dims_vec;
        } else {
            trt.output_dims[i - 1] = dims_vec;
        }

        size_t size = std::accumulate(dims_vec.begin(), dims_vec.end(), 1, std::multiplies<int>());
        CHECK(cudaMalloc(&trt.buffers[i], size * sizeof(float)));
    }

    return true;
}

void preprocessImage(const cv::Mat& img, float* gpu_input, const std::vector<int>& input_dims) {
    int height = input_dims[1];
    int width = input_dims[2];
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }
    
    std::vector<float> chw_data;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                chw_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    CHECK(cudaMemcpyAsync(gpu_input, chw_data.data(), 
                         chw_data.size() * sizeof(float), 
                         cudaMemcpyHostToDevice, nullptr));
}

std::vector<Detection> decodeOutputs(
    float* cls_logits, float* bbox_regression, float* bbox_ctrness, float* anchors,
    const std::vector<int>& cls_dims, const std::vector<int>& reg_dims,
    float score_thresh = 0.5, float nms_thresh = 0.5) {
    
    std::vector<Detection> detections;
    int num_anchors = reg_dims[1];
    int num_classes = cls_dims[2];
    
    for (int i = 0; i < num_anchors; ++i) {
        float max_score = -1.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = cls_logits[i * num_classes + c];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        float centerness = bbox_ctrness[i];
        max_score *= centerness;
        
        if (max_score < score_thresh || class_id < 0) continue;
        
        float* reg = &bbox_regression[i * 4];
        float* anchor = &anchors[i * 4];
        
        float x1 = anchor[0] - reg[0];
        float y1 = anchor[1] - reg[1];
        float x2 = anchor[2] + reg[2];
        float y2 = anchor[3] + reg[3];
        
        detections.push_back(Detection{
            cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
            max_score,
            class_id
        });
    }
    
    std::vector<Detection> final_detections;
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.score);
    }
    
    cv::dnn::NMSBoxes(boxes, scores, score_thresh, nms_thresh, indices);
    
    for (int idx : indices) {
        final_detections.push_back(detections[idx]);
    }
    
    return final_detections;
}

std::vector<Detection> runInference(TRTEngine& trt, const cv::Mat& img) {
    // Preprocess
    preprocessImage(img, (float*)trt.buffers[0], trt.input_dims);
    
    // Set input/output tensor addresses
    for (int i = 0; i < trt.engine->getNbIOTensors(); ++i) {
        const char* name = trt.engine->getIOTensorName(i);
        trt.context->setTensorAddress(name, trt.buffers[i]);
    }
    
    // Run inference
    if (!trt.context->enqueueV3(trt.stream)) {
        std::cerr << "Failed to enqueue inference" << std::endl;
        return {};
    }
    cudaStreamSynchronize(trt.stream);
    
    // Copy outputs
    std::vector<float> cls_logits(std::accumulate(trt.output_dims[0].begin(), trt.output_dims[0].end(), 1, std::multiplies<int>()));
    std::vector<float> bbox_regression(std::accumulate(trt.output_dims[1].begin(), trt.output_dims[1].end(), 1, std::multiplies<int>()));
    std::vector<float> bbox_ctrness(std::accumulate(trt.output_dims[2].begin(), trt.output_dims[2].end(), 1, std::multiplies<int>()));
    std::vector<float> anchors(std::accumulate(trt.output_dims[3].begin(), trt.output_dims[3].end(), 1, std::multiplies<int>()));
    
    CHECK(cudaMemcpyAsync(cls_logits.data(), trt.buffers[1], 
                         cls_logits.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost, trt.stream));
    CHECK(cudaMemcpyAsync(bbox_regression.data(), trt.buffers[2], 
                         bbox_regression.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost, trt.stream));
    CHECK(cudaMemcpyAsync(bbox_ctrness.data(), trt.buffers[3], 
                         bbox_ctrness.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost, trt.stream));
    CHECK(cudaMemcpyAsync(anchors.data(), trt.buffers[4], 
                         anchors.size() * sizeof(float), 
                         cudaMemcpyDeviceToHost, trt.stream));
    
    cudaStreamSynchronize(trt.stream);
    
    return decodeOutputs(cls_logits.data(), bbox_regression.data(), 
                        bbox_ctrness.data(), anchors.data(),
                        trt.output_dims[0], trt.output_dims[1]);
}

void cleanup(TRTEngine& trt) {
    for (auto& buffer : trt.buffers) {
        CHECK(cudaFree(buffer));
    }
    cudaStreamDestroy(trt.stream);
    if (trt.context) {
        delete trt.context;
    }
    if (trt.engine) {
        delete trt.engine;
    }
    if (trt.runtime) {
        delete trt.runtime;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_file> [score_thresh] [nms_thresh]" << std::endl;
        return -1;
    }
    
    std::string engineFile = argv[1];
    std::string imageFile = argv[2];
    float score_thresh = argc > 3 ? std::atof(argv[3]) : 0.5f;
    float nms_thresh = argc > 4 ? std::atof(argv[4]) : 0.5f;
    
    TRTEngine trt;
    if (!loadEngine(engineFile, trt)) {
        return -1;
    }
    
    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return -1;
    }
    
    auto detections = runInference(trt, img);
    
    for (const auto& det : detections) {
        if (det.score < score_thresh) continue;
        
        cv::rectangle(img, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.class_id) + ": " + std::to_string(det.score);
        cv::putText(img, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    std::string outputFile = "result.jpg";
    cv::imwrite(outputFile, img);
    std::cout << "Saved result to " << outputFile << std::endl;
    
    cleanup(trt);
    
    return 0;
}
