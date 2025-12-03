#include "yolov8.hpp"
#include "nlohmann/json.hpp"

typedef struct {
    std::string input_dir;
    std::string model;
    std::string golden_label;
    float conf_thresh;
    float nms_thresh;
    bool save_draw;
} CmdParams;

CmdParams parser(int argc, char** argv){
    const char * keys = 
        "{help h usage ? |      | print help message}"
        "{input_dir i      | datasets/img  | test image dir}"
        "{model m      | yolov8s_u8.nb  | model file}"
        "{conf_thresh c| 0.25 | confidence threshold for filter boxes}"
        "{nms_thresh n| 0.45 | iou threshold for nms}"
        "{save_draw s| 0 | save draw image}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        exit(0);
        return CmdParams{};
    }
    CmdParams config;
    config.input_dir = parser.get<std::string>("input_dir");
    config.model = parser.get<std::string>("model");
    config.conf_thresh = parser.get<float>("conf_thresh");
    config.nms_thresh = parser.get<float>("nms_thresh");
    config.save_draw = parser.get<bool>("save_draw");
    
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Input: " << config.input_dir << std::endl;
    std::cout << "Conf thresh: " << config.conf_thresh << std::endl;
    std::cout << "NMS thresh: " << config.nms_thresh << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    return config;
}

int main(int argc, char* argv[]){
    //1.Parse params and print
    CmdParams config = parser(argc, argv);
    auto model_file = config.model;
    auto image_dir = config.input_dir;
    auto nms_thresh = config.nms_thresh;
    auto conf_thresh = config.conf_thresh;
    printf("conf_thresh: %f, nms_thresh: %f\n",conf_thresh, nms_thresh);
    struct stat info;
    if (stat(model_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    
    if (stat(image_dir.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    std::shared_ptr<ta_runtime_context> nnrt_ctx = std::make_shared<ta_runtime_context>(0);

    YOLOV8 yolov8(nnrt_ctx,model_file,conf_thresh,nms_thresh);
    //2.Load and init model
    
    if (yolov8.Init() != 0) {
        fprintf(stderr, "Model initialization failed\n");
        yolov8.deInit();
        return -1;
    }

    if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
    mkdir("results/images", S_IRWXU);

    TimeStamp tick;
    TimeStamp *ts = &tick;
    
    yolov8.enableProfile(ts);
    int total = 0;

    std::vector<cv::Mat> images;
    std::vector<std::string> image_names;
    std::vector<YoloV8BoxVec> results;
    std::vector<nlohmann::json> results_json;
    std::cout << "Model Init successful" << std::endl;
    if(info.st_mode & S_IFREG){
        tick.start();
        cv::Mat image = cv::imread(image_dir,cv::IMREAD_COLOR |cv::IMREAD_RETRY_SOFTDEC);
        tick.time_accumulation("imread_time");
        if (image.empty()) {
            fprintf(stderr, "Failed to load image: %s\n", image_dir.c_str());
            return -1;
        }
        
        images.push_back(image);
        size_t index = image_dir.rfind("/");
        std::string img_name = image_dir.substr(index + 1);
        image_names.push_back(img_name);
        std::cout << "Network Running" << std::endl;
        yolov8.Detect(images,results);
        total++;
        yolov8.draw_objects(image, results[0], "result.jpg");
        

    }else if(info.st_mode & S_IFDIR){
        // get files
        std::vector<std::string> files_vector;
        DIR *pDir;
        struct dirent* ptr;
        pDir = opendir(image_dir.c_str());
        while((ptr = readdir(pDir))!=0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(image_dir + "/" + ptr->d_name);
            }
        }
        closedir(pDir);
        
        std::vector<std::string>::iterator iter;
        for (iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            std::string img_file = *iter;
            tick.start();
            cv::Mat image = cv::imread(img_file,cv::IMREAD_COLOR |cv::IMREAD_RETRY_SOFTDEC);
            tick.time_accumulation("imread_time");
            size_t index = img_file.rfind("/");
            std::string img_name = img_file.substr(index + 1);
            images.push_back(image);
            image_names.push_back(img_name);
    
            total++;
            
        }
        std::cout << "Network Running" << std::endl;
        yolov8.Detect(images,results);
        
        for(int i = 0; i < images.size(); i++){
            nlohmann::json image_result;
            nlohmann::json bboxes_json = nlohmann::json::array();
            for(auto result : results[i]){
                nlohmann::json bbox_json;
                bbox_json["category_id"] = result.class_id;
                bbox_json["score"] = result.score;
                bbox_json["bbox"] = {result.x, result.y, result.width, result.height};
                bboxes_json.push_back(bbox_json);
            }
            std::string output_image_path = "results/images/" + image_names[i];
            if(config.save_draw)
                yolov8.draw_objects(images[i], results[i], output_image_path);
            image_result["image_name"] = image_names[i];
            image_result["bboxes"] = bboxes_json;
            results_json.push_back(image_result);
        }        
        
        {
            std::string json_file_name = "results/output.json";
            std::ofstream ofs(json_file_name);
            if (!ofs.is_open()) {
                std::cerr << "Failed to open output file: " << json_file_name << std::endl;
                return -1;
            }
            ofs << std::setw(4) << results_json << std::endl;
            ofs.close();

            std::cout << "Results saved to: " << json_file_name << std::endl;
        }

    }
    
    
    std::cout << "On average:" << std::endl;
    for (const auto& pair : tick.time_map_lab) {
        std::cout << pair.first << ": " << pair.second/total << " ms\n";
    }
    //4.Deinit model
    yolov8.deInit();
    
    image_names.clear();
    images.clear();
    results.clear();
    results_json.clear();

    return 0;

}