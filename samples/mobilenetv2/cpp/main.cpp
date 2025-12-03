#include "mobilenetv2.hpp"

typedef struct {
    std::string input_dir;
    std::string model;
    std::string golden_label;
} CmdParams;

CmdParams parser(int argc, char** argv){
    const char * keys = 
        "{help h usage ? |      | print help message}"
        "{input_dir i      | datasets/img  | test image dir}"
        "{model m      | mobilenetv2_int8.nb  | model file}"
        "{accuracy a      | datasets/label.txt  | accuracy}"
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
    config.golden_label = parser.get<std::string>("accuracy");
    return config;
}

bool readLabelFile(const std::string& file_path, std::map<std::string, int>& label_map) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;

        // 分割路径和标签（按空格或制表符）
        std::istringstream iss(line);
        std::string path;
        int label;
        
        if (!(iss >> path >> label)) {
            std::cerr << "文件格式错误，行内容: " << line << std::endl;
            return false;
        }

        label_map[path] = label;
    }

    file.close();
    return true;
}

void compareLabelFiles(const std::string& file1, const std::string& file2) {
    std::map<std::string, int> map1, map2;

    // 读取两个文件
    if (!readLabelFile(file1, map1)) return;
    if (!readLabelFile(file2, map2)) return;

    // 统计差异
    int correct_top1 = 0;
    int total_mismatched = 0;
    int total = 0;
    // 检查file1中的所有条目
    for (const auto& pair : map1) {
        const std::string& path = pair.first;
        int label1 = pair.second;

        if (map2.find(path) != map2.end()) {
            // 条目在两个文件中都存在
            int label2 = map2[path];
            if (label1 == label2) {
                correct_top1++;
            } 
        }
        total++;
    }
    std::cout << "Top-1 Accuracy: " << ((float)correct_top1 / total) * 100 << "%" << std::endl;

}

int main(int argc, char* argv[]){
    //1.Parse params and print
    CmdParams config = parser(argc, argv);
    auto model_file = config.model;
    auto image_dir = config.input_dir;
    auto golden_label = config.golden_label;
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

    MOBILENETV2 mobilenetv2(nnrt_ctx,model_file);
    //2.Load and init model
  
    if (mobilenetv2.Init() != true) {
        fprintf(stderr, "Model initialization failed\n");
        return -1;
    }


    TimeStamp tick;
    TimeStamp *ts = &tick;
    
    mobilenetv2.enableProfile(ts);
    int total = 0;

    std::vector<cv::Mat> images;
    std::vector<std::string> image_names;
    std::vector<std::pair<int, float>> results;
    if(info.st_mode & S_IFREG){ // process single image
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
        
        mobilenetv2.Classify(images, results);
        total++;
        std::cout << image_dir << " pred: " << results[0].first << ", score:" << results[0].second << std::endl;


    }else if(info.st_mode & S_IFDIR){
        // get files and process imges in the dir
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
            if (image.empty()) {
                fprintf(stderr, "Failed to load image: %s\n", img_file.c_str());
                continue;
            }
            size_t index = img_file.rfind("/");
            std::string img_name = img_file.substr(index + 1);
            images.push_back(image);
            image_names.push_back(img_name);
    
            total++;
            
        }

        mobilenetv2.Classify(images, results);
        // std::cout<< " pred: " << results[0].first << ", score:" << result[0].second << std::endl;
        
        {
            std::ofstream out_file("result.txt");
            for(int i = 0; i < results.size(); i++){
                out_file << image_dir << image_names[i] << " " << results[i].first << "\n";
            }
            out_file.close();
        }
        
        if(golden_label.c_str()!= nullptr){
            compareLabelFiles("result.txt", golden_label.c_str());
        }

    }
    
    //4.Deinit model
    
    mobilenetv2.deInit();
    
    image_names.clear();
    images.clear();
    results.clear();
   
    //5.Print test results
    std::cout << "On average:" << std::endl;
    for (const auto& pair : tick.time_map_lab) {
        std::cout << pair.first << ": " << pair.second/total << " ms\n";
    }

    return 0;

}