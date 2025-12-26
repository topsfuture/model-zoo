#include "clip.hpp"
#include "nlohmann/json.hpp"

#include "tokenizer/tokenizer.hpp"


typedef struct {
    std::string image_path;
    std::string text;
    std::vector<std::string> words_vector;
    std::string image_model;
    std::string text_model;
    std::string text_projection_path;

    std::string golden_label;
    bool save_draw;
} CmdParams;



void get_text_features(CLIPTokenizer& tokenizer, std::string word, std::vector<int>& ids, size_t max_token_len) {
    ids = tokenizer.tokenize(word, nullptr, max_token_len, true);
}




std::vector<std::string> get_image_paths(const std::string& image_path) {
    /* get all image path in folder */
    std::vector<std::string> image_paths;
    if (std::filesystem::is_regular_file(image_path)) {
        image_paths.push_back(image_path);
    } else if (std::filesystem::is_directory(image_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(image_path)) {
            if (entry.is_regular_file()) {
                image_paths.push_back(entry.path().string());
            }
        }
    }
    return image_paths;
}

std::vector<std::string> split(const std::string& input) {
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, ',')) {
        item.erase(std::remove(item.begin(), item.end(), '\"'), item.end());
        result.push_back(item);
    }

    return result;
}


CmdParams parser(int argc, char** argv){
    const char * keys = 
        "{help h usage ? |      | print help message}"
        "{image_path | ./dataset | path to the image directory}"
        "{text | \"a diagram,a person ,a Car\" | text inputs for prediction (multiple texts can be separated by spaces and must be quoted)}"
        "{image_model | ./models/clip_image_float16.nb | path to the image model file}"
        "{text_model | ./models/clip_text_float16.nb | path to the text model file}"
        "{text_projection_path | ./text_projection_512_512.npy | path to the text projection file}"
        ;


    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        exit(0);
        return CmdParams{};
    }

    CmdParams config;
    config.image_path = parser.get<std::string>("image_path");
    config.text = parser.get<std::string>("text");
    config.words_vector = split(config.text);
    config.image_model = parser.get<std::string>("image_model");
    config.text_model = parser.get<std::string>("text_model");
    config.text_projection_path = parser.get<std::string>("text_projection_path");

    // check params
    struct stat info;
    if (stat(config.image_model.c_str(), &info) != 0) {
        std::cout << "Cannot find valid image model file: " << config.image_model << std::endl;
        exit(1);
    }
    if (stat(config.text_model.c_str(), &info) != 0) {
        std::cout << "Cannot find valid text model file: " << config.text_model << std::endl;
        exit(1);
    }
    if (stat(config.text_projection_path.c_str(), &info) != 0) {
        std::cout << "Cannot find valid text_projection_path: " << config.text_projection_path << std::endl;
        exit(1);
    }
    if (stat(config.image_path.c_str(), &info) != 0) {
        std::cout << "Cannot find input path: " << config.image_path << std::endl;
        exit(1);
    }


    
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Model: " << config.image_model << config.text_model << std::endl;
    std::cout << "Input: " << config.image_path << config.text << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    return config;
}

int main(int argc, char* argv[]){
    //1.Parse params and print
    CmdParams config = parser(argc, argv);

    auto image_path = config.image_path;
    auto text = config.text;
    auto words_vector = config.words_vector;
    auto image_model = config.image_model;
    auto text_model = config.text_model;
    auto text_projection_path = config.text_projection_path;

    // create context
    std::shared_ptr<ta_runtime_context> nnrt_ctx_text = std::make_shared<ta_runtime_context>(0);
    std::shared_ptr<ta_runtime_context> nnrt_ctx_image = std::make_shared<ta_runtime_context>(0);

    /* instantiate clip object */
    CLIP clip;
    if (clip.init(image_model, text_model, text_projection_path, nnrt_ctx_text, nnrt_ctx_image) != 0) {
        fprintf(stderr, "Model initialization failed\n");
        clip.deinit(nnrt_ctx_image, nnrt_ctx_text);
        return -1;
    }

    CLIPTokenizer tokenizer;
    std::vector<std::vector<int>> texts_features;
    size_t max_token_len = clip.get_token_len();

    TimeStamp tick;
    TimeStamp *ts = &tick;
    clip.enableProfile(ts);
    int total = 3;  // size of inference
    
    clip.ts_->start();
    for (const auto& label : words_vector){
        std::vector<int> text_vec_ids;
        get_text_features(tokenizer, label, text_vec_ids, max_token_len);
        texts_features.push_back(text_vec_ids);
    }
    clip.ts_->time_accumulation("tokenlize_time");

    // infer
    auto image_paths = get_image_paths(image_path);

    clip.infer(image_paths, texts_features, words_vector, 
               clip, nnrt_ctx_text, nnrt_ctx_image
               );
    std::cout << "On average:" << std::endl;
    for (const auto& pair : tick.time_map_lab) {
        std::cout << pair.first << ": " << pair.second/total << " ms\n";
    }
    // log times
    clip.deinit(nnrt_ctx_image, nnrt_ctx_text);

}