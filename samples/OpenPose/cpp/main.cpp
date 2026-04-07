#include "openpose.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>

typedef struct
{
    std::string input_dir;
    std::string model;
    std::string golden_label;
    // float conf_thresh;
    float nms_thresh;
    bool save_draw;
    bool draw_obj;
} CmdParams;

static bool isVideoFile(const std::string &path)
{
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" || ext == "webm");
}

CmdParams parser(int argc, char **argv)
{
    const char *keys =
        "{help h usage ? |      | print help message}"
        "{input_dir i      | datasets/img  | test image dir}"
        "{model m      | openpose_coco_int8.nb  | model file}"
        "{nms_thresh n      | 0.5  | nms threshold}"
        "{save_draw s| 0 | save draw image}"
        "{draw_obj d| 0 | draw object}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        exit(0);
        return CmdParams{};
    }
    CmdParams config;
    config.input_dir = parser.get<std::string>("input_dir");
    config.model = parser.get<std::string>("model");
    // config.conf_thresh = parser.get<float>("conf_thresh");
    config.nms_thresh = parser.get<float>("nms_thresh");
    config.save_draw = parser.get<bool>("save_draw");
    config.draw_obj = parser.get<bool>("draw_obj");

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Input: " << config.input_dir << std::endl;
    // std::cout << "Conf thresh: " << config.conf_thresh << std::endl;
    std::cout << "NMS thresh: " << config.nms_thresh << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    return config;
}

int main(int argc, char *argv[])
{
    // 1.Parse params and print
    CmdParams config = parser(argc, argv);
    auto model_file = config.model;
    auto image_dir = config.input_dir;
    auto nms_thresh = config.nms_thresh;
    struct stat info;
    if (stat(model_file.c_str(), &info) != 0)
    {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }

    if (stat(image_dir.c_str(), &info) != 0)
    {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }
   
    std::shared_ptr<ta_runtime_context> nnrt_ctx = std::make_shared<ta_runtime_context>(0);

    OpenPose openpose(nnrt_ctx, model_file, nms_thresh, 0);
    // 2.Load and init model

    if (openpose.Init() != 0)
    {
        fprintf(stderr, "Model initialization failed\n");
        openpose.deInit();
        return -1;
    }
    PoseKeyPoints::EModelType model_type = openpose.get_model_type();
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    TimeStamp tick;
    TimeStamp *ts = &tick;

    openpose.enableProfile(ts);
    int total = 0;

    std::vector<cv::Mat> images;
    std::vector<std::string> image_names;
    std::vector<PoseKeyPoints> vct_keypoints;
    std::vector<nlohmann::json> results_json;
    std::cout << "Model Init successful" << std::endl;

    // 判断是视频文件还是图片目录
    if (info.st_mode & S_IFREG && isVideoFile(image_dir))
    {
        // 处理视频文件
        std::cout << "Processing video file: " << image_dir << std::endl;

        cv::VideoCapture cap;
        cap.open(image_dir);
        if (!cap.isOpened())
        {
            std::cerr << "Cannot open video file: " << image_dir << std::endl;
            openpose.deInit();
            return -1;
        }

        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "Video info: " << frame_width << "x" << frame_height
                  << " @ " << fps << " fps, " << total_frames << " frames" << std::endl;

        // 创建输出视频
        std::string output_video_path = "results/output_pose.mp4";
        cv::VideoWriter writer(output_video_path,
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               fps,
                               cv::Size(frame_width, frame_height));

        if (!writer.isOpened())
        {
            std::cerr << "Cannot create output video file" << std::endl;
            cap.release();
            openpose.deInit();
            return -1;
        }

        cv::Mat frame;
        int frame_count = 0;

        while (cap.read(frame))
        {
            frame_count++;

            std::vector<cv::Mat> single_frame = {frame.clone()};
            std::vector<PoseKeyPoints> frame_keypoints;

            tick.start();
            openpose.Detect(single_frame, frame_keypoints);
            tick.time_accumulation("infer_time");

            // 绘制关键点
            if (!frame_keypoints.empty() && !frame_keypoints[0].keypoints.empty())
            {
                OpenPosePostProcess::renderPoseKeypointsCpu(
                    frame,
                    frame_keypoints[0].keypoints,
                    frame_keypoints[0].shape,
                    0.05, 1.0, model_type);
            }

            writer.write(frame);
            total++;

            if (frame_count % 30 == 0)
            {
                std::cout << "Processed " << frame_count << "/" << total_frames << " frames" << std::endl;
            }
        }

        cap.release();
        writer.release();
        std::cout << "Video processing complete. Output saved to: " << output_video_path << std::endl;
    }
    else if (info.st_mode & S_IFREG)
    {
        std::cout << "info.st_mode & S_IFREG." << std::endl;
    }
    else if (info.st_mode & S_IFDIR)
    {
        // get files
        std::vector<std::string> files_vector;
        DIR *pDir;
        struct dirent *ptr;
        pDir = opendir(image_dir.c_str());
         if (!pDir) {
            std::cerr << "Cannot open directory: " << image_dir << std::endl;
            return -1;
        }
        while ((ptr = readdir(pDir)) != 0)
        {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            {
                files_vector.push_back(image_dir + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        std::vector<std::string>::iterator iter;
        for (iter = files_vector.begin(); iter != files_vector.end(); iter++)
        {
            std::string img_file = *iter;
            tick.start();
            cv::Mat image = cv::imread(img_file, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
            tick.time_accumulation("imread_time");
            size_t index = img_file.rfind("/");
            std::string img_name = img_file.substr(index + 1);
            images.push_back(image);
            image_names.push_back(img_name);

            total++;
        }
        std::cout << "Network Running" << std::endl;
        openpose.Detect(images, vct_keypoints);
        std::cout << "Network OVER" << std::endl;

      for (size_t i = 0; i < vct_keypoints.size(); ++i)
        {   
                nlohmann::json person_json;
                person_json["image_name"] = image_names[i];
                person_json["keypoints"] = vct_keypoints[i].keypoints;
              //  printf("points:%d\r\n",vct_keypoints[i].keypoints.size());
                results_json.push_back(person_json);
            
            std::string output_image_path = "results/images/" + image_names[i];
            OpenPosePostProcess::renderPoseKeypointsCpu(images[i], vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0,model_type);
            cv::imwrite(output_image_path,images[i]);
        }
        std::cout << "renderPoseKeypointsCpu OVER" << std::endl;
        {
            std::string json_file_name = "results/output.json";
            std::ofstream ofs(json_file_name);
            if (!ofs.is_open())
            {
                std::cerr << "Failed to open output file: " << json_file_name << std::endl;
                return -1;
            }
            ofs << std::setw(4) << results_json << std::endl;
            ofs.close();

            std::cout << "Results saved to: " << json_file_name << std::endl;
        }
        std::cout << "results OVER" << std::endl;

    }

    std::cout << "On average:" << std::endl;
    for (const auto &pair : tick.time_map_lab)
    {
        std::cout << pair.first << ": " << pair.second / total << " ms\n";
    }
    // 4.Deinit model
    openpose.deInit();
    image_names.clear();
    images.clear();
    results_json.clear();
    return 0;
}
