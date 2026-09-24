#pragma once
#include <chrono>
#include <map>
#include <string>

class TimeStamp {
private:
    std::chrono::high_resolution_clock::time_point start_time, end_time;
public:
    TimeStamp() { start(); time_map_lab["imread_time"] = 0; time_map_lab["pre_time"] = 0;
                  time_map_lab["infer_time"] = 0; time_map_lab["post_time"] = 0;
                  time_map_lab["mask_time"] = 0; }
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_time   = std::chrono::high_resolution_clock::now(); }
    std::map<std::string, float> time_map_lab;
    float cost() {
        stop();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        return static_cast<float>(ms) / 1000.0f;
    }
    void time_accumulation(const std::string& label) {
        stop();
        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        time_map_lab[label] += gap / 1000.0f;
    }
};
