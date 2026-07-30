#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "ta-runtime-api.h"
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <stdint.h>
#include <time.h>
#include <dirent.h> 
#include <math.h>   
#include <sys/time.h> 

#define MAX_CORES        2
#define MAX_MODELS       2
#define STATS_CORE       2
#define ALIGN_SIZE       256
#define CLAMP(val, lo, hi)  (((val) < (lo)) ? (lo) : (((val) > (hi)) ? (hi) : (val)))
enum core_mode_e {
    CORE_MODE_CORE0 = 0,
    CORE_MODE_CORE1 = 1,
    CORE_MODE_DUAL  = 2
};

#define TEST_DDR_MODEL 1

static bool is_run = true;
static uint32_t g_frame_count[2] = {0, 0};
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static double g_total_inference_time[MAX_CORES] = {0.0, 0.0};


typedef struct test_task_s {
    char                  *model_path;
    ta_runtime_context    context;
    uint32_t              loop_count;
    uint32_t              input_count;
    uint32_t              output_count;
    taconn_input_t        *input;
    taconn_buffer_t       *output;
    uint32_t              core_mask;
} test_task_t;

static uint32_t get_data_format_size(taconn_data_format_t format)
{
    switch (format) {
    case TACONN_DATA_FORMAT_FP32:
    case TACONN_DATA_FORMAT_INT32:
    case TACONN_DATA_FORMAT_UINT32:
        return 4;
    case TACONN_DATA_FORMAT_FP16:
    case TACONN_DATA_FORMAT_UINT16:
    case TACONN_DATA_FORMAT_INT16:
    case TACONN_DATA_FORMAT_BFP16:
        return 2;
    case TACONN_DATA_FORMAT_UINT8:
    case TACONN_DATA_FORMAT_INT8:
    case TACONN_DATA_FORMAT_CHAR:
    case TACONN_DATA_FORMAT_BOOL8:
        return 1;
    case TACONN_DATA_FORMAT_INT64:
    case TACONN_DATA_FORMAT_UINT64:
    case TACONN_DATA_FORMAT_FP64:
        return 8;
    case TACONN_DATA_FORMAT_INT4:
    case TACONN_DATA_FORMAT_UINT4:
        return 1;
    default:
        return 1;
    }
}

static uint32_t calc_tensor_size(taconn_inout_attr_t *attr)
{
    uint32_t size = get_data_format_size(attr->data_format);
    for (uint32_t i = 0; i < attr->dim_count; i++) {
        size *= attr->dim_size[i];
    }
    return size;
}

static void generate_random_data(void *buffer, uint32_t size)
{
    uint8_t *p = (uint8_t *)buffer;
    for (uint32_t i = 0; i < size; i++) {
        p[i] = rand() & 0xFF;
    }
}

void* load_file(const char* filename, int* file_size, bool is_align)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int file_len = ftell(fp);
    void* file = NULL;
    if (is_align) {
        posix_memalign(&file, 256, file_len);
    } else {
        file = malloc(file_len);
    }
    fseek(fp, 0, SEEK_SET);
    if (file_len != fread(file, 1, file_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(file);
        fclose(fp);
        return NULL;
    }
    *file_size = file_len;
    fclose(fp);
    return file;
}

int create_network(test_task_t *task, uint32_t core_mask)
{
    int status = 0;
    printf("create network file_path = %s\n", task->model_path);
#if TEST_DDR_MODEL
    int model_size = 0;
    void* model = load_file(task->model_path, &model_size, 0);
    if (model == NULL) {
        printf("load model file failed\n");
        return -1;
    }
    status = ta_runtime_load_model_at_ddr(&task->context, model, model_size, core_mask);
    free(model);
#else
    status = ta_runtime_load_model_from_file(&task->context, task->model_path, core_mask);
#endif
    if (status != 0) {
        printf("create network failed, status=%d\n", status);
        return -1;
    }
    return 0;
}

int query_io_info(test_task_t *task)
{
    int status = 0;
    taconn_input_output_num_t io_num = {0};
    status = ta_runtime_query(&task->context, TACONN_QUERY_IN_OUT_NUM, &io_num);
    if (status != 0) {
        printf("query input/output num failed\n");
        return -1;
    }
    task->input_count = io_num.input_num;
    task->output_count = io_num.output_num;
    printf("Model has %d inputs and %d outputs\n", task->input_count, task->output_count);
    return 0;
}

int set_input_random(test_task_t *task)
{
    int status = 0;
    taconn_inout_attr_t attr = {0};
    task->input = (taconn_input_t *)malloc(sizeof(taconn_input_t) * task->input_count);
    if (task->input == NULL) {
        printf("malloc input array failed\n");
        return -1;
    }
    for (uint32_t i = 0; i < task->input_count; i++) {
        attr.index = i;
        status = ta_runtime_query(&task->context, TACONN_QUERY_INPUT_ATTR, &attr);
        if (status != 0) {
            printf("query input attr %d failed\n", i);
            return -1;
        }
        uint32_t input_size = calc_tensor_size(&attr);
        printf("Input[%d] name=%s, dims=%d, size=%u bytes, format=%s\n",
               i, attr.name, attr.dim_count, input_size,
               get_type_string(attr.data_format)); 
        task->input[i].index = i;
        task->input[i].size = input_size;
        posix_memalign(&task->input[i].data, 256, input_size);
        if (task->input[i].data == NULL) {
            printf("malloc input data %d failed\n", i);
            return -1;
        }
        generate_random_data(task->input[i].data, input_size);
    }
    status = ta_runtime_set_input_cva(&task->context, task->input_count, task->input);
    if (status != 0) {
        printf("set input failed\n");
        return -1;
    }
    return 0;
}

int set_output(test_task_t *task)
{
    int status = 0;
    taconn_inout_attr_t attr = {0};
    task->output = (taconn_buffer_t *)malloc(sizeof(taconn_buffer_t) * task->output_count);
    if (task->output == NULL) {
        printf("malloc output array failed\n");
        return -1;
    }
    for (uint32_t i = 0; i < task->output_count; i++) {
        attr.index = i;
        status = ta_runtime_query(&task->context, TACONN_QUERY_OUTPUT_ATTR, &attr);
        if (status != 0) {
            printf("query output attr %d failed\n", i);
            return -1;
        }
        uint32_t output_size = calc_tensor_size(&attr);
        printf("Output[%d] name=%s, dims=%d, size=%u bytes, format=%s\n",
               i, attr.name, attr.dim_count, output_size,
               get_type_string(attr.data_format));
        status = ta_runtime_create_buffer(&task->context, output_size, &task->output[i]);
        if (status != 0) {
            printf("create buffer failed\n");
            return -1;
        }
    }
    status = ta_runtime_set_output(&task->context, task->output_count, task->output);
    if (status != 0) {
        printf("set output failed\n");
        return -1;
    }
    return 0;
}


static void* read_tensor(const char *path, taconn_data_format_t format, size_t *out_count)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        printf("Cannot open file: %s\n", path);
        return NULL;
    }

    size_t elem_size = 0;
    const char *fmt = NULL;
    switch (format) {  // extensible
        case TACONN_DATA_FORMAT_FP32:
            elem_size = sizeof(float);
            fmt = "%f";
            break;
        case TACONN_DATA_FORMAT_INT32:
            elem_size = sizeof(int32_t);
            fmt = "%d";
            break;
        default:
            printf("Unsupported format for text reading: %d\n", format);
            fclose(fp);
            return NULL;
    }

    size_t cap = 1024, cnt = 0;
    void *data = malloc(cap * elem_size);
    if (!data) {
        fclose(fp);
        printf("malloc failed for %s\n", path);
        return NULL;
    }

    while (fscanf(fp, fmt, (char*)data + cnt * elem_size) == 1) {
        cnt++;
        if (cnt >= cap) {
            cap *= 2;
            void *new_data = realloc(data, cap * elem_size);
            if (!new_data) {
                free(data);
                fclose(fp);
                printf("realloc failed for %s\n", path);
                return NULL;
            }
            data = new_data;
        }
    }

    fclose(fp);
    if (cnt == 0) {
        free(data);
        printf("Empty tensor file: %s\n", path);
        return NULL;
    }
    *out_count = cnt;
    return data;
}



/* ---------- 在目录中查找唯一的 .nb 文件 ---------- */
static char* find_nb_file_in_dir(const char *dir_path)
{
    DIR *dir = opendir(dir_path);
    if (!dir) {
        printf("Cannot open directory: %s\n", dir_path);
        return NULL;
    }
    char *found = NULL;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        const char *ext = strrchr(entry->d_name, '.');
        if (ext && strcasecmp(ext, ".nb") == 0) {
            if (found) {
                printf("Multiple .nb files found in %s, abort.\n", dir_path);
                free(found);
                closedir(dir);
                return NULL;
            }
            size_t len = strlen(dir_path) + strlen(entry->d_name) + 2;
            found = malloc(len);
            snprintf(found, len, "%s/%s", dir_path, entry->d_name);
        }
    }
    closedir(dir);
    if (!found) {
        printf("No .nb file found in %s\n", dir_path);
    }
    return found;
}


static uint16_t f32_to_f16(float value) {
    uint32_t x = *(uint32_t*)(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
}

static int set_input_from_file(test_task_t *task, const char *ref_dir)
{
    task->input = (taconn_input_t *)malloc(sizeof(taconn_input_t) * task->input_count);
    if (!task->input) {
        printf("malloc input array failed\n");
        return -1;
    }

    for (uint32_t i = 0; i < task->input_count; i++) {
        taconn_inout_attr_t attr = {0};
        attr.index = i;
        int status = ta_runtime_query(&task->context, TACONN_QUERY_INPUT_ATTR, &attr);
        if (status != 0) {
            printf("query input attr %d failed\n", i);
            return -1;
        }

        uint32_t elem_count = calc_tensor_size(&attr) / get_data_format_size(attr.data_format);
        uint32_t tensor_size = calc_tensor_size(&attr);

        // 构造 input_%d.tensor 路径
        size_t path_len = strlen(ref_dir) + 30;
        char *txt_path = malloc(path_len);
        snprintf(txt_path, path_len, "%s/input_%d.tensor", ref_dir, i);

        size_t read_elem = 0;
        void *src_data = NULL;  
        taconn_data_format_t read_format;
        if (attr.quant_format == TACONN_QNT_TYPE_NONE && attr.data_format == TACONN_DATA_FORMAT_INT32) {
            read_format = TACONN_DATA_FORMAT_INT32; 
        } else {
            read_format = TACONN_DATA_FORMAT_FP32;  
        }
        src_data = read_tensor(txt_path, read_format, &read_elem);
        free(txt_path);
        if (!src_data) {
            return -1;
        }

        if (read_elem != elem_count) {
            printf("[FAIL] input_%d.tensor element count mismatch: expected %u, got %zu\n",
                   i, elem_count, read_elem);
            free(src_data);
            return -1;
        }

        posix_memalign(&task->input[i].data, ALIGN_SIZE, tensor_size);
        if (!task->input[i].data) {
            printf("posix_memalign failed for input %d\n", i);
            free(src_data);
            return -1;
        }

        if (attr.quant_format == TACONN_QNT_TYPE_NONE) {
            if (attr.data_format == TACONN_DATA_FORMAT_FP32) {
                memcpy(task->input[i].data, src_data, tensor_size);
            } else if (attr.data_format == TACONN_DATA_FORMAT_FP16) {
                float *src_f32 = (float *)src_data; 
                uint16_t *dst = (uint16_t *)task->input[i].data;
                for (uint32_t e = 0; e < elem_count; e++) {
                    dst[e] = f32_to_f16(src_f32[e]);
                }
            } else if (attr.data_format == TACONN_DATA_FORMAT_UINT8) {
                /*带 prenode */
                float *src_f32 = (float *)src_data;
                uint8_t *dst = (uint8_t *)task->input[i].data;
                for (uint32_t e = 0; e < elem_count; e++) {
                    int32_t val = (int32_t)lrintf(src_f32[e]);
                    dst[e] = (uint8_t)CLAMP(val, 0, 255);
                }
            } else if (attr.data_format == TACONN_DATA_FORMAT_INT32) {
                memcpy(task->input[i].data, src_data, tensor_size);
            } else {
                printf("Unsupported non-quantized format: %d\n", attr.data_format);
                free(src_data);
                return -1;
            }
        } else if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
            float *src_f32 = (float *)src_data;
            float scale = attr.quant_data.affine.tf_scale;
            int32_t zero_point = attr.quant_data.affine.tf_zero_point;
            if (attr.data_format == TACONN_DATA_FORMAT_INT8) {
                int8_t *dst = (int8_t *)task->input[i].data;
                for (uint32_t e = 0; e < elem_count; e++) {
                    int32_t q = (int32_t)roundf(src_f32[e] / scale) + zero_point;
                    dst[e] = (int8_t)CLAMP(q, -128, 127);
                }
            } else if (attr.data_format == TACONN_DATA_FORMAT_UINT8) {
                uint8_t *dst = (uint8_t *)task->input[i].data;
                for (uint32_t e = 0; e < elem_count; e++) {
                    int32_t q = (int32_t)roundf(src_f32[e] / scale) + zero_point;
                    dst[e] = (uint8_t)CLAMP(q, 0, 255);
                }
            } else {
                printf("Unsupported asymmetric quantized format: %d\n", attr.data_format);
                free(src_data);
                return -1;
            }
        } else if (attr.quant_format == TACONN_QNT_TYPE_DFP) {
            float *src_f32 = (float *)src_data;
            int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
            if (attr.data_format == TACONN_DATA_FORMAT_INT8) {
                int8_t *dst = (int8_t *)task->input[i].data;
                for (uint32_t e = 0; e < elem_count; e++) {
                    int32_t scaled = lrintf(src_f32[e] * (1 << fixed_point_pos));
                    dst[e] = (int8_t)CLAMP(scaled, -128, 127);
                }
            } else {
                printf("Unsupported DFP format: %d\n", attr.data_format);
                free(src_data);
                return -1;
            }
        } else {
            printf("Unknown quant format: %d\n", attr.quant_format);
            free(src_data);
            return -1;
        }

        task->input[i].index = i;
        task->input[i].size = tensor_size;
        printf("Input[%d] name=%s, dims=%d, size=%u bytes, format=%s\n",
               i, attr.name, attr.dim_count, tensor_size,
               get_type_string(attr.data_format)); 

        free(src_data);  
    }

    int status = ta_runtime_set_input_cva(&task->context, task->input_count, task->input);
    if (status != 0) {
        printf("set input failed\n");
        return -1;
    }
    return 0;
}


float dequantize_float32(void* data, size_t idx, int32_t, float) {
    return ((float*)data)[idx];                 
}

float dequantize_float16(void* data, size_t idx, int32_t, float) {
    uint16_t h = ((uint16_t*)data)[idx];            
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t f = (sign << 31);
            return *(float*)&f;                    
        }
        const uint32_t exp_offset = 103;
        uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
        return *(float*)&f;
    } 
    else if (exponent == 31) {
        uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *(float*)&f;
    }
    
    exponent += (127 - 15);
    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *(float*)&f;
}

float dequantize_uint8(void* data, size_t idx, int32_t zp, float scale) {
    uint8_t val = ((uint8_t*)data)[idx];      
    return ((float)val - (float)zp) * scale;
}

float dequantize_int8(void* data, size_t idx, int32_t zp, float scale) {
    int8_t val = ((int8_t*)data)[idx];        
    return ((float)val - (float)zp) * scale;
}
static int compare_output_with_reference(test_task_t *task, const char *ref_dir)
{
    bool all_pass = true;

    for (uint32_t i = 0; i < task->output_count; i++) {
        taconn_inout_attr_t attr = {0};
        attr.index = i;
        ta_runtime_query(&task->context, TACONN_QUERY_OUTPUT_ATTR, &attr);

        uint32_t tensor_size = calc_tensor_size(&attr);
        uint32_t elem_count = tensor_size / get_data_format_size(attr.data_format);

        printf("Output[%d]: name=%s, dims=[", i, attr.name);
        for (uint32_t d = 0; d < attr.dim_count; d++) {
            printf("%u%s", attr.dim_size[d], (d+1 < attr.dim_count) ? "," : "");
        }
        printf("], elements=%u\n", elem_count);

        // 构造 output_%d.tensor 路径
        size_t path_len = strlen(ref_dir) + 28;
        char *txt_path = malloc(path_len);
        snprintf(txt_path, path_len, "%s/output_%d.tensor", ref_dir, i);

        taconn_data_format_t read_format;
        if (attr.data_format == TACONN_DATA_FORMAT_INT32) {
            read_format = TACONN_DATA_FORMAT_INT32;
        } else {
            read_format = TACONN_DATA_FORMAT_FP32;  
        }

        size_t ref_elem = 0;
        void *ref_data = read_tensor(txt_path, read_format, &ref_elem);
        free(txt_path);
        if (!ref_data) {
            all_pass = false;
            continue;
        }

        if (ref_elem != elem_count) {
            printf("[FAIL] output_%d.tensor element count mismatch: expected %u, got %zu\n",
                   i, elem_count, ref_elem);
            free(ref_data);
            all_pass = false;
            continue;
        }

        // actual 数据
        void *actual = task->output[i].data;
        if (!actual) {
            printf("[FAIL] Output buffer %d has NULL data pointer\n", i);
            free(ref_data);
            all_pass = false;
            continue;
        }

        double dot = 0.0, norm_ref = 0.0, norm_act = 0.0;

        if (attr.data_format == TACONN_DATA_FORMAT_INT32) {
            // 实际输出和参考数据均为 int32_t
            int32_t *act_i32 = (int32_t *)actual;
            int32_t *ref_i32 = (int32_t *)ref_data;
            for (uint32_t e = 0; e < elem_count; e++) {
                double r = (double)ref_i32[e];
                double a = (double)act_i32[e];
                dot += r * a;
                norm_ref += r * r;
                norm_act += a * a;
            }
        } else {
            // 其他格式：参考数据为 float，实际输出需反量化
            float *ref_f32 = (float *)ref_data;
            float (*dequant)(void*, size_t, int32_t, float) = NULL;
            float scale = 0.0f;
            int32_t zero_point = 0;

            switch (attr.data_format) {
                case TACONN_DATA_FORMAT_FP32:
                    dequant = dequantize_float32;
                    break;
                case TACONN_DATA_FORMAT_FP16:
                    dequant = dequantize_float16;
                    break;
                case TACONN_DATA_FORMAT_UINT8:
                    dequant = dequantize_uint8;
                    scale = attr.quant_data.affine.tf_scale;
                    zero_point = attr.quant_data.affine.tf_zero_point;
                    break;
                case TACONN_DATA_FORMAT_INT8:
                    dequant = dequantize_int8;
                    scale = attr.quant_data.affine.tf_scale;
                    zero_point = attr.quant_data.affine.tf_zero_point;
                    break;
                default:
                    printf("[FAIL] Unsupported data format for tensor %d: %d\n", i, attr.data_format);
                    free(ref_data);
                    all_pass = false;
                    continue;
            }

            float *act_f32 = (float *)malloc(elem_count * sizeof(float));
            if (!act_f32) {
                printf("malloc for act_f32 failed\n");
                free(ref_data);
                all_pass = false;
                continue;
            }
            for (uint32_t e = 0; e < elem_count; e++) {
                act_f32[e] = dequant(actual, e, zero_point, scale);
            }

            for (uint32_t e = 0; e < elem_count; e++) {
                double r = (double)ref_f32[e];
                double a = (double)act_f32[e];
                dot += r * a;
                norm_ref += r * r;
                norm_act += a * a;
            }
            free(act_f32);
        }

        double cosine = 0.0;
        double denom = sqrt(norm_ref) * sqrt(norm_act);
        if (denom > 1e-12) {
            cosine = dot / denom;
        } else {
            cosine = 1.0;
        }

        printf("Cosine similarity for tensor %d: %.6f\n", i, cosine);
        if (cosine < 0.99) {
            printf("[FAIL] Cosine similarity = %.6f, below threshold 0.99\n", cosine);
            all_pass = false;
        } else {
            printf("[PASS] Tensor %d cosine similarity OK\n", i);
        }

        free(ref_data);
    }

    return all_pass ? 0 : -1;
}

void* inference_thread(void *arg)
{
    test_task_t *task = (test_task_t *)arg;
    int status = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(task->core_mask, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask) != 0) {
        perror("Failed to set CPU affinity for inference thread");
        return NULL;
    }
    for (uint32_t i = 0; i < task->loop_count; i++) {
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);

        status = ta_runtime_run_network(&task->context);
        if (status != 0) {
            printf("run network failed\n");
        }

        gettimeofday(&tv_end, NULL);
        double elapsed = (tv_end.tv_sec - tv_start.tv_sec) +
                         (tv_end.tv_usec - tv_start.tv_usec) / 1e6;

        pthread_mutex_lock(&g_mutex);
        g_frame_count[task->core_mask]++;
        g_total_inference_time[task->core_mask] += elapsed;
        pthread_mutex_unlock(&g_mutex);

        status = ta_runtime_invalidate_buffer(&task->context, task->output);
        if (status != 0) {
            printf("invalidate buffer failed\n");
        }
    }
    return NULL;
}

void* stats_thread(void* arg)
{
    uint32_t last_count[MAX_CORES] = {0, 0};
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(STATS_CORE, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask) != 0) {
        perror("Failed to set CPU affinity for stats thread");
    }
    while (is_run) {
        sleep(1);
        pthread_mutex_lock(&g_mutex);
        printf("FPS - Core0: %d, Core1: %d\n",
               g_frame_count[0] - last_count[0],
               g_frame_count[1] - last_count[1]);
        last_count[0] = g_frame_count[0];
        last_count[1] = g_frame_count[1];
        pthread_mutex_unlock(&g_mutex);
    }
    return NULL;
}

/* ---------- usage ---------- */
void print_usage(const char *prog_name)
{
    printf("Usage: %s -m model.nb [-l loop_count] [-c core]\n", prog_name);
    printf("       %s -m model1.nb model2.nb [-l loop_count]\n", prog_name);
    printf("       %s --context_dir <dir> [-l loop_count]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -m: Model file path(s)\n");
    printf("  -l: Loop count for inference (default: 1)\n");
    printf("  -c: Core selection (only valid for single model)\n");
    printf("      0 - Run on core 0 only\n");
    printf("      1 - Run on core 1 only\n");
    printf("      2 - Run on both cores in parallel (default for single model)\n");
    printf("  --context_dir, -d: Directory containing model.nb, input_*.tensor, output_*.tensor, ...\n");
    printf("                     Mutually exclusive with -m.\n");
    printf("\nExamples:\n");
    printf("  Single model on both cores:  %s -m model.nb -l 100\n", prog_name);
    printf("  With verification:           %s --context_dir ./test_data -l 100\n", prog_name);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    uint32_t loop_count = 1;
    uint32_t network_count = 0;
    char *model_paths[MAX_MODELS] = {0};
    int core_mode = CORE_MODE_DUAL;
    int status = 0;
    test_task_t *task = NULL;
    uint32_t task_count = 0;
    char *context_dir = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            i++;
            while (i < argc && argv[i][0] != '-' && network_count < 2) {
                model_paths[network_count++] = argv[i];
                i++;
            }
            i--;
        } else if (strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                loop_count = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                core_mode = atoi(argv[i + 1]);
                if (core_mode < 0 || core_mode > 2) {
                    printf("Error: Invalid core mode %d (must be 0, 1, or 2)\n", core_mode);
                    exit(1);
                }
                i++;
            }
        } else if (strcmp(argv[i], "--context_dir") == 0 || strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                context_dir = argv[i + 1];
                i++;
            } else {
                printf("Error: --context_dir requires an argument\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    /* 互斥检查 */
    if (context_dir && network_count > 0) {
        printf("Error: -m and --context_dir are mutually exclusive\n");
        exit(1);
    }

    if (!context_dir && network_count == 0) {
        printf("Error: No model file specified. Use -m or --context_dir\n");
        print_usage(argv[0]);
        exit(1);
    }

    /* 如果使用 context_dir，查找 .nb 文件 */
    if (context_dir) {
        char *nb_path = find_nb_file_in_dir(context_dir);
        if (!nb_path) {
            exit(1);
        }
        model_paths[0] = nb_path;
        network_count = 1;
    }

    if (network_count > 2) {
        printf("Error: Maximum 2 models supported\n");
        exit(1);
    }

    /* 确定任务数和核心分配 */
    if (network_count == 1) {
        if (core_mode == CORE_MODE_DUAL) {
            task_count = 2;
            printf("Mode: Single model on both cores (parallel)\n");
        } else {
            task_count = 1;
            printf("Mode: Single model on core %d only\n", core_mode);
        }
    } else {
        task_count = 2;
        printf("Mode: Two models, one per core\n");
    }
    if (context_dir) { task_count = 1; core_mode = 0; }


    printf("Number of models: %d, tasks: %d, loop count: %d\n",
           network_count, task_count, loop_count);

    srand(time(NULL));

    task = (test_task_t *)calloc(task_count, sizeof(test_task_t));
    if (task == NULL) {
        printf("malloc task array failed\n");
        return -1;
    }

    /* Setup tasks */
    if (network_count == 1) {
        if (core_mode == 2) {
            task[0].model_path = model_paths[0];
            task[0].loop_count = loop_count;
            task[0].core_mask = 0;
            task[1].model_path = model_paths[0];
            task[1].loop_count = loop_count;
            task[1].core_mask = 1;
        } else {
            task[0].model_path = model_paths[0];
            task[0].loop_count = loop_count;
            task[0].core_mask = core_mode;
        }
    } else {
        task[0].model_path = model_paths[0];
        task[0].loop_count = loop_count;
        task[0].core_mask = 0;
        task[1].model_path = model_paths[1];
        task[1].loop_count = loop_count;
        task[1].core_mask = 1;
    }

    status = ta_runtime_init();
    if (status != 0) {
        printf("ta_runtime_init failed\n");
        free(task);
        exit(1);
    }

    /* 创建网络并设置输入输出 */
    for (uint32_t i = 0; i < task_count; i++) {
        status = create_network(&task[i], task[i].core_mask);
        printf("create network status = %d, core_mask = %d\n", status, task[i].core_mask);
        if (status != 0) {
            printf("create network failed\n");
            goto Error;
        }

        status = query_io_info(&task[i]);
        if (status != 0) {
            printf("query io info failed\n");
            goto Error;
        }

        if (context_dir) {
            status = set_input_from_file(&task[i], context_dir);
        } else {
            status = set_input_random(&task[i]);
        }
        if (status != 0) {
            printf("set input failed\n");
            goto Error;
        }

        status = set_output(&task[i]);
        if (status != 0) {
            printf("set output failed\n");
            goto Error;
        }
    }

    /* 如果使用 context_dir，先进行一次验证推理 */
    if (context_dir) {
        printf("\n=== Verification Inference ===\n");
        bool all_verified = true;
        for (uint32_t i = 0; i < task_count; i++) {
            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);

            status = ta_runtime_run_network(&task[i].context);
            if (status != 0) {
                printf("Verification inference failed for task %d\n", i);
                all_verified = false;
                continue;
            }

            gettimeofday(&tv_end, NULL);
            double elapsed = (tv_end.tv_sec - tv_start.tv_sec) +
                             (tv_end.tv_usec - tv_start.tv_usec) / 1e6;
            printf("Task %d inference time: %.6f seconds\n", i, elapsed);

            ta_runtime_invalidate_buffer(&task[i].context, task[i].output);

            status = compare_output_with_reference(&task[i], context_dir);
            if (status != 0) {
                printf("[FAIL] Verification failed for task %d\n", i);
                all_verified = false;
            } else {
                printf("[PASS] Output matches reference for task %d\n", i);
            }
        }

        if (!all_verified) {
            printf("Verification FAILED. Exiting.\n");
            status = -1;
            goto Error;
        }
        printf("=== Verification Passed ===\n\n");
        return 0;
    }

    /* 启动统计线程 */
    pthread_t stats_tid;
    pthread_create(&stats_tid, NULL, stats_thread, NULL);

    /* 启动推理线程 */
    pthread_t *thread = (pthread_t *)malloc(sizeof(pthread_t) * task_count);
    if (!thread) {
        printf("malloc thread array failed\n");
        goto Error;
    }
    struct timeval infer_start, infer_end;
    gettimeofday(&infer_start, NULL);
    for (uint32_t i = 0; i < task_count; i++) {

        pthread_create(&thread[i], NULL, inference_thread, &task[i]);
    }

    for (uint32_t i = 0; i < task_count; i++) {
        pthread_join(thread[i], NULL);
    }
    gettimeofday(&infer_end, NULL); 
    is_run = false;
    pthread_join(stats_tid, NULL);
    free(thread);

    /* ===== 打印 avg 统计 ===== */
    printf("\n========== Final Statistics ==========\n");

    uint32_t total_frames = 0;
    for (int c = 0; c < MAX_CORES; c++) {
        if (g_frame_count[c] > 0) {
            double avg_time_ms = (g_total_inference_time[c] / g_frame_count[c]) * 1000.0;
            double avg_fps_core = (avg_time_ms > 0) ? (1000.0 / avg_time_ms) : 0.0;
            printf("Core%d: frames=%u, avg inference time=%.3f ms, avg FPS=%.2f\n",
                   c, g_frame_count[c], avg_time_ms, avg_fps_core);
            total_frames += g_frame_count[c];
        }
    }
    double overall_elapsed = (infer_end.tv_sec - infer_start.tv_sec) +
                           (infer_end.tv_usec - infer_start.tv_usec) / 1e6;
    if (overall_elapsed > 0) { /* overall: wait for thread join */
        printf("Overall_Threads: total frames=%u, overall elapsed=%.3f s, overall avg FPS=%.2f\n",
               total_frames, overall_elapsed, total_frames / overall_elapsed);
    }
    printf("======================================\n");

Error:
    /* 清理 */
    for (uint32_t i = 0; i < task_count; i++) {
        if (task[i].output) {
            for (uint32_t j = 0; j < task[i].output_count; j++) {
                ta_runtime_destroy_buffer(&task[i].context, &task[i].output[j]);
            }
            free(task[i].output);
            task[i].output = NULL;
        }
        if (task[i].input) {
            for (uint32_t j = 0; j < task[i].input_count; j++) {
                if (task[i].input[j].data) {
                    free(task[i].input[j].data);
                    task[i].input[j].data = NULL;
                }
            }
            free(task[i].input);
            task[i].input = NULL;
        }
        if (task[i].context != 0) {
            ta_runtime_destroy_context(&task[i].context);
        }
    }
    if (context_dir && model_paths[0]) {
        free(model_paths[0]); 
    }
    free(task);
    ta_runtime_deinit();
    return status;
}