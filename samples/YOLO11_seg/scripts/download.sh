#!/bin/bash
# YOLO11s-seg 模型和数据下载脚本（阿里云盘）
#
# 下载内容：
#   1. 模型文件 (models/) — ONNX + NB + 配置
#   2. 测试图片 (test_images/)
#   3. 数据集 (datasets/) — COCO val2017 子集（用于精度评估）
#
# 依赖：wget, unzip, curl, jq
#
# 用法：
#   chmod +x scripts/download.sh
#   ./scripts/download.sh
#   ./scripts/download.sh --debug   # 调试模式

set -e

DATASET_URL="https://bj25486.apps.aliyunfile.com/disk/s/JmiMMtjA7Rw?domainId=bj25486"

TEST_IMAGE_URL="https://bj25486.apps.aliyunfile.com/disk/s/44M76miFvAB?domainId=bj25486"

# YOLO11s-seg model files (ONNX + NB + config) — upload to Aliyun Drive and update this URL
# Upload package: /tmp/yolo11s_seg_package.tar.gz (55MB, already prepared)
MODEL_URL="https://bj25486.apps.aliyunfile.com/disk/s/MDc6dfU4hnX?domainId=bj25486"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DEBUG="false"
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG="true"
    fi
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "[DEBUG] $1" >&2
    fi
}

# Check dependencies
for cmd in wget unzip curl jq; do
    if ! command -v $cmd &> /dev/null; then
        log_error "Please install $cmd on your system!"
        exit 1
    fi
done

###############################################################################
# Aliyun Drive download functions (same as YOLOv8_det)
###############################################################################

parse_share_id() {
    local share_url="$1"
    if [[ "$share_url" =~ /s/([a-zA-Z0-9]+)(\?|$) ]]; then
        local share_id="${BASH_REMATCH[1]}"
        echo "$share_id"
        return 0
    else
        log_error "Unable to parse share_id from the share link"
        return 1
    fi
}

extract_domain() {
    local share_url="$1"
    if [[ "$share_url" =~ domainId=([a-zA-Z0-9]+) ]]; then
        local domain="${BASH_REMATCH[1]}"
        echo "$domain"
        return 0
    fi
    local domain=$(echo "$share_url" | sed -E 's|^https?://||')
    local subdomain=$(echo "$domain" | cut -d'.' -f1)
    if [ -z "$subdomain" ]; then
        log_error "Unable to extract domain from the share link"
        return 1
    fi
    echo "$subdomain"
    return 0
}

get_share_token() {
    local domain="$1"
    local share_id="$2"

    log_debug "Getting x-share-token..."

    local api_base="https://${domain}.api.aliyunfile.com"
    local url="${api_base}/v2/share_link/get_share_token"

    local data=$(jq -n --arg share_id "$share_id" '{
        "share_id": $share_id,
        "expire_sec": 7200
    }')

    local headers=(
        "Content-Type: application/json"
        "Accept: application/json"
        "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"
    )

    local temp_response=$(mktemp)
    local http_code
    http_code=$(curl -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -d "$data")

    local response_body
    response_body=$(cat "$temp_response" 2>/dev/null || echo "")
    rm -f "$temp_response" 2>/dev/null

    if ! [[ "$http_code" =~ ^[0-9]+$ ]]; then
        log_error "Invalid HTTP status code: $http_code"
        return 1
    fi

    if [ "$http_code" -eq 200 ]; then
        local share_token=$(echo "$response_body" | jq -r '.share_token')
        if [ "$share_token" != "null" ] && [ -n "$share_token" ]; then
            echo "$share_token"
            return 0
        fi
    fi
    log_error "Failed to get share token"
    return 1
}

get_file_id_from_share() {
    local domain="$1"
    local share_id="$2"
    local share_token="$3"

    local api_base="https://${domain}.api.aliyunfile.com"
    local url="${api_base}/v2/file/list"

    local data=$(jq -n --arg share_id "$share_id" '{
        "share_id": $share_id,
        "parent_file_id": "root",
        "limit": 100
    }')

    local headers=(
        "x-share-token: $share_token"
        "Content-Type: application/json"
        "Accept: application/json"
        "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"
    )

    local temp_response=$(mktemp)
    local http_code
    http_code=$(curl -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -d "$data")

    local response_body
    response_body=$(cat "$temp_response" 2>/dev/null || echo "")
    rm -f "$temp_response" 2>/dev/null

    if ! [[ "$http_code" =~ ^[0-9]+$ ]]; then
        log_error "Invalid HTTP status code: $http_code"
        return 1
    fi

    if [ "$http_code" -eq 200 ]; then
        local file_id=$(echo "$response_body" | jq -r '.items[0].file_id')
        if [ "$file_id" != "null" ] && [ -n "$file_id" ]; then
            echo "$file_id"
            return 0
        else
            log_error "No files found in the share link"
            return 1
        fi
    fi
    log_error "Failed to get file list"
    return 1
}

get_auto_file_id() {
    local share_url="$1"
    local file_type="$2"

    log_info "Auto getting $file_type file ID..."

    local share_id=$(parse_share_id "$share_url")
    [ $? -ne 0 ] && return 1

    local domain=$(extract_domain "$share_url")
    [ $? -ne 0 ] && return 1

    local share_token=$(get_share_token "$domain" "$share_id")
    [ $? -ne 0 ] && return 1

    local file_id=$(get_file_id_from_share "$domain" "$share_id" "$share_token")
    [ $? -ne 0 ] && return 1

    echo "$file_id"
    return 0
}

get_download_url() {
    local domain="$1"
    local file_id="$2"
    local share_id="$3"
    local share_token="$4"

    local api_base="https://${domain}.api.aliyunfile.com"
    local url="${api_base}/v2/file/get_download_url"

    local data=$(jq -n --arg share_id "$share_id" --arg file_id "$file_id" '{
        "share_id": $share_id,
        "file_id": $file_id,
        "expire_sec": 7200
    }')

    local headers=(
        "x-share-token: $share_token"
        "Content-Type: application/json"
        "Accept: application/json,text/plain,*/*"
        "Origin: https://${domain}.apps.aliyunfile.com"
        "Referer: https://${domain}.apps.aliyunfile.com/"
        "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"
    )

    local temp_response=$(mktemp)
    local http_code
    http_code=$(curl --http1.1 -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -H "${headers[3]}" \
        -H "${headers[4]}" \
        -d "$data")

    local response_body
    response_body=$(cat "$temp_response")
    rm -f "$temp_response"

    if ! [[ "$http_code" =~ ^[0-9]+$ ]]; then
        log_error "Invalid HTTP status code: $http_code"
        return 1
    fi

    if [ "$http_code" -eq 200 ]; then
        local download_url=$(echo "$response_body" | jq -r '.url')
        if [ "$download_url" != "null" ] && [ -n "$download_url" ]; then
            echo "$download_url"
            return 0
        fi
    fi

    log_error "Failed to get download URL"
    return 1
}

download_aliyun_file() {
    local share_url="$1"
    local file_id="$2"
    local output="$3"

    local share_id=$(parse_share_id "$share_url")
    [ $? -ne 0 ] && return 1

    local domain=$(extract_domain "$share_url")
    [ $? -ne 0 ] && return 1

    local share_token=$(get_share_token "$domain" "$share_id")
    [ $? -ne 0 ] && return 1

    local download_url=$(get_download_url "$domain" "$file_id" "$share_id" "$share_token")
    [ $? -ne 0 ] && return 1

    log_debug "Downloading to: $output"
    curl -L -o "$output" "$download_url"
    if [ $? -eq 0 ]; then
        log_success "Download successful: $output"
        return 0
    else
        log_error "Download failed: $share_url"
        return 1
    fi
}

###############################################################################
# Main download logic
###############################################################################

ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
log_info "root dir: $ROOT_DIR"

# Download models
log_info "Starting to download model..."
if [[ "$MODEL_URL" == TODO* ]]; then
    log_warning "MODEL_URL not configured yet. Please upload models to Aliyun Drive and update MODEL_URL in this script."
    log_warning "Expected contents: yolo11s_seg.onnx, yolo11s_seg_float16.nb, yolo11s_seg_config_fp16.json, dataset.txt"
else
    MODEL_FILE_ID=$(get_auto_file_id "$MODEL_URL" "model")
    download_aliyun_file "$MODEL_URL" "$MODEL_FILE_ID" "${ROOT_DIR}/models.zip" || {
        log_error "Model download failed"
        exit 1
    }
    unzip "${ROOT_DIR}/models.zip" -d "${ROOT_DIR}"
    rm "${ROOT_DIR}/models.zip"
fi

# Download dataset (reuse YOLOv8_det dataset URL — same COCO val2017 subset)
log_info "Starting to download dataset..."
DATASET_FILE_ID=$(get_auto_file_id "$DATASET_URL" "dataset")
mkdir -p "${ROOT_DIR}/datasets"
download_aliyun_file "$DATASET_URL" "$DATASET_FILE_ID" "${ROOT_DIR}/datasets.zip" || {
    log_error "Dataset download failed"
    exit 1
}
unzip "${ROOT_DIR}/datasets.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/datasets.zip"

log_success "All files downloaded successfully!"
tree -L 2 "${ROOT_DIR}"
