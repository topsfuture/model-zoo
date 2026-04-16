#! /bin/bash

DATASET_URL="https://bj25486.apps.aliyunfile.com/disk/s/ztHjiEjZ2tk?domainId=bj25486"

TEST_IMAGE_URL="https://bj25486.apps.aliyunfile.com/disk/s/44M76miFvAB?domainId=bj25486"

MODEL_URL="https://bj25486.apps.aliyunfile.com/disk/s/StNnYsviMzV?domainId=bj25486"



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


# Log functions
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

# Debug log function
log_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "[DEBUG] $1" >&2
    fi
}

res=$(which wget)
if [ $? != 0 ];
then 
    log_error "Please install wget on your system!"
    exit 1
fi

res=$(which unzip)
if [ $? != 0 ];
then
    log_error "Please install unzip on your system!"
    exit 1
fi

res=$(which curl)
if [ $? != 0 ];
then
    log_error "Please install curl on your system!"
    exit 1
fi

res=$(which jq)
if [ $? != 0 ];
then
    log_error "Please install jq on your system!"
    exit 1
fi


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
    log_debug "Getting share token via official API..."
    
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
    
    log_debug "Calling API: $url"
    log_debug "Request parameters: $data"
    
    # Use temporary file to store response
    local temp_response=$(mktemp)
    local temp_headers=$(mktemp)
    
    # POST
    local http_code
    http_code=$(curl -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -d "$data" \
        --dump-header "$temp_headers")
    
    local response_body
    response_body=$(cat "$temp_response" 2>/dev/null || echo "")
    rm -f "$temp_response" 2>/dev/null
    log_debug "Raw HTTP status code: $http_code"
    # Security check for HTTP status code
    if ! [[ "$http_code" =~ ^[0-9]+$ ]]; then
        log_error "Invalid HTTP status code: $http_code"
        log_debug "Response body: $response_body"
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

# 新增函数：获取文件ID
get_file_id_from_share() {
    local domain="$1"
    local share_id="$2"
    local share_token="$3"
    
    log_debug "Getting file ID from share..."
    
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
    
    log_debug "Calling list files API: $url"
    log_debug "Request parameters: $data"
    
    # Use temporary file to store response
    local temp_response=$(mktemp)
    local temp_headers=$(mktemp)
    
    # POST
    local http_code
    http_code=$(curl  -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -d "$data" \
        --dump-header "$temp_headers")
    
    local response_body
    response_body=$(cat "$temp_response" 2>/dev/null || echo "")
    rm -f "$temp_response" 2>/dev/null
    log_debug "Raw HTTP status code: $http_code"
    # Security check for HTTP status code
    if ! [[ "$http_code" =~ ^[0-9]+$ ]]; then
        log_error "Invalid HTTP status code: $http_code"
        log_debug "Response body: $response_body"
        return 1
    fi

    
    if [ "$http_code" -eq 200 ]; then
        local file_id=$(echo "$response_body" | jq -r '.items[0].file_id')
        if [ "$file_id" != "null" ] && [ -n "$file_id" ]; then
            local file_name=$(echo "$response_body" | jq -r '.items[0].name')
            log_debug "Found file: $file_name with ID: $file_id"
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

# 新增函数：自动获取文件ID
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
log_debug "share_token:  $share_token"
    
    local headers=(
        "x-share-token: $share_token"
        "Content-Type: application/json"
        "Accept: application/json,text/plain,*/*"
        "Origin: https://${domain}.apps.aliyunfile.com"
        "Referer: https://${domain}.apps.aliyunfile.com/"
        "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"
    )   
    
    local temp_response=$(mktemp)
    local temp_headers=$(mktemp)
    
    log_debug "Sending request to: $url"
    
    local http_code
    http_code=$(curl --http1.1 -s -o "$temp_response" -w "%{http_code}" -X POST "$url" \
        -H "${headers[0]}" \
        -H "${headers[1]}" \
        -H "${headers[2]}" \
        -H "${headers[3]}" \
        -H "${headers[4]}" \
        -d "$data" \
        --dump-header "$temp_headers")
    local response_body
    response_body=$(cat "$temp_response")
    
    rm -f "$temp_response" "$temp_headers"
    log_debug "Response status code: $http_code"
    
    # Security check for HTTP status code
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
    
    log_debug "Starting to process Aliyun download: $share_url"
    
    local share_id=$(parse_share_id "$share_url")
    [ $? -ne 0 ] && return 1
    
    local domain=$(extract_domain "$share_url")
    [ $? -ne 0 ] && return 1
    
    local share_token=$(get_share_token "$domain" "$share_id")
    [ $? -ne 0 ] && return 1
    
    local download_url=$(get_download_url "$domain" "$file_id" "$share_id" "$share_token")
    [ $? -ne 0 ] && return 1
    
    log_debug "Starting to download file to: $output"
    curl -L -o "$output" "$download_url"
    if [ $? -eq 0 ]; then
        log_success "Download successful: $output"
        return 0
    else
        log_error "Download failed: $share_url"
        return 1
    fi
}

ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
log_info "root dir: $ROOT_DIR"

log_info "Starting to download dataset..."
DATASET_FILE_ID=$(get_auto_file_id "$DATASET_URL" "dataset")
mkdir -p "${ROOT_DIR}/datasets"
download_aliyun_file "$DATASET_URL" "$DATASET_FILE_ID" "${ROOT_DIR}/datasets.zip" || { 
    log_error "Dataset download failed"
    exit 1 
}
unzip "${ROOT_DIR}/datasets.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/datasets.zip"


log_info "Starting to download model..."
MODEL_FILE_ID=$(get_auto_file_id "$MODEL_URL" "model")
download_aliyun_file "$MODEL_URL" "$MODEL_FILE_ID" "${ROOT_DIR}/models.zip" || { 
    log_error "Model download failed"
    exit 1 
}
unzip "${ROOT_DIR}/models.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/models.zip"

log_info "Starting to download test_image..."
TEST_IMAGE_FILE_ID=$(get_auto_file_id "$TEST_IMAGE_URL" "test image")
download_aliyun_file "$TEST_IMAGE_URL" "$TEST_IMAGE_FILE_ID" "${ROOT_DIR}/test_image.zip" || { 
    log_error "test_image download failed"
    exit 1 
}
unzip "${ROOT_DIR}/test_image.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/test_image.zip"



log_success "All files downloaded successfully!"
tree -L 2 "${ROOT_DIR}"