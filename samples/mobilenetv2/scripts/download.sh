#!/bin/bash

# 检查必要工具
res=$(which wget)
if [ $? != 0 ];
then
    echo "Please install wget on your system!"
    exit 1
fi

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit 1
fi

# 下载函数，带错误检查
download_with_retry() {
    local url=$1
    local output=$2
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        wget --no-check-certificate -O "$output" "$url"
        if [ $? -eq 0 ]; then
            echo "Download successful: $url"
            return 0
        fi
        
        retry_count=$((retry_count+1))
        echo "Download failed (attempt $retry_count/$max_retries): $url"
        sleep 2
    done
    
    echo "Failed to download after $max_retries attempts: $url"
    return 1
}

ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
echo "root dir: $ROOT_DIR"


DOWNLOAD_MODEL_URL="https://models.topsfuture.com.cn/models/mobilenetv2/models.zip"
DOWNLOAD_DATASET_URL="https://models.topsfuture.com.cn/datasets/imagenet/datasets.zip"

echo "开始下载：$DOWNLOAD_MODEL_URL"
curl -o "models.zip" "$DOWNLOAD_MODEL_URL" || { 
    echo "错误：下载失败"
    exit 1 
}
unzip "${ROOT_DIR}/models.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/models.zip"

echo "开始下载：$DOWNLOAD_DATASET_URL"
curl -o "datasets.zip" "$DOWNLOAD_DATASET_URL" || { 
    echo "错误：下载失败"
    exit 1 
}
unzip "${ROOT_DIR}/datasets.zip" -d "${ROOT_DIR}"
rm "${ROOT_DIR}/datasets.zip"


echo "done."
echo "${ROOT_DIR}"
tree -L 2 "${ROOT_DIR}"
