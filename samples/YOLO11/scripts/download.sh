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


DOWNLOAD_MODEL_URL="https://pub-2bc59661537449fcb5161f11c05ebb6f.r2.dev/models/yolo11/models.zip"
DOWNLOAD_DATASET_URL="https://pub-2bc59661537449fcb5161f11c05ebb6f.r2.dev/datasets/coco_val_2017/datasets.zip"
DOWNLOAD_TEST_IMAGE_URL="https://pub-2bc59661537449fcb5161f11c05ebb6f.r2.dev/test_image/yolo11/test_images.zip"

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

echo "开始下载：$DOWNLOAD_TEST_IMAGE_URL"
curl -o "test_images.zip" "$DOWNLOAD_TEST_IMAGE_URL" || { 
    echo "错误：下载失败"
    exit 1 
}
unzip "${ROOT_DIR}/test_images.zip" -d "${ROOT_DIR}/test_images"
rm "${ROOT_DIR}/test_images.zip"


echo "done."
echo "${ROOT_DIR}"
tree -L 2 "${ROOT_DIR}"
