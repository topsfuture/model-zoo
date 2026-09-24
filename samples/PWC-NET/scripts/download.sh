#!/usr/bin/env bash

# Download PWC-Net models and MPI Sintel data from Aliyun Drive.
#
# Please fill in the two Aliyun Drive share links after uploading the
# archives. They can also be overridden with PWC_MODEL_URL and
# PWC_DATASET_URL environment variables.

set -e
set -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_URL="https://bj25486.apps.aliyunfile.com/disk/s/KjGmNjqtnL6?domainId=bj25486"
DATASET_URL="https://bj25486.apps.aliyunfile.com/disk/s/iaUNEszigV9?domainId=bj25486"
if [ -n "${PWC_MODEL_URL:-}" ]; then
    MODEL_URL="$PWC_MODEL_URL"
fi
if [ -n "${PWC_DATASET_URL:-}" ]; then
    DATASET_URL="$PWC_DATASET_URL"
fi
DEBUG="false"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat >&2 <<'EOF'
Usage:
  PWC_MODEL_URL='<Aliyun model share URL>' \
  PWC_DATASET_URL='<Aliyun dataset share URL>' \
  ./scripts/download.sh [--debug]

The model archive must contain a top-level models/ directory.
The dataset archive must contain a top-level sintel/ directory; it will be
extracted to datasets/sintel/.
EOF
}

log_info() {
    echo -e "$BLUE[INFO]$NC $1" >&2
}

log_success() {
    echo -e "$GREEN[SUCCESS]$NC $1" >&2
}

log_error() {
    echo -e "$RED[ERROR]$NC $1" >&2
}

log_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo "[DEBUG] $1" >&2
    fi
}

for arg in "$@"; do
    case "$arg" in
        --debug)
            DEBUG="true"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

for command_name in curl jq unzip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        log_error "Please install $command_name before running this script."
        exit 1
    fi
done

if [ -z "$MODEL_URL" ] || [ -z "$DATASET_URL" ]; then
    log_error "PWC_MODEL_URL and PWC_DATASET_URL must be set."
    usage
    exit 1
fi

parse_share_id() {
    local share_url="$1"
    local share_id
    share_id="$(echo "$share_url" | sed -nE 's|.*/s/([a-zA-Z0-9]+).*|\1|p')"
    if [ -n "$share_id" ]; then
        echo "$share_id"
    else
        log_error "Unable to parse share ID from: $share_url"
        return 1
    fi
}

extract_domain() {
    local share_url="$1"
    local domain
    domain="$(echo "$share_url" | sed -nE 's|.*domainId=([a-zA-Z0-9]+).*|\1|p')"
    if [ -z "$domain" ]; then
        domain="$(echo "$share_url" | sed -E 's|^https?://||' | cut -d/ -f1 | cut -d. -f1)"
    fi
    if [ -z "$domain" ]; then
        log_error "Unable to extract Aliyun domain from: $share_url"
        return 1
    fi
    echo "$domain"
}

get_share_token() {
    local domain="$1"
    local share_id="$2"
    local api_base="https://$domain.api.aliyunfile.com"
    local url="$api_base/v2/share_link/get_share_token"
    local data
    data="$(jq -n --arg share_id "$share_id" \
        '{share_id: $share_id, expire_sec: 7200}')"
    local response
    response="$(curl --http1.1 -fsS -X POST "$url" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json' \
        -H 'User-Agent: Mozilla/5.0' \
        -d "$data")" || {
        log_error "Failed to request share token"
        return 1
    }

    local token
    token="$(echo "$response" | jq -r '.share_token // empty')"
    if [ -z "$token" ]; then
        log_debug "Share-token response: $response"
        log_error "Aliyun did not return a share token"
        return 1
    fi
    echo "$token"
}

get_file_id_from_share() {
    local domain="$1"
    local share_id="$2"
    local share_token="$3"
    local api_base="https://$domain.api.aliyunfile.com"
    local url="$api_base/v2/file/list"
    local data
    data="$(jq -n --arg share_id "$share_id" \
        '{share_id: $share_id, parent_file_id: "root", limit: 100}')"
    local response
    response="$(curl --http1.1 -fsS -X POST "$url" \
        -H "x-share-token: $share_token" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json' \
        -H 'User-Agent: Mozilla/5.0' \
        -d "$data")" || {
        log_error "Failed to list files in Aliyun share"
        return 1
    }

    local file_id
    file_id="$(echo "$response" | jq -r '.items[0].file_id // empty')"
    if [ -z "$file_id" ]; then
        log_debug "File-list response: $response"
        log_error "No file found in Aliyun share"
        return 1
    fi
    echo "$file_id"
}

get_download_url() {
    local domain="$1"
    local file_id="$2"
    local share_id="$3"
    local share_token="$4"
    local api_base="https://$domain.api.aliyunfile.com"
    local url="$api_base/v2/file/get_download_url"
    local data
    data="$(jq -n --arg share_id "$share_id" --arg file_id "$file_id" \
        '{share_id: $share_id, file_id: $file_id, expire_sec: 7200}')"
    local response
    response="$(curl --http1.1 -fsS -X POST "$url" \
        -H "x-share-token: $share_token" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json' \
        -H 'User-Agent: Mozilla/5.0' \
        -H "Origin: https://$domain.apps.aliyunfile.com" \
        -H "Referer: https://$domain.apps.aliyunfile.com/" \
        -d "$data")" || {
        log_error "Failed to request Aliyun download URL"
        return 1
    }

    local download_url
    download_url="$(echo "$response" | jq -r '.url // empty')"
    if [ -z "$download_url" ]; then
        log_debug "Download-url response: $response"
        log_error "Aliyun did not return a download URL"
        return 1
    fi
    echo "$download_url"
}

get_auto_file_id() {
    local share_url="$1"
    local file_type="$2"
    log_info "Getting $file_type file ID..."
    local share_id
    share_id="$(parse_share_id "$share_url")"
    local domain
    domain="$(extract_domain "$share_url")"
    local share_token
    share_token="$(get_share_token "$domain" "$share_id")"
    get_file_id_from_share "$domain" "$share_id" "$share_token"
}

download_aliyun_file() {
    local share_url="$1"
    local file_id="$2"
    local output="$3"
    local share_id
    share_id="$(parse_share_id "$share_url")"
    local domain
    domain="$(extract_domain "$share_url")"
    local share_token
    share_token="$(get_share_token "$domain" "$share_id")"
    local download_url
    download_url="$(get_download_url "$domain" "$file_id" "$share_id" "$share_token")"

    log_info "Downloading to $output"
    curl --fail --location --retry 3 --output "$output" "$download_url"
    log_success "Downloaded $output"
}

WORK_DIR="$(mktemp -d /tmp/pwc-net-download.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

log_info "PWC-NET root: $ROOT_DIR"
MODEL_ARCHIVE="$WORK_DIR/models.zip"
DATASET_ARCHIVE="$WORK_DIR/sintel.zip"

MODEL_FILE_ID="$(get_auto_file_id "$MODEL_URL" "model")"
download_aliyun_file "$MODEL_URL" "$MODEL_FILE_ID" "$MODEL_ARCHIVE"
unzip -oq "$MODEL_ARCHIVE" -d "$ROOT_DIR"

DATASET_FILE_ID="$(get_auto_file_id "$DATASET_URL" "MPI Sintel dataset")"
download_aliyun_file "$DATASET_URL" "$DATASET_FILE_ID" "$DATASET_ARCHIVE"
unzip -oq "$DATASET_ARCHIVE" -d "$ROOT_DIR"

if [ ! -d "$ROOT_DIR/models" ]; then
    log_error "Model archive must contain a top-level models/ directory."
    exit 1
fi
if [ ! -d "$ROOT_DIR/datasets/sintel/training" ]; then
    log_error "Dataset archive must contain sintel/training/ under its root."
    exit 1
fi

log_success "Models and MPI Sintel dataset are ready."
log_info "Next: python3 scripts/make_sintel_manifests.py"
log_info "Next: python3 scripts/prepare_sintel_board_subset.py"
