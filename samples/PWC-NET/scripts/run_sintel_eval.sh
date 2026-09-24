#!/bin/bash
# MPI Sintel Clean Evaluation on EA6530
# Reads manifest_imgpair.txt, runs PWC-Net, evaluates against GT.
#
# Usage:
#   ./run_sintel_eval.sh                              # uses default 960x512 NBG
#   ./run_sintel_eval.sh --model /path/to/model.nb  # custom NBG

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROGDIR="${PROGDIR:-$SCRIPT_DIR}"
PREDSDIR="$PROGDIR/sintel_preds"
RESULTDIR="$PROGDIR/sintel_results"
MANIFEST="/tmp/sintel_eval/manifest_imgpair.txt"
EVAL_MANIFEST="$PROGDIR/eval_manifest.txt"

# Default model (can override with --model)
MODEL="$PROGDIR/pwc_net_960x512_float16.nb"

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model /path/to/model.nb]"
            echo "  Default model: $PROGDIR/pwc_net_960x512_float16.nb"
            exit 0
            ;;
        *)
            echo "Unknown: $1"
            exit 1
            ;;
    esac
done

set -e

mkdir -p "$PREDSDIR" "$RESULTDIR"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL"
    exit 1
fi

TOTAL_PAIRS=$(grep -c '' "$MANIFEST" 2>/dev/null || echo 0)
echo "=== MPI Sintel Evaluation ==="
echo "Model:   $MODEL"
echo "Pairs:   $TOTAL_PAIRS"
echo ""

# --- Build eval manifest (pred_path gt_path) ---
echo "[1/3] Building eval manifest..."
> "$EVAL_MANIFEST"
while IFS= read -r line || [ -n "$line" ]; do
    set -- $line
    img0="$1"; img1="$2"; gt="$3"
    # Extract unique name from GT path: e.g. alley_1_frame_0001
    gt_sub=$(echo "$gt" | sed 's|/tmp/sintel_eval/training/flow/||')
    name="${gt_sub%.flo}"
    name=$(echo "$name" | tr '/' '_')
    pred_file="$PREDSDIR/${name}.flo"
    echo "$pred_file $gt" >> "$EVAL_MANIFEST"
done < "$MANIFEST"

# --- Run inference and collect timing ---
echo "[2/3] Running PWC-Net inference ($TOTAL_PAIRS pairs)..."
echo "       (Timing: disk read + preprocess + NBG + write)"

> "$RESULTDIR/pwc_timing.csv"
> "$RESULTDIR/pwc_nbg_timing.csv"
> "$RESULTDIR/pwc_profile.log"
count=0
failed=0
total_time=0
total_nbg=0
nbg_samples=0

while IFS= read -r line || [ -n "$line" ]; do
    count=$((count + 1))
    set -- $line
    img0="$1"; img1="$2"; gt="$3"
    gt_sub=$(echo "$gt" | sed 's|/tmp/sintel_eval/training/flow/||')
    name="${gt_sub%.flo}"
    name=$(echo "$name" | tr '/' '_')
    pred_file="$PREDSDIR/${name}.flo"

    t0=$(python3 -c 'import time; print(time.time())')
    $PROGDIR/pwc_net_imgpair --model "$MODEL" "$img0" "$img1" > "$pred_file" 2>> "$RESULTDIR/pwc_profile.log"
    status=$?
    t1=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(f'{(float('$t1') - float('$t0')) * 1000:.0f}')")

    if [ $status -eq 0 ] && [ -s "$pred_file" ]; then
        echo "$elapsed" >> "$RESULTDIR/pwc_timing.csv"
        total_time=$(python3 -c "print($total_time + float('$elapsed'))")
        nbg_ms=$(tail -n 20 "$RESULTDIR/pwc_profile.log" | grep '^pwc_nbg_ms=' | tail -1 | sed 's/^pwc_nbg_ms=//' | awk '{print $1}')
        if [ -n "$nbg_ms" ]; then
            echo "$nbg_ms" >> "$RESULTDIR/pwc_nbg_timing.csv"
            total_nbg=$(python3 -c "print($total_nbg + float('$nbg_ms'))")
            nbg_samples=$((nbg_samples + 1))
        fi
    else
        failed=$((failed + 1))
        echo "  FAIL pair $count: $name (exit $status)"
    fi

    if [ $((count % 20)) -eq 0 ]; then
        avg=$(python3 -c "print(f'{$total_time / $count:.0f}')")
        echo "  Progress: $count / $TOTAL_PAIRS (avg ${avg}ms/pair)"
    fi
done < "$MANIFEST"

echo ""
if [ -f "$RESULTDIR/pwc_timing.csv" ]; then
    avg_time=$(python3 -c "with open('$RESULTDIR/pwc_timing.csv') as f: vals=[float(x) for x in f]; print(f'{sum(vals)/len(vals):.1f}') if vals else 'N/A'")
    total_inf=$(python3 -c "with open('$RESULTDIR/pwc_timing.csv') as f: vals=[float(x) for x in f]; print(f'{sum(vals):.1f}') if vals else 'N/A'")
else
    avg_time="N/A"; total_inf="N/A"
fi
echo "Inference complete: avg=${avg_time}ms/pair ($failed failed)"
if [ "$nbg_samples" -gt 0 ]; then
    avg_nbg=$(python3 -c "print(f'{$total_nbg / $nbg_samples:.3f}')")
else
    avg_nbg="N/A"
fi
echo "Pure NBG: avg=${avg_nbg}ms/pair (${nbg_samples} samples)"

# --- Evaluate ---
echo ""
echo "[3/3] Running evaluation..."
t0=$(python3 -c 'import time; print(time.time())')
$PROGDIR/pwc_net_eval --list "$EVAL_MANIFEST" --summary "$RESULTDIR/eval_summary.csv" 2>&1 | tee "$RESULTDIR/eval_output.log"
t1=$(python3 -c 'import time; print(time.time())')
eval_time=$(python3 -c "print(f'{(float('$t1') - float('$t0')) * 1000:.1f}')")

# Extract metrics
pairs=$(grep '^pairs=' "$RESULTDIR/eval_output.log" | tail -1 | sed 's/pairs=//')
epe=$(grep 'mean_epe=' "$RESULTDIR/eval_output.log" | tail -1 | sed 's/mean_epe=//')
bad3=$(grep 'bad3_percent=' "$RESULTDIR/eval_output.log" | tail -1 | sed 's/bad3_percent=//')
ang=$(grep 'mean_angular_error_deg=' "$RESULTDIR/eval_output.log" | tail -1 | sed 's/mean_angular_error_deg=//')
bad5=$(grep 'bad5_percent=' "$RESULTDIR/eval_output.log" | tail -1 | sed 's/bad5_percent=//')

echo ""
echo "============================================================"
echo "         MPI Sintel Clean Evaluation"
echo "============================================================"
echo "Model:         $MODEL"
echo "Pairs:         $pairs"
echo "Failed:        $failed"
echo ""
echo "--- Timing ---"
echo "Avg inference: ${avg_time}ms/frame"
echo "Total infer:  ${total_inf}ms"
echo "Avg pure NBG: ${avg_nbg}ms/frame"
echo "Eval time:    ${eval_time}ms"
echo ""
echo "--- Accuracy (native prediction resolution; GT downsampled) ---"
echo "mean EPE:     ${epe}"
echo "mean Angular:  ${ang} deg"
echo "bad3%:         ${bad3}"
echo "bad5%:         ${bad5}"
echo "============================================================"

# Save summary
cat > "$RESULTDIR/final_summary.txt" << SUMMARY
model=$MODEL
pairs=$pairs
failed=$failed
avg_inference_ms=$avg_time
total_inference_ms=$total_inf
avg_pure_nbg_ms=$avg_nbg
eval_ms=$eval_time
mean_epe=$epe
mean_angular_deg=$ang
bad3_percent=$bad3
bad5_percent=$bad5
SUMMARY

echo ""
echo "Results:   $RESULTDIR/final_summary.txt"
echo "CSV:       $RESULTDIR/eval_summary.csv"
