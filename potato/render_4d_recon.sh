#!/bin/bash
export PYTHONPATH=/kubric:$PYTHONPATH
export KUBRIC_USE_GPU=0

config_file=$1
output_dir_base=$2
iter=$3
num_parallel=${4:-1}  # default to 1 if not provided

# Timestamp for this render batch
DATE_STR=$(date +"%Y%m%d_%H%M%S")

# Directories
LOGDIR="potato/logs/${DATE_STR}"
mkdir -p "$LOGDIR"
MAIN_LOG="${LOGDIR}/render_main.txt"


# Convert JSON to CLI args
json_to_args() {
    local config_file=$1
    python - <<EOF
import json, sys
cfg = json.load(open("$config_file"))
args = []
for k, v in cfg.items():
    # replace underscores with hyphen for CLI consistency
    key = "--" + k
    if isinstance(v, bool):
        if v: args.append(key)
    else:
        args.append(f"{key} {v}")
print(" ".join(args))
EOF
}

CONFIG_ARGS=$(json_to_args $config_file)

run_render() {
    local job_dir=$1
    local job_idx=$2
    local LOGFILE="${LOGDIR}/render_job_${job_idx}.txt"

    local start_time=$(date +%s)
    CMD="python potato/scripts/4d_rendering.py $CONFIG_ARGS --job-dir $job_dir"

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [INFO] Starting job $job_idx → $job_dir" | tee -a "$MAIN_LOG"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [INFO] Running: $CMD" | tee -a "$LOGFILE"

    { time eval "$CMD"; } &>> "$LOGFILE"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] [INFO] Job $job_idx finished in ${duration}s" | tee -a "$MAIN_LOG"

}

# Parallel loop
i=0
while [ $i -lt $iter ]; do
    running_jobs=0
    while [ $running_jobs -lt $num_parallel ] && [ $i -lt $iter ]; do
        output_dir="${output_dir_base}/scene${i}"
        run_render "$output_dir" "$i" &
        ((i++))
        ((running_jobs++))
    done
    wait  # wait for this batch to finish
done

