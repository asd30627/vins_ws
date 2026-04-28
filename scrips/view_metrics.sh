#!/bin/bash

SATA_BASE="/mnt/sata4t/ivlab3_data/vins_project"
MODE=${1:-"fog"} # 預設查看 fog，可用引數指定 imu
RESULTS_DIR="${SATA_BASE}/results/kaist/${MODE}"

echo "==================== [ 模式: ${MODE^^} ] ===================="
printf "%-20s | %-12s | %-12s | %-12s | %-12s\n" "Sequence" "VIO APE" "VIO RPE" "LOOP APE" "LOOP RPE"
echo "------------------------------------------------------------------------------------"

get_rmse() {
    local file=$1
    if [[ -f "$file" ]]; then
        local val=$(grep -i "rmse" "$file" | awk '{print $2}')
        if [[ -n "$val" ]]; then
            printf "%.4f" "$val"
            return
        fi
    fi
    echo "N/A"
}

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "找不到 ${MODE} 模式的結果資料夾。"
    exit 1
fi

for seq_dir in "${RESULTS_DIR}"/*; do
  if [ -d "$seq_dir" ]; then
    seq_name=$(basename "$seq_dir")
    ape_vio=$(get_rmse "${seq_dir}/metrics/ape_vio.txt")
    rpe_vio=$(get_rmse "${seq_dir}/metrics/rpe_vio.txt")
    ape_loop=$(get_rmse "${seq_dir}/metrics/ape_loop.txt")
    rpe_loop=$(get_rmse "${seq_dir}/metrics/rpe_loop.txt")
    
    printf "%-20s | %-12s | %-12s | %-12s | %-12s\n" "$seq_name" "$ape_vio" "$rpe_vio" "$ape_loop" "$rpe_loop"
  fi
done
echo "===================================================================================="
