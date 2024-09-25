#!/bin/bash

# 指定要保留的檔案名稱列表
keep_files=(
    "fitness_explore.txt"
    "fitness_local_search.txt"
    "generation_all_best.txt"
    "population_explore.txt"
    "population_local_search.txt"
    "cal_dict.pkl"
    "cal_results.csv"
    "predict_dict.pkl"
    "predict_results.csv"
)

# 要處理的根資料夾，預設當前資料夾
root_dir="RESULTS/"

# 遞迴檢查並刪除不需要的檔案
function clean_directory() {
    local dir="$1"
    
    # 進入該目錄
    cd "$dir" || exit

    # 列出目錄內所有檔案和子資料夾
    for file in *; do
        if [ -d "$file" ]; then
            # 如果是資料夾，遞迴處理
            clean_directory "$file"
        elif [ -f "$file" ]; then
            # 如果是檔案，檢查是否在保留清單內
            keep=false
            for keep_file in "${keep_files[@]}"; do
                if [ "$file" == "$keep_file" ]; then
                    keep=true
                    break
                fi
            done
            # 刪除不在保留清單內的檔案
            if [ "$keep" == false ]; then
                echo "Removing: $dir/$file"
                rm "$file"
            fi
        fi
    done

    # 返回上一層目錄
    cd ..
}

# 開始處理
clean_directory "$root_dir"
