## bash strcip to run the script
#!/bin/bash

source .venv/bin/activate

# Set CUDA device 0 as visible and hide other devices
export CUDA_VISIBLE_DEVICES=1
echo "Using CUDA device 1"


# Ensure Hugging Face cache is writable (avoid PermissionError under ~/.cache)
export HF_HOME="/data/home/xiong/data/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

input_dir="/data/home/xiong/data/Fund/pdf_parse/Fund_Document/input/Program_2025-2026_01_update_PDF"
output_dir=/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output/Program_2025-2026_01_update_PDF
mkdir -p "$output_dir"
mineru -p "$input_dir" -o "$output_dir"


# # Process all PDFs recursively under input_dir, placing results under output_dir
# find "$input_dir" -type f -iname '*.pdf' -print0 | while IFS= read -r -d '' pdf_path; do
# rel_path="${pdf_path#$input_dir/}"
# rel_dir="$(dirname "$rel_path")"
# base_name="$(basename "$pdf_path" .pdf)"
# dest_dir="$output_dir/$rel_dir/$base_name"
# mkdir -p "$dest_dir"
# echo "Processing: $pdf_path -> $dest_dir"
# mineru -p "$pdf_path" -o "$dest_dir"
# done

