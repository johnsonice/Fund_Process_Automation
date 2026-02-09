#!/bin/bash
#source .venv/bin/activate

input_base_dir="/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output/Program_2025-2026_01_update_PDF"
output_base_dir="/data/home/xiong/data/Fund/pdf_parse/Fund_Document/output_organized/Program_2025-2026_01_update_PDF"

mkdir -p "$output_base_dir"

python /data/home/xiong/dev/Fund_Process_Automation/PDF_Parsing/post_process_results.py \
  --input_base_dir "$input_base_dir" \
  --output_base_dir "$output_base_dir"
