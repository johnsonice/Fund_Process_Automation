## bash strcip to run the script
#!/bin/bash

# Set CUDA device 0 as visible and hide other devices
export CUDA_VISIBLE_DEVICES=2
echo "Using CUDA device 2"

python MinerU_convert_pdfs.py \
--input_dir /ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/input/Program_PDF \
--output_dir /ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/Program_json_md \
--n_workers 8
