## bash strcip to run the script
#!/bin/bash

# Set CUDA device 0 as visible and hide other devices
export CUDA_VISIBLE_DEVICES=1
echo "Using CUDA device 1"

python MinerU_convert_pdfs.py \
--input_dir /ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/input/All_AIV_before_2008_PDF \
--output_dir /ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/All_AIV_before_2008_json_md \
--n_workers 1
