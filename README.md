# Fund Process Automation

Automation tools for fund management processes: resume analysis, document parsing, and technical assessment evaluation.

## Structure

- **HR_Review_Bot/**: Resume analysis and candidate information extraction
- **PDF_Parsing/**: PDF to text conversion using MinerU  
- **libs/**: Shared utilities (LLM integration, logging, cost tracking)

## Setup

```bash
conda activate traction
```

Create `.env` file with OpenAI API key.

## Usage

```bash
# Resume processing
python HR_Review_Bot/src/process.py

# PDF conversion
python PDF_Parsing/MinerU_convert_pdfs.py --input_dir /path/to/pdfs

# Technical evaluation
/eval_test ## use claude code
```
