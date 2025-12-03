---
name: process_resume_skill
description: Process candidate resume PDFs to extract Name, Gender, Nationality and URR status
---

# Process Candidate Resumes

This skill processes candidate resume PDFs and extracts key information using an LLM.

## What it does:
- Extracts text from PDF resumes
- Uses GPT-4o-mini to extract: Name, Gender, Country of Nationality, and URR status
- Saves results to a CSV file

## Usage:
Run the process script with:
```bash
cd /data/home/xiong/dev/Fund_Process_Automation/HR_Review_Bot/src
python process.py
```

## Configuration:
- Requires `.env` file with `OPENAI_API_KEY`
- Input: PDF folder path (default: `/ephemeral/home/xiong/data/Fund/Resumes/current`)
- Output: `candidates_info.csv` in the same folder

## URR Countries:
The script identifies candidates from Under-Represented Regions (URR) based on a predefined list of countries including Afghanistan, Algeria, Angola, China, Egypt, India, Indonesia, Japan, Korea, Nigeria, Pakistan, Philippines, Saudi Arabia, South Africa, Thailand, Vietnam, and many others.

## Notes:
- Uses BSAgent from llm_utils library
- Temperature set to 0 for consistent results
- Handles errors gracefully by marking them in the output CSV
