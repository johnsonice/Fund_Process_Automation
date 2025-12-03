---
description: Generate a markdown summary report from candidate_profile.csv
---

# Generate Candidate Summary

This skill generates a comprehensive markdown summary report from the candidate profile CSV data.

## What it does:
- Reads `candidate_profile.csv`
- Calculates statistics on gender distribution, URR representation, and nationality diversity
- Creates a formatted markdown report with tables and insights

## Usage:
Run the summary generation script with:
```bash
cd /data/home/xiong/dev/Fund_Process_Automation/HR_Review_Bot/src
python generate_summary.py
```

## Input:
- File: `/data/home/xiong/dev/Fund_Process_Automation/candidate_profile.csv`
- Required columns: Gender, URR, Country of Nationality

## Output:
- File: `/data/home/xiong/dev/Fund_Process_Automation/summary.md`
- Includes:
  - Total candidates processed
  - Gender distribution (Male/Female/Unknown) with percentages
  - URR vs Non-URR distribution
  - Top 10 nationalities with counts
  - Key insights and URR countries identified

## Example Summary Sections:
- Overview with total count
- Summary statistics tables
- Key insights (gender balance, URR representation, geographic diversity)
- List of URR countries represented

## Notes:
- Requires pandas library
- Generates markdown tables with clean formatting
- Includes percentages and absolute counts for all metrics
