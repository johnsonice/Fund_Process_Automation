# Process Candidate Resumes

You are helping to process candidate resume PDFs to extract key information and generate a comprehensive summary report. This workflow uses two specialized skills to automate the entire process.

## Overview

This command orchestrates two skills:
1. **process_resume_skill**: Extracts candidate information from PDF resumes
2. **generate_candidate_summary_skill**: Creates a comprehensive markdown summary report

## Environment Setup

**IMPORTANT:** All commands must be run in the `agent` conda environment:

```bash
conda activate agent
```

Verify the environment is active before proceeding with any steps.

## Step 1: Process Resume PDFs (use process_resume_skill)

Execute the `process_resume_skill` to extract candidate information from all PDF resumes:

```bash
conda activate agent
python .claude/skills/process_resume_skill/scripts/process_resume.py --pdf_folder /data/home/xiong/data/Fund/Resumes/current/candidates --output_file /data/home/xiong/data/Fund/Resumes/current/candidate_profile.csv
```

**What this skill does:**
- Recursively finds all PDF files in the folder (handles nested folders)
- Processes PDFs directly using Google Gemini API (handles both text-based and image-based PDFs)
- Uses Gemini 2.0 Flash to extract: Name, Gender, Nationality, and URR status
- Validates URR status against the official URR country list
- Saves results to `candidate_profile.csv`
- Displays processing statistics in console

**Expected Output:**
- CSV file: `/data/home/xiong/data/Fund/Resumes/current/candidate_profile.csv`
- Columns: Name, Gender, Country of Nationality, URR
- Console statistics showing total candidates, gender distribution, and URR counts

**Reference:** See [.claude/skills/process_resume_skill/SKILL.md](.claude/skills/process_resume_skill/SKILL.md) for detailed documentation.

## Step 2: Generate Summary Report (use generate_candidate_summary_skill)

After Step 1 completes successfully, generate a comprehensive markdown summary report:

```bash
conda activate agent
python .claude/skills/generate_candidate_summary_skill/generate_summary.py --csv_file /data/home/xiong/data/Fund/Resumes/current/candidate_profile.csv --output_file /data/home/xiong/data/Fund/Resumes/current/summary.md
```

**What this skill does:**
- Reads the candidate profile CSV generated in Step 1
- Calculates comprehensive statistics:
  - Gender distribution (Male/Female/Unknown) with percentages
  - URR vs Non-URR distribution with percentages
  - Top 10 nationalities with counts and URR status
  - Geographic diversity metrics
- Generates formatted markdown report with tables and insights
- Lists all URR countries represented in the candidate pool

**Expected Output:**
- Markdown file: `/data/home/xiong/data/Fund/Resumes/current/summary.md`
- Contains: Overview, summary statistics tables, key insights, URR countries list
- Console confirmation showing total candidates and distribution statistics

**Reference:** See [.claude/skills/generate_candidate_summary_skill/SKILL.md](.claude/skills/generate_candidate_summary_skill/SKILL.md) for detailed documentation.

## Step 3: Clean Up

After both skills complete successfully, remove any intermediate files:
- Temporary text files from PDF extraction
- PDF conversion artifacts in `/tmp` or working directory
- Any other temporary processing files

**Note:** The main output files should be preserved:
- Keep: `candidate_profile.csv` (raw data)
- Keep: `summary.md` (summary report)

## Workflow Execution

When executing this command, you should:

1. **Activate conda environment:**
   - Run `conda activate agent` before any processing commands
   - Verify the environment is active

2. **Run Step 1** (process_resume_skill):
   - Execute the bash command to process all PDF resumes
   - Wait for completion and verify the CSV file was created
   - Review console output for any errors

3. **Run Step 2** (generate_candidate_summary_skill):
   - Execute the bash command to generate the summary report
   - Verify the markdown file was created
   - Review console output for statistics

4. **Run Step 3** (Clean up):
   - Identify and remove temporary files
   - Preserve the final output files (CSV and summary.md)

5. **Report to user**:
   - Confirm successful completion of all steps
   - Provide file paths for the generated outputs
   - Share key statistics from the summary

## Expected Outputs

**Files Generated:**
1. `/data/home/xiong/data/Fund/Resumes/current/candidate_profile.csv`
   - Raw candidate data with Name, Gender, Country of Nationality, URR columns

2. `/data/home/xiong/data/Fund/Resumes/current/summary.md`
   - Comprehensive markdown report with:
     - Overview and total candidate count
     - Gender distribution table with percentages
     - URR distribution table with percentages
     - Top 10 nationalities with URR status
     - Key insights (gender balance, URR representation, diversity)
     - List of URR countries represented

## Important Context

**URR Country List** (these countries are considered Under-Represented Regions):
Afghanistan; Algeria; Angola; Bahrain; Benin; Botswana; Brunei Darussalam; Burkina Faso; Cabo Verde; Cambodia; Cameroon; Central African Republic; Chad; China; Comoros; Côte d'Ivoire; Democratic Republic of the Congo; Djibouti; Egypt; Equatorial Guinea; Eritrea; Ethiopia; Gabon; Ghana; Guinea; Guinea-Bissau; Hong Kong SAR; Indonesia; Iran; Iraq; Japan; Jordan; Kenya; Korea; Kuwait; Lao P.D.R.; Lebanon; Lesotho; Liberia; Libya; Macao SAR; Madagascar; Malawi; Malaysia; Mali; Mauritania; Mauritius; Morocco; Mozambique; Myanmar; Namibia; Niger; Nigeria; Oman; Pakistan; Philippines; Qatar; Republic of Congo; Rwanda; São Tomé and Príncipe; Saudi Arabia; Senegal; Seychelles; Sierra Leone; Singapore; Somalia; South Africa; South Sudan; Sudan; Swaziland; Syria; Tanzania; Thailand; The Gambia; Togo; Tunisia; Uganda; Vietnam; West Bank & Gaza; Yemen; Zambia; Zimbabwe

**Data Validation:**
- Gender values: Male, Female, or Unknown
- URR values: "yes" or "no" (lowercase)
- Nationality: Country name or "Unknown"
- All fields should have proper capitalization and no extra whitespace

## Prerequisites

**Required:**
- **Conda environment:** `agent` environment must be activated
- **Google API key:** Configured in `.env` file at project root as `GOOGLE_API_KEY`
- **Python 3.x:** Installed within the `agent` environment
- **Required Python packages:** google-genai, python-dotenv, pandas, tqdm (installed in `agent` environment)

**Input:**
- PDF folder containing candidate resumes (supports nested folders)
- Default path: `/data/home/xiong/data/Fund/Resumes/current/candidates`

## Troubleshooting

**If process_resume_skill fails:**
- **Check environment:** Ensure `agent` conda environment is activated (`conda activate agent`)
- **Verify Google API key:** Set in `.env` file: `GOOGLE_API_KEY=...`
- **Check PDF folder:** Verify the path exists and contains PDF files
- **Ensure dependencies:** Install `google-genai` in `agent` environment if missing: `pip install google-genai`
- **Use `--skip_test` flag:** Bypass Gemini API connection test if needed
- **Check logs:** Review specific PDF files that failed to process

**If generate_candidate_summary_skill fails:**
- Verify the CSV file from Step 1 exists at the expected location
- Ensure CSV has required columns: Gender, URR, Country of Nationality
- Check that the CSV is not empty
- Verify pandas is installed: `pip install pandas`

**Common Issues:**
- Missing API key: Add `GOOGLE_API_KEY` to `.env` file
- Empty PDF folder: Verify PDFs exist in the specified folder
- Permission errors: Check read/write permissions for input/output directories
- Image-based PDFs: Gemini API handles these automatically with OCR, no additional setup needed