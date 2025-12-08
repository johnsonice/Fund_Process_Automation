# Fund Process Automation

## Project Overview

This project automates fund management processes including resume analysis, candidate profiling, document parsing, and technical assessment evaluation. It uses AI/LLM capabilities to extract information from resumes, generate candidate summaries, and evaluate coding assignments.

## Architecture

The project is organized into three main components:

### 1. HR_Review_Bot
Handles resume analysis and candidate information extraction.

**Key Features:**
- Processes PDF resumes using Google Gemini API
- Extracts: Name, Gender, Nationality, and URR (Under-Represented Region) status
- Handles both text-based and scanned (image-based) PDFs
- Generates CSV output with candidate profiles
- Provides statistics on gender distribution and URR representation

**Main Scripts:**
- [process_resume.py](.claude/skills/process_resume_skill/scripts/process_resume.py) - Core resume processing logic

### 2. PDF_Parsing
Converts PDF documents to text using MinerU.

**Key Features:**
- Batch PDF to text conversion
- Post-processing of conversion results
- High-quality extraction for both text and scanned documents

**Main Scripts:**
- [MinerU_convert_pdfs.py](PDF_Parsing/MinerU_convert_pdfs.py) - PDF conversion utility
- [post_process_results.py](PDF_Parsing/post_process_results.py) - Post-processing pipeline

### 3. Technical_Test_Evaluation
Evaluates Research Analyst candidate coding assignments.

**Purpose:**
- Grades Part 3 and Part 4 of technical take-home assignments
- Assesses code quality, correctness, structure, and documentation
- Generates Excel output matching grading template format

**Configuration:**
- Instruction DOCX: Source of requirements
- Grading template: Schema and evaluation criteria
- Candidate submissions: Individual folders per candidate

### 4. libs/
Shared utilities used across components.

**Modules:**
- [llm_utils.py](libs/llm_utils.py) - OpenAI/LLM integration, token counting, cost tracking
- [utils.py](libs/utils.py) - Common utilities, logging, error handling

## Claude Code Skills

This project includes two custom Claude Code skills for automation:

### process_resume_skill
Processes candidate resume PDFs to extract key information.

**Usage:**
```bash
python .claude/skills/process_resume_skill/scripts/process_resume.py \
  --pdf_folder /path/to/resumes \
  --output_file candidates_info.csv \
  --model gemini-2.0-flash-exp
```

**Output:** CSV with columns: Name, Gender, Nationality, URR

See [SKILL.md](.claude/skills/process_resume_skill/SKILL.md) for details.

### generate_candidate_summary_skill
Generates markdown summary reports from candidate profile data.

**Usage:**
```bash
python .claude/skills/generate_candidate_summary_skill/generate_summary.py \
  --csv_file candidate_profile.csv \
  --output_file summary.md
```

**Output:** Markdown report with statistics, insights, and URR country breakdown

See [SKILL.md](.claude/skills/generate_candidate_summary_skill/SKILL.md) for details.

## Custom Commands

### /eval_test
Evaluates RA coding performance for Parts 3 & 4 of technical assignments.

**Inputs:**
- Instruction DOCX with requirements
- Grading template (SPRAI_Grades.xlsx)
- Candidate submission folders

**Output:**
- Excel file (SPRAI_Grades_EVAL.xlsx) with evaluation results
- One row per candidate with scores and justification notes

**Evaluation Criteria:**
- Correctness with respect to task
- Code quality and readability
- Structure and modularity
- Documentation and usage clarity
- Data handling and edge cases

See [eval_test.md](.claude/commands/eval_test.md) for full specification.

### /profile_collect
Command for collecting and processing candidate profiles.

See [profile_collect.md](.claude/commands/profile_collect.md) for details.

## Environment Setup

### Prerequisites
- Python 3.x
- Conda environment named `traction` (or `agent` for skills)
- API keys for OpenAI and Google Gemini

### Configuration

1. Create `.env` file in project root:
```bash
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

2. Activate conda environment:
```bash
conda activate traction
```

### Dependencies

**Core Python Libraries:**
- openai - OpenAI API client
- google-generativeai - Google Gemini API
- pandas - Data manipulation
- tiktoken - Token counting
- python-dotenv - Environment variable management
- huggingface_hub - Model downloads

**Optional:**
- openpyxl/xlrd - Excel file handling
- MinerU - PDF conversion engine

## Common Workflows

### 1. Process Resume Batch
```bash
# Activate environment
conda activate agent

# Process resumes
python .claude/skills/process_resume_skill/scripts/process_resume.py \
  --pdf_folder /path/to/resumes \
  --output_file candidate_profile.csv

# Generate summary
python .claude/skills/generate_candidate_summary_skill/generate_summary.py \
  --csv_file candidate_profile.csv \
  --output_file summary.md
```

### 2. Convert PDFs
```bash
python PDF_Parsing/MinerU_convert_pdfs.py --input_dir /path/to/pdfs
```

### 3. Evaluate Technical Tests
Use the Claude Code command:
```bash
/eval_test
```

## URR Countries

The system identifies candidates from Under-Represented Regions based on a predefined list including:
- Afghanistan, Algeria, Angola, Bahrain, Benin, Botswana, Brunei Darussalam
- Burkina Faso, Cabo Verde, Cambodia, Cameroon, Central African Republic
- Chad, China, Comoros, Côte d'Ivoire, Democratic Republic of the Congo
- Djibouti, Egypt, and many others

See [process_resume.py:33-40](.claude/skills/process_resume_skill/scripts/process_resume.py#L33-L40) for the complete list.

## Cost Tracking

The project includes built-in cost tracking for LLM API calls:

**Pricing (per 1M tokens):**
- GPT-4o: $5 (prompt), $15 (completion)
- GPT-4o-mini: $0.15 (prompt), $0.60 (completion)
- GPT-4: $5 (prompt), $15 (completion)
- GPT-3.5-turbo: $0.50 (prompt), $1.50 (completion)

Cost tracking is automatic in the `BSAgent` class ([llm_utils.py:40-57](libs/llm_utils.py#L40-L57)).

## Key Classes and Functions

### BSAgent (llm_utils.py)
Base OpenAI agent class for LLM interactions.

**Methods:**
- `get_completion()` - Get LLM completion with cost tracking
- `get_response_content()` - Get response text only
- `extract_json_string()` - Extract JSON from markdown code blocks
- `parse_load_json_str()` - Parse JSON responses

**Usage:**
```python
from libs.llm_utils import BSAgent

agent = BSAgent(model="gpt-4o", temperature=0)
response = agent.get_completion(
    prompt_template={"system": "...", "user": "..."},
    return_cost=True
)
```

### Resume Processing Functions

**process_single_resume()** - Process one resume with Gemini
**process_resumes()** - Batch process all resumes in folder
**print_statistics()** - Display summary statistics
**get_extraction_prompt()** - Get the information extraction prompt

## Data Schema

### Candidate Profile CSV
| Column | Type | Description |
|--------|------|-------------|
| Name | string | Candidate's full name |
| Gender | string | Male/Female (inferred) |
| Nationality | string | Country of nationality |
| URR | string | "yes" or "no" |

### Grading Excel Template
Follows the structure in SPRAI_Grades.xlsx with categories for:
- Code correctness
- Code quality
- Structure/modularity
- Documentation
- Overall assessment

## Error Handling

All scripts include comprehensive error handling:
- Graceful failures with informative messages
- Retry logic for API calls (3 attempts with 5s delay)
- Validation of input files and required columns
- Missing submission handling with clear notes

## Development Notes

### Git Status
Current branch: `main`

**Modified files:**
- .claude/commands/profile_collect.md
- .claude/skills/process_resume_skill/SKILL.md
- .claude/skills/process_resume_skill/scripts/process_resume.py

**Untracked:**
- .claude/skills/generate_candidate_summary_skill/

### Best Practices

1. **API Keys:** Never commit .env file - it's gitignored
2. **PDF Processing:** Gemini handles both text and scanned PDFs natively
3. **Error Logging:** All scripts use centralized logging from libs/utils.py
4. **Cost Awareness:** Monitor API costs, especially with large batches
5. **Code Evaluation:** Static analysis only - do not execute candidate code

## Troubleshooting

**Issue: API key not found**
- Ensure `.env` file exists in project root
- Check that `OPENAI_API_KEY` or `GOOGLE_API_KEY` is set

**Issue: PDF processing fails**
- Verify PDF files are readable (not corrupted)
- Check Gemini API quota and rate limits
- Use `--skip_test` flag to skip connection test

**Issue: Missing candidate data**
- Check PDF folder path is correct
- Ensure PDFs contain extractable text or images
- Review error logs for specific failures

**Issue: Excel template mismatch**
- Verify SPRAI_Grades.xlsx exists and has expected structure
- Check column names match exactly
- Ensure scoring scales are consistent

## Future Enhancements

Potential areas for improvement:
- Add support for more LLM providers (Anthropic Claude, etc.)
- Implement parallel processing for large resume batches
- Add web UI for easier interaction
- Expand technical evaluation to more question types
- Add automated testing suite

## Contact & Support

For issues or questions:
- Check error logs in console output
- Review skill documentation in `.claude/skills/*/SKILL.md`
- Verify environment setup and API keys

## License

This is an internal automation tool for fund management processes.
