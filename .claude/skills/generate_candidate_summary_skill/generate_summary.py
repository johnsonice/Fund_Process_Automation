#!/usr/bin/env python3
"""
Generate summary.md from candidate_profile.csv
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_summary(csv_path: str, output_path: str):
    """
    Generate a markdown summary report from candidate profile CSV.

    Args:
        csv_path: Path to the input CSV file
        output_path: Path to the output markdown file
    """
    # Validate input file exists
    if not os.path.exists(csv_path):
        logger.error(f"Input CSV file not found: {csv_path}")
        sys.exit(1)

    # Read the CSV file
    logger.info(f"Reading CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = ['Gender', 'URR', 'Nationality']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        sys.exit(1)
    
    # Calculate statistics
    logger.info("Calculating statistics...")
    total_candidates = len(df)

    if total_candidates == 0:
        logger.error("CSV file is empty, no candidates to process")
        sys.exit(1)

    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    male_count = gender_counts.get('Male', 0)
    female_count = gender_counts.get('Female', 0)
    unknown_gender_count = gender_counts.get('Unknown', 0)

    # URR distribution
    urr_counts = df['URR'].value_counts()
    urr_yes = urr_counts.get('yes', 0)
    urr_no = urr_counts.get('no', 0)

    # Top nationalities
    nationality_counts = df['Nationality'].value_counts()

    # Get current date for report
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create markdown summary
    summary_content = f"""# Candidate Profile Summary

## Overview
This analysis covers {total_candidates} candidate resumes processed from the Fund recruitment database.

## Summary Statistics

### Total Candidates
- **Total candidates processed: {total_candidates}**

### Gender Distribution
| Gender | Count | Percentage |
|--------|-------|------------|
| Male | {male_count} | {male_count/total_candidates*100:.1f}% |
| Female | {female_count} | {female_count/total_candidates*100:.1f}% |
| Unknown | {unknown_gender_count} | {unknown_gender_count/total_candidates*100:.1f}% |

### Under-Represented Region (URR) Distribution
| URR Status | Count | Percentage |
|------------|-------|------------|
| URR (Yes) | {urr_yes} | {urr_yes/total_candidates*100:.1f}% |
| Non-URR (No) | {urr_no} | {urr_no/total_candidates*100:.1f}% |

### Top Nationalities Represented
| Country | Count | URR Status |
|---------|-------|------------|"""
    
    # Add top 10 nationalities
    for country, count in nationality_counts.head(10).items():
        # Determine URR status for this country
        urr_status = "Yes" if df[df['Nationality'] == country]['URR'].iloc[0] == 'yes' else "No"
        summary_content += f"\n| {country} | {count} | {urr_status} |"
    
    summary_content += f"""

## Key Insights

1. **Gender Balance**: {female_count} female candidates ({female_count/total_candidates*100:.1f}%) vs {male_count} male candidates ({male_count/total_candidates*100:.1f}%)
2. **URR Representation**: {urr_yes} candidates ({urr_yes/total_candidates*100:.1f}%) are from Under-Represented Regions
3. **Geographic Diversity**: Candidates represent {len(nationality_counts)} different countries/regions
4. **Most Common Nationality**: {nationality_counts.index[0]} with {nationality_counts.iloc[0]} candidates

## URR Countries Identified
The following URR countries are represented in our candidate pool:"""
    
    # List URR countries found
    urr_candidates = df[df['URR'] == 'yes']
    urr_countries = urr_candidates['Nationality'].value_counts()
    
    for country, count in urr_countries.items():
        summary_content += f"\n- {country}: {count} candidate(s)"
    
    summary_content += f"""

---
*Analysis generated on: {current_date}*
*Total PDFs processed: {total_candidates}*
"""

    # Write to file
    logger.info(f"Writing summary to: {output_path}")
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(summary_content)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")
        sys.exit(1)

    logger.info("Summary generated successfully!")
    print("\n" + "="*50)
    print("SUMMARY REPORT GENERATED")
    print("="*50)
    print(f"Output file: {output_path}")
    print(f"Total candidates: {total_candidates}")
    print(f"Male: {male_count}, Female: {female_count}, Unknown: {unknown_gender_count}")
    print(f"URR: {urr_yes}, Non-URR: {urr_no}")
    print("="*50 + "\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate markdown summary report from candidate profile CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --csv_file /path/to/candidate_profile.csv
  %(prog)s --csv_file input.csv --output_file report.md
        """
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default='/data/home/xiong/dev/Fund_Process_Automation/candidate_profile.csv',
        help='Path to input CSV file (default: candidate_profile.csv in project root)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='/data/home/xiong/dev/Fund_Process_Automation/summary.md',
        help='Path to output markdown file (default: summary.md in project root)'
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    generate_summary(args.csv_file, args.output_file)

if __name__ == "__main__":
    main()