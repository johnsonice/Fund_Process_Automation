#!/usr/bin/env python3
"""
Generate summary.md from candidate_profile.csv
"""

import pandas as pd

def generate_summary():
    # Read the CSV file
    df = pd.read_csv('/data/home/xiong/dev/Fund_Process_Automation/candidate_profile.csv')
    
    # Calculate statistics
    total_candidates = len(df)
    
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
    nationality_counts = df['Country of Nationality'].value_counts()
    
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
        urr_status = "Yes" if df[df['Country of Nationality'] == country]['URR'].iloc[0] == 'yes' else "No"
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
    urr_countries = urr_candidates['Country of Nationality'].value_counts()
    
    for country, count in urr_countries.items():
        summary_content += f"\n- {country}: {count} candidate(s)"
    
    summary_content += f"""

---
*Analysis generated on: 2025-09-12*  
*Total PDFs processed: {total_candidates}*
"""
    
    # Write to file
    with open('/data/home/xiong/dev/Fund_Process_Automation/summary.md', 'w') as f:
        f.write(summary_content)
    
    print("Summary generated successfully!")
    print(f"Total candidates: {total_candidates}")
    print(f"Male: {male_count}, Female: {female_count}, Unknown: {unknown_gender_count}")
    print(f"URR: {urr_yes}, Non-URR: {urr_no}")

if __name__ == "__main__":
    generate_summary()