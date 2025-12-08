#!/usr/bin/env python3
"""
Process candidate resumes to extract Name, Gender, Nationality and URR status
"""

import os
import sys
from pathlib import Path

# Add libs folder to path using absolute path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent.parent
libs_path = project_root / 'libs'
sys.path.insert(0, str(libs_path))

import argparse
import logging
import json
import re
import time
from typing import Dict

from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils import get_all_files
from tqdm import tqdm
import pandas as pd

# Load environment variables
load_dotenv(project_root / '.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URR country list
URR_COUNTRIES = """Afghanistan; Algeria; Angola; Bahrain; Benin; Botswana; Brunei Darussalam; Burkina Faso; Cabo Verde;
Cambodia; Cameroon; Central African Republic; Chad; China; Comoros; Côte d'Ivoire; Democratic Republic of the Congo;
Djibouti; Egypt; Equatorial Guinea; Eritrea; Ethiopia; Gabon; Ghana; Guinea; Guinea-Bissau; Hong Kong SAR;
Indonesia; Iran; Iraq; Japan; Jordan; Kenya; Korea; Kuwait; Lao P.D.R.; Lebanon; Lesotho; Liberia; Libya;
Macao SAR; Madagascar; Malawi; Malaysia; Mali; Mauritania; Mauritius; Morocco; Mozambique; Myanmar; Namibia; Niger;
Nigeria; Oman; Pakistan; Philippines; Qatar; Republic of Congo; Rwanda; São Tomé and Príncipe; Saudi Arabia; Senegal;
Seychelles; Sierra Leone; Singapore; Somalia; South Africa; South Sudan; Sudan; Swaziland; Syria; Tanzania;
Thailand; The Gambia; Togo; Tunisia; Uganda; Vietnam; West Bank & Gaza; Yemen; Zambia; Zimbabwe"""

def get_extraction_prompt() -> str:
    """Get the prompt for candidate information extraction."""
    return f"""You are a helpful assistant extracting information from a candidate's resume.

### 1 Based on the information provided, please extract candidate's Name, Gender and Country of Nationality.
Gender and Nationality information may not be extracted directly. Please infer based on candidate's name and locations of past experiences.

### 2 Based on the nationality, please determine if the candidate is from an under represented region.
Here is the list of countries in under represented regions (URR):
{URR_COUNTRIES}

### 3 Double check the extracted information and make sure it is correct.
Especially make sure the URR identification is strictly based on the list provided.

Provide your answer in the following JSON format:
{{{{
    "Name":"<name>",
    "Gender": "<gender>",
    "Nationality": "<country of nationality>",
    "URR": "<yes or no>"
}}}}"""

def parse_json_response(response_text: str) -> Dict[str, str]:
    """Parse JSON from response text, handling markdown code blocks."""
    try:
        # Try to parse directly first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"Name"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")

def process_single_resume(pdf_path: str, client: genai.Client, model: str, retry_count: int = 2) -> Dict[str, str]:
    """Process a single resume and extract candidate information using Gemini API."""
    for attempt in range(retry_count + 1):
        try:
            # Read PDF bytes
            pdf_bytes = Path(pdf_path).read_bytes()

            # Create prompt
            prompt = get_extraction_prompt()

            # Generate content with Gemini
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type='application/pdf',
                    ),
                    prompt
                ]
            )

            # Parse response
            response_text = response.text
            dict_res = parse_json_response(response_text)

            # Validate required fields
            required_fields = ["Name", "Gender", "Nationality", "URR"]
            for field in required_fields:
                if field not in dict_res:
                    raise ValueError(f"Missing required field: {field}")

            return dict_res

        except Exception as e:
            if attempt < retry_count:
                logger.warning(f"Attempt {attempt + 1} failed for {pdf_path}: {e}. Retrying...")
                time.sleep(2)  # Wait 2 seconds before retry
            else:
                logger.error(f"Failed to process {pdf_path} after {retry_count + 1} attempts: {e}")
                return {
                    "Name": Path(pdf_path).stem,
                    "Gender": "error",
                    "Nationality": "error",
                    "URR": "error"
                }

def process_resumes(pdf_folder: str, client: genai.Client, model: str, delay: float = 0.5) -> pd.DataFrame:
    """Process all resumes in the folder."""
    pdfs = get_all_files(pdf_folder, end_with='.pdf')

    if not pdfs:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return pd.DataFrame()

    logger.info(f"Found {len(pdfs)} PDF files to process")

    res_list = []
    for i, pdf_path in enumerate(tqdm(pdfs, desc="Processing resumes")):
        result = process_single_resume(pdf_path, client, model)
        res_list.append(result)

        # Add delay between requests to avoid rate limiting (except for last item)
        if i < len(pdfs) - 1:
            time.sleep(delay)

    return pd.DataFrame(res_list)

def print_statistics(df: pd.DataFrame) -> None:
    """Print statistics about processed candidates."""
    if df.empty:
        logger.warning("No data to display statistics")
        return

    total = len(df)
    num_female = (df['Gender'] == 'Female').sum()
    num_male = (df['Gender'] == 'Male').sum()
    num_urr = (df['URR'] == 'yes').sum()
    num_non_urr = (df['URR'] == 'no').sum()
    num_errors = (df['Gender'] == 'error').sum()

    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total candidates: {total}")
    print(f"Female candidates: {num_female} ({num_female/total*100:.1f}%)")
    print(f"Male candidates: {num_male} ({num_male/total*100:.1f}%)")
    print(f"URR candidates: {num_urr} ({num_urr/total*100:.1f}%)")
    print(f"Non-URR candidates: {num_non_urr} ({num_non_urr/total*100:.1f}%)")
    if num_errors > 0:
        print(f"Processing errors: {num_errors} ({num_errors/total*100:.1f}%)")
    print(f"{'='*50}\n")

def unit_test(client: genai.Client, model: str) -> bool:
    """Test Gemini API connection."""
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say 'Hello' if you can hear me."
        )
        logger.info(f"Unit test passed: {response.text}")
        return True
    except Exception as e:
        logger.error(f"Unit test failed: {e}")
        return False

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process candidate resumes to extract Name, Gender, Nationality and URR status.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdf_folder /path/to/resumes
  %(prog)s --pdf_folder /path/to/resumes --output_file results.csv --skip_test
        """
    )
    parser.add_argument(
        '--pdf_folder',
        type=str,
        default='/ephemeral/home/xiong/data/Fund/Resumes/current',
        help='Path to folder containing PDF resumes'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to output CSV file (default: candidates_info.csv in pdf_folder)'
    )
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip the unit test'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash-image',
        help='Gemini model to use (default: gemini-2.5-flash-image)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay in seconds between API requests (default: 0.5)'
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate pdf_folder exists
    if not os.path.exists(args.pdf_folder):
        logger.error(f"PDF folder does not exist: {args.pdf_folder}")
        sys.exit(1)

    # Initialize Gemini client
    logger.info(f"Initializing Gemini client with model: {args.model}")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Run unit test if not skipped
    if not args.skip_test:
        logger.info("Running unit test...")
        if not unit_test(client, args.model):
            logger.error("Unit test failed. Exiting.")
            sys.exit(1)

    # Process resumes
    logger.info(f"Processing resumes from: {args.pdf_folder}")
    df = process_resumes(args.pdf_folder, client, args.model, args.delay)

    if df.empty:
        logger.error("No resumes were processed successfully")
        sys.exit(1)

    # Determine output file path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.pdf_folder, 'candidates_info.csv')

    # Save results
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")

    # Print statistics
    print_statistics(df)

if __name__ == "__main__":
    main()
