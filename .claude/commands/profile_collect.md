# Collect Information from Candidates' Resumes in PDF
You are a helpful agent. You will be provided with a folder of candidate's resumes. Follow the steps to extract the requested information. When ever need LLM for decusion, just use claude. 

The folder path you should work on is : /data/home/xiong/data/Fund/Resumes/current/candidates

## Task
1) Analysize the folder struction. There can be nested folder and make sure you get all pdf files in the folders. 

2) Use pypdf to extract pdf information and based on the info extract the candidate's **Name**, **Gender**, and **Country of Nationality**.
   - If not explicitly stated, infer **Gender** (target values: "Male" or "Female") and **Nationality** using the candidate’s name and locations of past education/work.

3) Using the **Nationality**, determine whether the candidate is from an **Under-Represented Region (URR)**.
   - URR must be **strictly** based on the list below. If Nationality matches any country in the list (case-insensitive), set `"URR": "yes"`, otherwise `"URR": "no"`.
   - If Nationality is "Unknown", set `"URR": "no"`.

** there are python scripts for procseeing pdfs and extractions info under /data/home/xiong/dev/Fund_Process_Automation/HR_Review_Bot, make sure you reuse those existing modules **

4) Double-check your results before replying:
   - Ensure **URR identification** is strictly based on the provided list.
   - Ensure **Gender** ∈ {"Male","Female","Unknown"}.
   - Ensure **Nationality** is a country name or "Unknown".
   - Trim whitespace and use title case for names/countries when appropriate.

## URR Country List (match exactly against these names; case-insensitive)
Afghanistan; Algeria; Angola; Bahrain; Benin; Botswana; Brunei Darussalam; Burkina Faso; Cabo Verde; Cambodia; Cameroon; Central African Republic; Chad; China; Comoros; Côte d'Ivoire; Democratic Republic of the Congo; Djibouti; Egypt; Equatorial Guinea; Eritrea; Ethiopia; Gabon; Ghana; Guinea; Guinea-Bissau; Hong Kong SAR; Indonesia; Iran; Iraq; Japan; Jordan; Kenya; Korea; Kuwait; Lao P.D.R.; Lebanon; Lesotho; Liberia; Libya; Macao SAR; Madagascar; Malawi; Malaysia; Mali; Mauritania; Mauritius; Morocco; Mozambique; Myanmar; Namibia; Niger; Nigeria; Oman; Pakistan; Philippines; Qatar; Republic of Congo; Rwanda; São Tomé and Príncipe; Saudi Arabia; Senegal; Seychelles; Sierra Leone; Singapore; Somalia; South Africa; South Sudan; Sudan; Swaziland; Syria; Tanzania; Thailand; The Gambia; Togo; Tunisia; Uganda; Vietnam; West Bank & Gaza; Yemen; Zambia; Zimbabwe

## Output Format
Return a csv file namned "candidate_profile.csv" with **Name**, **Gender**, **Country of Nationality** and **URR identification**

## Post statistical analysis
Lastly, read the output csv file provide a markdown file named summary.md with a summary table, showing the total number of candidates, number of male and female ; and number of URR and non-URR

## Clean up 
If you generated additional inmediate files to complete, make sure to clean them up once finished the task