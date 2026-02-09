### Evaluate RA Coding Performance — Part 3 Only

Objective: Evaluate Research Analyst candidates' coding performance for Part 3 of the take‑home assignment, using the provided instructions and grading template. Ignore Part 1 (Excel), Part 2 (Stata), and Part 4 and another else.

###Inputs
- Instruction DOCX (source of requirements): `/data/home/xiong/data/Fund/Resumes/technical_test/current/IMF SPR Research Analyst Take-Home Assignments.docx`
- Grading template (schema, categories, examples): `/data/home/xiong/data/Fund/Resumes/technical_test/current/SPR_Grades_Template.xlsx`
- Candidate submissions root (each candidate has its own folder): `/data/home/xiong/data/Fund/Resumes/technical_test/current/raw_tests`

###Output
- A single Excel file that replicates the exact sheet structure, column names, and ordering of `SPR_Grades_Template.xlsx`, populated with one row per candidate assessed for Part 3.
- Do not overwrite the template. Save the new file to: `/data/home/xiong/data/Fund/Resumes/technical_test/current/SPR_Grades_EVAL.xlsx`

###Scope & Rules
- Only assess Part 3 requirements as defined in the DOCX. Do not grade Excel, Stata, or Part 4 components.
- Use the same categories, scales, and notes style/examples as the template. If weights exist in the template, respect them; otherwise, provide raw scores per category as in past examples.
- Provide concise, specific notes justifying each category score (mirroring tone/format of past examples in the template).
- If a candidate is missing Part 3, mark clearly as Missing/Not Submitted in the appropriate fields and explain in notes.

###Procedure
1) Read the DOCX and extract only the detailed requirements for Part 3. Treat these as the authoritative rubric for what to check.
2) Open `SPR_Grades_Template.xlsx` and infer:
   - Sheet name(s) to use
   - Exact column names/order
   - Scoring scales/weights (if present)
   - Examples of notes language and level of detail
3) Enumerate candidates: each top‑level folder under `raw_tests/` is one candidate. The folder name is the candidate identifier unless the template specifies a different identifier field.
4) For each candidate:
   - Locate the Part 3 Jupyter notebook. Part 3 submissions are typically `.ipynb` files with names containing "assignment3", "task3", "part3", "q3", or similar Part 3 indicators (e.g., `xxx_assignment3.ipynb`, `task3.ipynb`, `Part_3.ipynb`). Search the candidate's folder recursively for `.ipynb` files and identify the one corresponding to Part 3. If multiple notebooks exist, pick the one whose name most clearly references Part 3. If no `.ipynb` file is found, check for `.py` files with similar naming patterns as a fallback. If still unclear, record as ambiguous and explain in notes.
   - Evaluate against the Part 3 requirements with emphasis on: correctness with respect to the task, code quality/readability, structure/modularity, efficiency/complexity (where applicable), documentation/usage clarity, testing/reproducibility, data handling/edge cases. Map these assessments onto the template's actual categories.
   - Do not try to run candidates' code. only read the code and evaluate the quality.
   - Assign scores strictly following the template's scales. For each scored category, add a short justification note. If the template includes an overall notes/summary field, add a brief synthesized rationale.
5) Write the results for each candidate as a new row in the output Excel, matching the template's sheet and schema exactly. Do not change formatting or add columns. Ensure column types (numeric vs text) are consistent with the template.
6) Validate the output:
   - The output file exists at the specified path.
   - The header row(s) and sheet name(s) match the template exactly.
   - All candidate rows are populated only for Part 3 categories; fields unrelated to Part 3 remain empty as appropriate per template conventions.

###Edge Cases
- No Part 3 submission: leave scores blank or zero per template convention and add a clear "Not submitted" note.
- Non‑runnable or environment‑specific projects: base grading on static review; explain limitations in notes.
- Extra/irrelevant files: ignore except where they clarify candidate intent.

###Deliverable
- Save the final graded workbook to `/data/home/xiong/data/Fund/Resumes/technical_test/current/SPR_Grades_EVAL.xlsx`.
- Return a brief text summary listing candidate identifiers processed and any with missing/ambiguous Part 3.
