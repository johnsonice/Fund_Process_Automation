#%%
import os, sys
sys.path.insert(0,'../libs')
import openai
from pypdf import PdfReader
from dotenv import load_dotenv
from llm_utils import BSAgent
from utils import get_all_files
from tqdm import tqdm
import pandas as pd
load_dotenv('../.env')
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#%%
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def unit_test():
    llm_agent  = BSAgent(model="gpt-4o-mini",  #gpt-40 gpt-4o-mini
                        temperature=0)
    ## just run one test, make sure the api works 
    pt = {'system':'You are a helpful assistant.',
        'user':'What is your name?'}
    res = llm_agent.get_response_content(prompt_template=pt)
    print(res) 
    
    return None

#%%
if __name__ == "__main__":

    # ### unit test for base agent
    unit_test()
    #%%
    pdf_folder = '/data/home/xiong/data/Fund/Resumes/MU'
    pdfs = get_all_files(pdf_folder,end_with='.pdf')
    #%%
    llm_agent  = BSAgent(model="gpt-4o-mini",  #gpt-40 gpt-4o-mini
                        temperature=0)
    res_list = []
    for p in tqdm(pdfs):
        resume_info = extract_text_from_pdf(p)
        pt = {'system':"""You are a helpful assistant.
                You will be provided with text information from a candidate's resume. Based the information provided, please extract candidate's Name, Gender and Country of Nationality. 
                Gender and Nationality information may not by extracted directly. Please infer based on candidate's name and locations of past experiences. 
                Provide your answer in the following JSON format:
                {{
                    "name":"<name>",
                    "gender": "<gender>", ## make sure it is either "Male" or "Female"
                    "nationality": "<country of nationality>"
                }}
                
                Please see candidate's resume info in user message below: 
            """,
            'user':resume_info}
        try:
            res = llm_agent.get_response_content(prompt_template=pt)
            dict_res = llm_agent.parse_load_json_str(res)
            res_list.append(dict_res)
        except:
            res_list.append({
                    "name":p,
                    "gender": "error",
                    "nationality": "error"
                })
    
    #%%
    res_df = pd.DataFrame(res_list)
    res_df.to_csv(os.path.join(pdf_folder,'candidates_info.csv'),index=False)
# %%
