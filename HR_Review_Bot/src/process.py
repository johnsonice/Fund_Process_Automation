#%%
import os, sys
sys.path.insert(0,'../../libs')
import openai
from pypdf import PdfReader
from dotenv import load_dotenv
from llm_utils import BSAgent
from utils import get_all_files
from tqdm import tqdm
import pandas as pd
load_dotenv('../../.env')
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
    pdf_folder = '/ephemeral/home/xiong/data/Fund/Resumes/current'
    pdfs = get_all_files(pdf_folder,end_with='.pdf')
    #%%
    llm_agent  = BSAgent(model="gpt-4o-mini",  #gpt-40 gpt-4o-mini
                        temperature=0)
    res_list = []
    for p in tqdm(pdfs):
        resume_info = extract_text_from_pdf(p)
        pt = {'system':"""You are a helpful assistant.
                You will be provided with text information from a candidate's resume. Please follow the instructions to extract the information requested.
                
                ### 1 Based the information provided, please extract candidate's Name, Gender and Country of Nationality. Gender and Nationality information may not by extracted directly. 
                Please infer based on candidate's name and locations of past experiences. 
                
                ### 2 Based on the nationality, please determine if the candidate is from an under represented region.Here is the list of countries in under represented regions (URR):
                    Afghanistan; Algeria; Angola; Bahrain; Benin; Botswana; Brunei Darussalam; Burkina Faso; Cabo Verde; 
                    Cambodia; Cameroon; Central African Republic; Chad; China; Comoros; Côte d'Ivoire; Democratic Republic of the Congo; 
                    Djibouti; Egypt; Equatorial Guinea; Eritrea; Ethiopia; Gabon; Ghana; Guinea; Guinea-Bissau; Hong Kong SAR; 
                    Indonesia; Iran; Iraq; Japan; Jordan; Kenya; Korea; Kuwait; Lao P.D.R.; Lebanon; Lesotho; Liberia; Libya; 
                    Macao SAR; Madagascar; Malawi; Malaysia; Mali; Mauritania; Mauritius; Morocco; Mozambique; Myanmar; Namibia; Niger; 
                    Nigeria; Oman; Pakistan; Philippines; Qatar; Republic of Congo; Rwanda; São Tomé and Príncipe; Saudi Arabia; Senegal; 
                    Seychelles; Sierra Leone; Singapore; Somalia; South Africa; South Sudan; South Sudan; Sudan; Swaziland; Syria; Tanzania; 
                    Thailand; The Gambia; Togo; Tunisia; Uganda; Vietnam; West Bank & Gaza; Yemen; Yemen; Zambia; Zimbabwe

                ### 3 double check the extracted information and make sure it is correct.Especially make sure the URR identification is strickly baed on the list provided.
                
                
                Provide your answer in the following JSON format:
                {{
                    "Name":"<name>",
                    "Gender": "<gender>", ## make sure it is either "Male" or "Female"
                    "Nationality": "<country of nationality>" ## make sure it is a country name
                    "URR": "<yes or no>"   ## whether the candidate is in the list of under represented regions based on the list 
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
                    "Name":p,
                    "Gender": "error",
                    "Nationality": "error",
                    "URR": "error"
                })
    
    #%%
    res_df = pd.DataFrame(res_list)
    res_df.to_csv(os.path.join(pdf_folder,'candidates_info.csv'),index=False)
    
    # Calculate statistics
    num_female = res_df[res_df['Gender'] == 'Female'].shape[0]
    num_male = res_df[res_df['Gender'] == 'Male'].shape[0]
    num_urr = res_df[res_df['URR'] == 'yes'].shape[0]
    num_non_urr = res_df[res_df['URR'] == 'no'].shape[0]
    # Print the results
    print(f"Number of Female candidates: {num_female}")
    print(f"Number of Male candidates: {num_male}")
    print(f"Number of URR candidates: {num_urr}")
    print(f"Number of Non-URR candidates: {num_non_urr}")
# %%
