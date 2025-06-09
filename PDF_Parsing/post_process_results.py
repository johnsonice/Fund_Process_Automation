#%%
import os
import shutil
import glob
from pathlib import Path

def reorganize_files(input_base_dir,output_base_dir):
    """
    Reorganize PDF parsing results from structure organized by filename to
    structure organized by file type (json, markdown, images) and then by filename.
    
    Current structure:
    - base_dir/
      - file_name_1/
        - file_name_1.json
        - file_name_1.md
        - images/
          - image1.png
          - image2.png
      - file_name_2/
        ...
    
    New structure:
    - base_dir/
      - json/
        - file_name_1/
          - file_name_1.json
      - markdown/
        - file_name_1/
          - file_name_1.md
      - images/
        - file_name_1/
          - image1.png
    """
    # Create new directories if they don't exist
    json_dir = os.path.join(output_base_dir, "json")
    markdown_dir = os.path.join(output_base_dir, "markdown")
    images_dir = os.path.join(output_base_dir, "images")
    
    for directory in [json_dir, markdown_dir, images_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Get all file name directories
    file_name_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d)) 
                     and d not in ["json", "markdown", "images"]]
    
    for file_name in file_name_dirs:
        file_dir = os.path.join(input_base_dir, file_name)
        
        # Process JSON files
        json_files = glob.glob(os.path.join(file_dir, "*content_list.json"))
        for json_file in json_files:
            shutil.copy2(json_file, os.path.join(json_dir, os.path.basename(json_file).replace('_content_list', '')))
        
        # Process Markdown files
        md_files = glob.glob(os.path.join(file_dir, "*.md"))
        for md_file in md_files:
            shutil.copy2(md_file, os.path.join(markdown_dir, os.path.basename(md_file)))
        
        # Process images folder
        img_folder = os.path.join(file_dir, "images")
        if os.path.exists(img_folder) and os.path.isdir(img_folder):
            file_img_dir = os.path.join(images_dir, file_name)
            if not os.path.exists(file_img_dir):
                os.makedirs(file_img_dir)
            
            for img in os.listdir(img_folder):
                img_path = os.path.join(img_folder, img)
                if os.path.isfile(img_path):
                    shutil.copy2(img_path, os.path.join(file_img_dir, img))

def clean_paths_in_files(directory, base_path_to_remove="/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/", 
                        file_patterns=["**/*.md", "**/*.json"]):
    """
    Process files in a directory and replace paths within their content.
    
    Args:
        directory (str or Path): The directory containing files to process
        base_path_to_remove (str): The base path to remove from file contents
        file_patterns (list): List of glob patterns for files to process
        
    Returns:
        int: Number of files processed
    """
    # Ensure directory is a Path object
    directory = Path(directory)
    
    # Find all matching files
    all_files = []
    for pattern in file_patterns:
        all_files.extend(list(directory.glob(pattern)))
    
    files_processed = 0
    
    # Process each file
    for file_path in all_files:
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the path in the content
            new_content = content.replace(base_path_to_remove, "")
            
            # Write the modified content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            files_processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return files_processed

#%%

if __name__ == "__main__":
    ## define path 
    input_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/All_AIV_2008-2025_json_md')
    output_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output_organized/All_AIV_2008-2025')
    # ## define path 
    # input_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/All_AIV_before_2008_json_md')
    # output_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output_organized/All_AIV_before_2008')
    # ## define path 
    # input_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output/Program_json_md')
    # output_base_dir = Path('/ephemeral/home/xiong/data/Fund/pdf_parse/Fund_Document/output_organized/Program')
    
    assert input_base_dir.exists()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reorganizing files in {input_base_dir}...")
    reorganize_files(input_base_dir,output_base_dir)

    print('Clearing paths in files...')
    files_processed = clean_paths_in_files(output_base_dir)
    files_processed = clean_paths_in_files(output_base_dir, base_path_to_remove="_json_md")
    
    print(f"Processed {files_processed} files to clean paths.")
    
# %%
