# project: lollms
# author: ParisNeo (built on lollms playground coding helper)
# script: scrapper.py
# descrkiption : scrapes the hugging face page of a model provider, recovers all models, then returns their model card, license, description, variants etc

import requests
from bs4 import BeautifulSoup
import yaml
from pathlib import Path
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import urllib.request
import argparse
import json
import time
from huggingface_hub import HfApi
api = HfApi(token = True)

DEFAULT_quantizer="TheBloke"
DEFAULT_MODEL_TYPE="ggml"

def hub_get_last_commit(repo_id):
    """
    Function to get the last commit of a hugging face repository.

    Parameters:
        repo_id: ID of the repository

    Returns:
        Commits: List of commits
    """    
    retry = True
    try_count = 0
    max_tries = 10
    while retry and try_count < max_tries:
        try:
            commits = api.list_repo_commits(repo_id = repo_id)
            retry = False
        except Exception as e:
            print(f"Error getting commits for: {repo_id} - {e}")
            try_count += 1
            time.sleep(1)
    return commits[-1]

def hugging_face_user(user):
    """
    Function to generate the path for a Hugging Face user.
    
    Parameters:
        user: The name of the user.

    Returns:
        A string representing the path to the Hugging Face user.
    """
    return f"https://huggingface.co/{user}"


def get_website_path(url):
    """
    Function to extract website URL from given input string (either http or https) and return it
    in an appropriate format for Selenium driver's use.
    parameters:
       - `url`: String containing either 'http://', 'https://', or empty ('')
                representing the base URL that needs to be extracted into a valid form
   returns:
       A formatted version of the provided URL suitable for usage with Selenium drivers"""
    parsed_url = urlparse(url)
    website_path = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return website_path

def click_expand_button(url):
    """Click on the Expand button within the Hugging Face webpage at the specified URL. 
    This will cause the site contents to expand so they can be fully rendered.
    -Parameters:
       - `url` - The URL pointing to the desired webpage where you want to activate the expansion feature.
    
    Returns:
        None
    """

    # Create a Selenium WebDriver instance
    driver = webdriver.Chrome()  # Adjust the driver according to your browser choice

    # Load the page
    driver.get(url)

    # Find the "Expand" button element by XPath and click it
    expand_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Expand")]'))
    )
    expand_button.click()

    # Wait for the page to load after the button click
    WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')


    # Get the HTML content of the expanded page
    expanded_html_content = driver.page_source

    # Close the browser
    driver.quit()

    return expanded_html_content

def get_model_entries(url, model_type="gguf", output_file="outputs_list.yaml"):
    """Extracts links to GGUF-containing pages from the HuggingFace website at the specified URL.

    Parameters:
    - `url`: String containing either 'http://', 'https://', or empty ('')
             representing the base URL that points to the target webpage.
    - `model_type`: String specifying what type of models should be extracted. Default is 'gguf'.
    - `output_file`: File object to write the list of found URLs to. If not supplied, defaults to 'models'.

    Returns:
        List of strings containing the full paths to each discovered link."""
    expanded_html_content = click_expand_button(url)

    prefix = get_website_path(url)

    # Parse the expanded HTML content using BeautifulSoup
    soup = BeautifulSoup(expanded_html_content, 'html.parser')

    # Find all <a> tags that contain 'GGUF' in their href
    model_links = soup.find_all('a', href=lambda href: href and model_type in href.lower())
    entries = []
    for model_link in tqdm(model_links):
        model_id = model_link['href'][1:]
        print(model_id)
        entries.append(model_id)
    with open(output_file, 'w') as f:
        yaml.dump({"entries":entries}, f)
    return entries

def extract_delimited_section(content):
    """
    Function to extract the delimited section from a text.

    Parameters:
        content: The content of the text containing the --- delimiters.

    Returns:
        The delimited section as a string.
    """   
    start_index = content.index('---')
    end_index = start_index+3+content[start_index+3:].index('---')
    yaml_str = ''.join(content[start_index + 4:end_index])
    return yaml.safe_load(yaml_str)

def get_file_size(url):
    """
    Function to get the size of a file from a given URL.
    The function will try to retrieve the content-length header from the URL and convert it to bytes, kilobytes, and megabytes.

    Parameters:
        url: string
            The URL of the file whose size is required.

    Returns:
        tuple
            (size_in_bytes, size_in_kb, size_in_mb)

    Raises:
        Exception
            If there is an error while retrieving the file size.
    """
    try:
        response = urllib.request.urlopen(url)
        size_in_bytes = response.headers.get('content-length')
        if size_in_bytes:
            size_in_bytes = int(size_in_bytes)
            size_in_kb = size_in_bytes / 1024
            size_in_mb = size_in_kb / 1024
            return size_in_bytes, size_in_kb, size_in_mb
        else:
            return None
    except Exception as e:
        print(f"An error occurred while retrieving file size: {e}")
        return None
    
def get_variants(model_id, model_type="gguf"):
    server_link = f"https://huggingface.co/{model_id}"
    model_url = f"{server_link}/tree/main"
    response = requests.get(model_url)
    model_html_content = response.text
    model_soup = BeautifulSoup(model_html_content, 'html.parser')

    # Find all <a> tags with '.gguf' in their href within the model repository
    links = model_soup.find_all('a', href=lambda href: href and (href.endswith(f'.{model_type}') or href.endswith(f'.bin')))
    v  = []
    for link in tqdm(links):
        file_name_ = link["href"].split('/')[-1]
        full_url = server_link+"/resolve/main/"+file_name_
        file_size = get_file_size(full_url)[0]
        v.append({"name":file_name_,"size":file_size})
    return v

def build_model_cards(entries, model_type='gguf', output_file="output_TheBloke_gguf.yaml"):
    """ Builds a yaml file of each model by scraping the model page and its content to extract the following parameters
    1- Model name: the name of the model that can be extracted from the entry itself
    2- Model creation date: uses the hugging face to track the date of the first commit
    3- Model description: Extracted from the model card. it is a section that follows a h2 tag that contains the code Description
    4- Model creator: Can be extracted from a README.md file in the repo from the metadata section. The entry is named model_creator
    5- license : The license of the model, it is also extracted from the README.ms file in the repo. The entry is named license 
    """
    cards = []
    for i,entry in enumerate(tqdm(entries)):
        card={}
        card["name"]=entry.split("/")[1]
        card["quantizer"]=entry.split("/")[0]
        card["type"]=model_type
        card["rank"]=1e10
        card["category"]="generic"
        try:
            # recover readme.md, example https://huggingface.co/TheBloke/ORCA_LLaMA_70B_QLoRA-GGUF/raw/main/README.md
            response = requests.get(f"https://huggingface.co/{entry}/raw/main/README.md")
            # Verify that the file exists
            if 200 <= response.status_code < 300:

                # Find metadata section that starts with --- and ends with --- at the beginning of the file
                metadata = extract_delimited_section(response.text)
                # print(f"metadata:\n{metadata}")
                if "model_creator" in metadata:
                    card["model_creator"]=metadata["model_creator"]
                else:
                    card["model_creator"]=card["quantizer"]

                if "base_model" in metadata:
                    card["model_creator_link"]="/".join(metadata["base_model"].split("/")[:-1])
                else:
                    if "model_link" in metadata:
                        card["model_creator_link"]="/".join(metadata["model_link"].split("/")[:-1])
                    else:
                        card["model_creator_link"]=f"https://huggingface.co/{card['quantizer']}"

                if "license" in metadata:
                    card["license"]=metadata["license"]
                else:
                    card["license"]="unknown"

                if "datasets" in metadata:
                    card["datasets"]=metadata["datasets"]
                else:
                    card["datasets"]="unknown"

            
            commit_time = hub_get_last_commit(entry).created_at
            card["last_commit_time"]=commit_time
            # let's find the variances
            variants = get_variants(entry, model_type)
            card["variants"]=variants
            #Open model_maker card
            model_creator_url=f"https://huggingface.co/{card['model_creator'].replace(' ','')}"
            response = requests.get(model_creator_url)
            if 200 <= response.status_code < 300:
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
                image_tags = soup.find_all('img')

                prefix = 'https://aeiljuispo.cloudimg.io'

                card["icon"]=""
                for tag in image_tags:
                    src = tag.get('src')
                    if src.startswith(prefix):
                        card["icon"]=src
                        break


            
        except Exception as e:
            card["license"]="unknown"
            card["datasets"]="unknown"
            card["variants"]=[]
        
        cards.append(card)
        # Save last file
        with open(output_file, 'w') as f:
            yaml.dump(cards, f)        
    return cards

def filter_entries(entries):
    with open(Path(__file__).parent/"models.yaml","r") as f:
        models = yaml.safe_load(f)
   
    filteredEntries=[] # Initialize an empty array to store new entries after filtering out duplicates based on name field
        
    for e in entries:   # Iterate through each element (entry) in input data set
        if not any([e == f"{m['quantizer']}/{m['name']}" for m in models]):      # If none of the elements match, add it to output collection
            filteredEntries.append(e)                            # Add this item into our cleaned up dictionary
            
    return filteredEntries              # Return a copy of all items that were added during processing

# Main program that takes a user name and scrapes his hugging face page using argparse, with default name as TheBloke and default model type gguf
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Scrape the Hugging Face profile of a specific user.')
    parser.add_argument("-n","--name", default=DEFAULT_quantizer, help="The username of the user whose profile we wish to scrape.")
    parser.add_argument("-t","--type", default=DEFAULT_MODEL_TYPE, help="The username of the user whose profile we wish to scrape.")
    args= parser.parse_args()
    # First we find the user hugging face urlbased on the entered username
    user_profile_url = hugging_face_user(args.name)
    # Now parse through the html content looking for any mention of the term defined earlier using get_model_entries method
    entries = get_model_entries(user_profile_url, model_type=args.type, output_file=f"outputs_list_{args.name}.yaml")
    # Filter entries
    entries = filter_entries(entries)
    # Now we open each of them and build a model card
    build_model_cards(entries, args.type, f"output_{args.name}_{args.type}.yaml")
    