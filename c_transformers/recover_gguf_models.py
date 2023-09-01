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
import traceback
import urllib.request

def load_current_models_list():
    pth = Path(__file__).parent/"models.yaml"
    with open(str(pth),'r') as f:
        data = yaml.safe_load(f)
    return data
def remove_string(lst, s):
    return [x for x in lst if s not in x]


def get_file_size(url):
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
    
    
def get_website_path(url):
    parsed_url = urlparse(url)
    website_path = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return website_path

def click_expand_button(url):
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

def get_model_entries(url, output_file):
    expanded_html_content = click_expand_button(url)

    prefix = get_website_path(url)

    # Parse the expanded HTML content using BeautifulSoup
    soup = BeautifulSoup(expanded_html_content, 'html.parser')

    # Find all <a> tags that contain 'GGUF' in their href
    model_links = soup.find_all('a', href=lambda href: href and 'GGUF' in href)
    entries = []
    for model_link in tqdm(model_links):
        model_url = prefix + model_link['href'] + "/tree/main"
        print(model_url)
        entries.append(model_url)
    with open("output_scraped_models_gguf.yaml", 'w') as f:
        yaml.dump({"entries":entries}, f)


def extract_model_cards(model_links, entries):
    paths=[]
    for model_link in tqdm(model_links):
        prefix = '/'.join(model_link.split('/')[0:3])

        model_url = model_link
        if "superhot" in model_url.lower():
            continue
        print(f"\nScrapping {model_url}")

        response = requests.get(model_url)
        model_html_content = response.text
        model_soup = BeautifulSoup(model_html_content, 'html.parser')

        # Find all <a> tags with '.gguf' in their href within the model repository
        gguf_links = model_soup.find_all('a', href=lambda href: href and href.endswith('.gguf'))

        for gguf_link in tqdm(gguf_links):
            path = gguf_link['href'].replace("resolve","blob")
            if path in paths:
                continue
            paths.append(path)
            # Send a GET request to the URL and retrieve the HTML content
            if not ("blob/main" in path or "tree/main" in path) or not ("q2" in path.lower() or "q4" in path.lower() or "q5" in path.lower()) :
                print(f"\nSkipping : {path}")
                continue

            try:
                url = prefix+path
                response = requests.get(url)
                html_content = response.text

                prefix = get_website_path(url)

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # Find the <a> tag with the text 'download' and extract its href
                download_link = soup.find('a', string='download')['href']
                SHA256 = soup.find('strong', string='SHA256:').parent.text.split("\t")[-1]
                try:
                    license = soup.find(lambda tag: tag.name and tag.get_text(strip=True) == 'License:').parent.text.split("\n")[-2]
                except:
                    license = "unknown"
                # Split the path to extract the file name
                file_name = Path(download_link).name

                # Split the server link and remove 'resolve/main/'
                server_link = prefix + str(Path(download_link).parent).replace("\\", "/")
                owner_link = "/".join(server_link.split("/")[:-2]) + "/"

                try:
                    response = requests.get(owner_link)
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    description = soup.find('div', class_='prose').find('h1').text.strip() + "("+url.split('.')[-2]+")"
                except:
                    description = f"{file_name} model"

                v  = []
                for gguf_link in gguf_links:
                    file_name_ = gguf_link["href"].split('/')[-1]
                    full_url = server_link+"/"+file_name_
                    file_size = get_file_size(full_url)[0]
                    v.append({"name":file_name_,"size":file_size})
                    
                full_url = server_link+"/"+file_name
                file_size = get_file_size(full_url)[0]
                # Create a dictionary with the extracted information
                data = {
                    'title': file_name,
                    'filename': file_name,
                    'file_size': file_size,
                    'description': description,
                    'variants': v,
                    'license': license,
                    'server': server_link,
                    'SHA256': SHA256,
                    'owner_link': owner_link,
                    'type': "GGUF",
                    'owner': "TheBloke",
                    'patreon': "https://www.patreon.com/TheBlokeAI",
                    'icon': 'https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/6426d3f3a7723d62b53c259b/tvPikpAzKTKGN5wrpadOJ.jpeg?w=200&h=200&f=face'
                }

                entries.append(data)  # Add the entry to the list
                break
            except Exception as ex :
                # Catch the exception and get the traceback as a list of strings
                traceback_lines = traceback.format_exception(type(ex), ex, ex.__traceback__)

                # Join the traceback lines into a single string
                traceback_text = ''.join(traceback_lines)

                print(f"\nCouldn't load {gguf_link['href']}.\nException: {ex}")
                print(traceback_text)

def html_to_yaml(url, output_file):
    get_model_entries(url, output_file)

def build_models(start_id, end_id, output_file):
    # Save the list of entries as YAML to the output file
    with open("output_scraped_models_gguf.yaml", 'r', encoding="utf8") as f:
        model_links = yaml.safe_load(f)

    models_list = load_current_models_list()
    print("Removing old models")
    for entry in models_list:
        model_links['entries']=remove_string(model_links['entries'], entry['filename'])
    print("Done")

    entries = []  # List to store the entries
    extract_model_cards(model_links["entries"][start_id: end_id], entries)    
    # Save the list of entries as YAML to the output file
    with open(output_file, 'w') as f:
        yaml.dump(entries, f)

    print(f"YAML data saved to {output_file}")

# Example usage
url = 'https://huggingface.co/TheBloke'
html_to_yaml(url, 'output_scraped_models_gguf.yaml')

build_models(0,-1,f"output_gguf.yaml")