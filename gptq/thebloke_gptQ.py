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

    # Find all <a> tags that contain 'GPTQ' in their href
    model_links = soup.find_all('a', href=lambda href: href and 'GPTQ' in href)
    entries = []
    for model_link in tqdm(model_links):
        model_url = model_link['href']
        print(model_url)
        entries.append(model_url)
    with open("output_gptq_scraped_models.yaml", 'w') as f:
        yaml.dump({"entries":entries}, f)


def extract_model_cards(model_links, entries):
    for model_link in tqdm(model_links):
        prefix = 'https://huggingface.co'

        model_url = prefix+model_link+'/tree/main'
        print(f"\nScrapping {model_url}")

        response = requests.get(model_url)
        model_html_content = response.text
        model_soup = BeautifulSoup(model_html_content, 'html.parser')

        # Find all <a> tags with '.bin' in their href within the model repository
        bin_links = model_soup.find_all('a', href=lambda href: href and href.endswith('.safetensors'))

        model_url = prefix+model_link
        print(f"\nScrapping {model_url}")

        response = requests.get(model_url)
        model_html_content = response.text
        model_soup = BeautifulSoup(model_html_content, 'html.parser')
        description = model_soup.find('div', class_='prose').find('h1').text.strip()

        for bin_link in bin_links:
            # Create a dictionary with the extracted information
            full_path = bin_link["href"]
            data = {
                'filename': full_path.replace("/tree/main","").replace("/blob/main",""),
                'description': description + " ("+model_link.split('/')[-1]+")",
                'license': "",
                'server': "",
                'SHA256': "",
                'owner_link': "",
                'owner': "TheBloke",
                'model_type':'api',
                'icon': 'https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/6426d3f3a7723d62b53c259b/tvPikpAzKTKGN5wrpadOJ.jpeg?w=200&h=200&f=face'
            }

            entries.append(data)  # Add the entry to the list

def html_to_yaml(url, output_file):
    get_model_entries(url, output_file)

def build_models(start_id, end_id, output_file):
    # Save the list of entries as YAML to the output file
    with open("output_gptq_scraped_models.yaml", 'r', encoding="utf8") as f:
        model_links = yaml.safe_load(f)

    entries = []  # List to store the entries
    extract_model_cards(model_links["entries"][start_id: end_id], entries)    
    # Save the list of entries as YAML to the output file
    with open(output_file, 'w') as f:
        yaml.dump(entries, f)

    print(f"YAML data saved to {output_file}")

# Example usage
# url = 'https://huggingface.co/TheBloke'
# html_to_yaml(url, 'output_gptq_scraped_models.yaml')

start=0
end=50
build_models(start,end,f"output_gptq_{start}_{end}.yaml")