import requests
from bs4 import BeautifulSoup
import yaml
from pathlib import Path
from urllib.parse import urlparse

def get_website_path(url):
    parsed_url = urlparse(url)
    website_path = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return website_path

def html_to_yaml(url_list, output_file):
    entries = []  # List to store the entries

    for url in url_list:
        # Send a GET request to the URL and retrieve the HTML content
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
        # Create a dictionary with the extracted information
        data = {
            'filename': file_name,
            'description': description,
            'license': license,
            'server': server_link,
            'SHA256': SHA256,
            'owner_link': owner_link,
            'owner': "TheBloke"
        }

        entries.append(data)  # Add the entry to the list

    # Save the list of entries as YAML to the output file
    with open(output_file, 'w') as f:
        yaml.dump(entries, f)

    print(f"YAML data saved to {output_file}")

# MPT 40B
url_list = [
    'https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_0.bin',
    'https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_1.bin',
    'https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q5_0.bin',
    'https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q5_1.bin',
    'https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q8_0.bin',
]


html_to_yaml(url_list, 'output.yaml')
