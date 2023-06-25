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
        license = soup.find(lambda tag: tag.name and tag.get_text(strip=True) == 'License:').parent.text.split("\n")[-2]

        # Split the path to extract the file name
        file_name = Path(download_link).name

        # Split the server link and remove 'resolve/main/'
        server_link = prefix + str(Path(download_link).parent).replace("\\", "/")
        owner_link = "/".join(server_link.split("/")[:-2]) + "/"

        response = requests.get(owner_link)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        description = soup.find('div', class_='prose').find('h1').text.strip() + "("+url.split('.')[-2]+")"

        # Create a dictionary with the extracted information
        data = {
            'filename': file_name,
            'description': description,
            'license': license,
            'server': server_link,
            'SHA256': SHA256,
            'owner_link': owner_link
        }

        entries.append(data)  # Add the entry to the list

    # Save the list of entries as YAML to the output file
    with open(output_file, 'w') as f:
        yaml.dump(entries, f)

    print(f"YAML data saved to {output_file}")


# Example usage:
# wizard falcon 7B
# url_list = [
#     'https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML/blob/main/wizard-falcon-7b.ggmlv3.q4_0.bin',
#     'https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML/blob/main/wizard-falcon-7b.ggmlv3.q4_1.bin',
#     'https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML/blob/main/wizard-falcon-7b.ggmlv3.q5_0.bin',
#     'https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML/blob/main/wizard-falcon-7b.ggmlv3.q5_1.bin',
#     'https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML/blob/main/wizard-falcon-7b.ggmlv3.q8_0.bin'
# ]
# Orca 3B
# url_list = [
#     'https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q4_0.bin',
#     'https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q4_1.bin',
#     'https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q5_0.bin',
#     'https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q5_1.bin',
#     'https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q8_0.bin',
# ]
# Orca 7B
url_list = [
    'https://huggingface.co/TheBloke/orca_mini_7B-GGML/blob/main/orca-mini-7b.ggmlv3.q4_0.bin',
    'https://huggingface.co/TheBloke/orca_mini_7B-GGML/blob/main/orca-mini-7b.ggmlv3.q4_1.bin',
    'https://huggingface.co/TheBloke/orca_mini_7B-GGML/blob/main/orca-mini-7b.ggmlv3.q5_0.bin',
    'https://huggingface.co/TheBloke/orca_mini_7B-GGML/blob/main/orca-mini-7b.ggmlv3.q5_1.bin',
    'https://huggingface.co/TheBloke/orca_mini_7B-GGML/blob/main/orca-mini-7b.ggmlv3.q8_0.bin',
]
# Orca 13B
# url_list = [
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q2_K.bin',
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q4_0.bin',
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q4_1.bin',
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q5_0.bin',
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q5_1.bin',
#     'https://huggingface.co/TheBloke/orca_mini_13B-GGML/blob/main/orca-mini-13b.ggmlv3.q8_0.bin',
# ]

# url_list = [
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q2_K.bin',
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q4_0.bin',
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q4_1.bin',
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q5_0.bin',
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q5_1.bin',
#     'https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/blob/main/baichuan-llama-7b.ggmlv3.q8_0.bin'
# ]
# Falcon 40B oasst
# url_list = [
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q2_k.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q4_0.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q4_1.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q4_k.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q5_0.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q5_1.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q5_k.bin',
#     'https://huggingface.co/TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GGML/blob/main/h2ogpt-falcon-40b.ggmlv3.q8_0.bin',
# ]
# Falcon 40B oasst
url_list = [
    'https://huggingface.co/TheBloke/mpt-30B-instruct-GGML/blob/main/mpt-30b-instruct.ggmlv0.q4_0.bin',
    'https://huggingface.co/TheBloke/mpt-30B-instruct-GGML/blob/main/mpt-30b-instruct.ggmlv0.q4_1.bin',
    'https://huggingface.co/TheBloke/mpt-30B-instruct-GGML/blob/main/mpt-30b-instruct.ggmlv0.q5_0.bin',
    'https://huggingface.co/TheBloke/mpt-30B-instruct-GGML/blob/main/mpt-30b-instruct.ggmlv0.q4_1.bin',
    'https://huggingface.co/TheBloke/mpt-30B-instruct-GGML/blob/main/mpt-30b-instruct.ggmlv0.q8_0.bin',
]




html_to_yaml(url_list, 'output.yaml')
