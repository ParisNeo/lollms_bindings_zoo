import requests
import yaml
from pathlib import Path

def get_website_path(url):
    parsed_url = urlparse(url)
    website_path = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return website_path

def json_to_yaml(json_url, output_file):
    response = requests.get(json_url)
    json_data = response.json()

    entries = []  # List to store the entries

    for entry in json_data:
        # Extract the required information from the JSON entry
        file_name = str(Path(entry['filename']))
        url = entry.get('url')
        server_link = url[:url.rfind('/')] if url else f"https://gpt4all.io/models/"
        owner_link = "https://gpt4all.io/"
        description = entry['description']
        license = ""
        SHA256 = entry['md5sum']

        # Create a dictionary with the extracted information
        data = {
            'filename': file_name,
            'description': description,
            'license': license,
            'server': server_link,
            'SHA256': SHA256,
            'owner_link': owner_link,
            'owner': "nomic-ai",
            'icon': 'https://gpt4all.io/gpt4all-128.png'
        }

        entries.append(data)  # Add the entry to the list

    # Save the list of entries as YAML to the output file
    with open(output_file, 'w') as f:
        yaml.dump(entries, f)

    print(f"YAML data saved to {output_file}")


# Example usage:
json_url = 'https://gpt4all.io/models/models.json'
output_file = 'output.yaml'

json_to_yaml(json_url, output_file)
