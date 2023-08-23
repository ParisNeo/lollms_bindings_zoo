import requests
import yaml
from pathlib import Path
import urllib.request

def get_file_size(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)
        size_in_bytes = response.headers.get('content-length')
        if size_in_bytes:
            size_in_bytes = int(size_in_bytes)
            return size_in_bytes
        else:
            return None
    except Exception as e:
        print(f"An error occurred while retrieving file size: {e}")
        return None
    
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
        server_link = server_link if server_link.endswith("/") else server_link+"/"
        v = [{"name":file_name,"size":get_file_size(server_link+file_name)}]
        
        # Create a dictionary with the extracted information
        data = {
            'title':file_name,
            'filename': file_name,
            'variants': v,
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
output_file = 'c_transformers/output.yaml'

json_to_yaml(json_url, output_file)
