import requests
from bs4 import BeautifulSoup
import json

# URL of the page to scrape
url = 'https://openrouter.ai/docs#quick-start'

# Send a GET request to the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Assuming the model names and IDs are in a table with class 'model-table'
    # You will need to inspect the HTML and adjust the class name accordingly
    models_table = soup.find('table', {'class': 'table-fixed w-full'})
    
    # Assuming each model is in a row with class 'model-row'
    # and the model name and ID are in separate cells
    # You will need to inspect the HTML and adjust the class names accordingly
    model_rows = models_table.find_all('tr', {'class': 'model-row'})
    
    models_data = []
    for row in model_rows:
        model_name = row.find('td', {'class': 'model-name'}).text.strip()
        model_id = row.find('td', {'class': 'model-id'}).text.strip()
        
        model_info = {
            "category": "generic",
            "datasets": "unknown",
            "icon": "",  # Placeholder for icon URL
            "last_commit_time": "",  # Placeholder for last commit time
            "license": "commercial",
            "model_creator": "",  # Placeholder for model creator
            "model_creator_link": "",  # Placeholder for model creator link
            "name": model_id,
            "quantizer": None,
            "rank": 0.0,  # Placeholder for rank
            "type": "api",
            "variants": [
                {
                    "name": model_name,
                    "size": "Not so much"  # Placeholder for size
                }
            ]
        }
        models_data.append(model_info)
    
    # Convert the list of dictionaries to JSON
    models_json = json.dumps(models_data, indent=2)
    
    # Output the JSON or write it to a file
    print(models_json)
    with open('models.json', 'w') as json_file:
        json_file.write(models_json)
else:
    print(f"Failed to retrieve the webpage, status code: {response.status_code}")
