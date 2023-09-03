import yaml
from pathlib import Path
with open(Path(__file__).parent/"models.yaml","r") as f:
    data = yaml.safe_load(f)

def remove_duplicates(list_of_dict):
    # Create a set from the list of dictionaries
    unique_items = {item["filename"]: item for item in list_of_dict} 
    
    # Return a new list containing only the unique items
    return [unique_item for filename, unique_item in unique_items.items()]

data2 = remove_duplicates(data)
print(f"initial number of entries : {len(data)}")
print(f"new number of entries : {len(data2)}")


with open(Path(__file__).parent/"models.yaml","w") as f:
    yaml.safe_dump(data2, f)


print("Done")