from pathlib import Path
from datetime import datetime
import json, yaml

leaderboards = Path(__file__).parent.parent/"leaderboard"

# Define a function to extract the datetime from the file name
def extract_datetime(file_path):
    # Split the file name using '_' and '.' as separators
    file_name = file_path.name
    parts = file_name.split('_')
    # Extract the datetime part and remove '.json'
    datetime_str = parts[1].replace('.json', '')
    # Convert the datetime string to a datetime object
    return datetime.strptime(datetime_str, "%Y-%m-%dT%H-%M-%S.%f")

# This function recovers latest leaderboard classification results from the leaderboard folder
def getLatestLeaderBoard(path:Path):
   # Get all JSON Files starting with 'results'
   resultFiles= [f for f in path.iterdir() if f.suffix==".json"]
   
   # Sort by date, oldest first
   sortedResultFileLists=sorted(resultFiles, key=extract_datetime)
       
   return sortedResultFileLists[0] 

leaderboard_file = getLatestLeaderBoard(leaderboards)
with open(leaderboard_file,"r") as f:
    leaderboard_data = json.load(f)

with open(Path(__file__).parent/"models.yaml","r") as f:
    models = yaml.safe_load(f)

for model in models:
    # find model in leaderboard
    model["rank"]=1