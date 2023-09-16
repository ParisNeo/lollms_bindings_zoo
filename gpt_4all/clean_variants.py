import yaml
from pathlib import Path
with open(Path(__file__).parent/"models.yaml","r") as f:
    data = yaml.safe_load(f)

def remove_bad_quantizations(list_of_dict):
    accepted = []
    for entry in list_of_dict:
        ok=False
        selected=""
        if "ggml" in entry["name"].lower():
            for v in entry["variants"]:
                if "q4_0" in v["name"].lower():
                    ok = True
                    selected = v
                    break
            if ok:
                entry["variants"] = [selected]
                accepted.append(entry)
    # Return a new list containing only the unique items
    return accepted

data2 = remove_bad_quantizations(data)
print(f"initial number of entries : {len(data)}")
print(f"new number of entries : {len(data2)}")


with open(Path(__file__).parent/"models.yaml","w") as f:
    yaml.safe_dump(data2, f)


print("Done")