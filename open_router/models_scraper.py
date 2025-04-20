import requests
import yaml
from typing import List, Dict
from pathlib import Path

class OpenRouterModelFetcher:
    API_URL = "https://openrouter.ai/api/v1/models"

    def fetch_models(self) -> List[Dict]:
        """Fetches the model data from the OpenRouter API"""
        try:
            response = requests.get(self.API_URL)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            return data.get('data', [])
        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

    def format_models(self, models_data: List[Dict]) -> List[Dict]:
        """Formats the raw API data into the desired structure"""
        formatted_models = []
        for model in models_data:
            formatted_model = {
                "category": "generic",
                "datasets": "unknown",
                "icon": "",
                "last_commit_time": "",
                "license": "commercial",
                "model_creator": "",
                "model_creator_link": f"/models/{model['id']}",
                "name": model['id'],
                "provider": None,
                "rank": 0.0,
                "type": "api",
                "context_length": model['context_length'],
                "architecture": {
                    "modality": model['architecture'].get('modality', ''),
                    "tokenizer": model['architecture'].get('tokenizer', ''),
                    "instruct_type": model['architecture'].get('instruct_type', '')
                },
                "per_request_limits": model.get('per_request_limits'),
                "variants": [{
                    "name": model['name'],
                    "size": f"Context length: {model['context_length']}",
                    "input_cost": float(model['pricing']['prompt']),
                    "output_cost": float(model['pricing']['completion'])
                }]
            }
            formatted_models.append(formatted_model)
        return formatted_models

    def save_models_yaml(self, output_path: str = None):
        """Fetches models, formats them, and saves to a YAML file"""
        if not output_path:
            output_path = str(Path(__file__).parent/"models.yaml")
        models_data = self.fetch_models()
        formatted_models = self.format_models(models_data)

        with open(output_path, 'w') as file:
            yaml.dump(formatted_models, file, default_flow_style=False)

        print(f"Models data saved to {output_path}")

# Usage
if __name__ == "__main__":
    fetcher = OpenRouterModelFetcher()
    fetcher.save_models_yaml()
