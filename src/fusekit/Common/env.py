from pathlib import Path

# from fusekit.Modeling.claude import Claude3p5_Sonnet, Claude3_Opus, Claude3_Sonnet, Claude3_Haiku
# from fusekit.Modeling.chatGPT import GPT4o, GPT4o_mini, GPTo1, GPTo1_mini
# from fusekit.Modeling.gemini import Gemini1p5_Pro, Gemini1p5_Flash_8B, Gemini1p5_Flash, Gemini2_Flash
from collections import defaultdict

import fusekit.Common.utils as utils

# Set the project directory based on the current file's location
PROJECT_DIR = Path(__file__).resolve().parent / '../'
ROOT_DIR = PROJECT_DIR / '../../../../'

# Define paths relative to the project directory
apikeys = PROJECT_DIR / 'apikeys'
datasets = PROJECT_DIR / 'Datasets'

models = ROOT_DIR / 'models'

config = PROJECT_DIR / 'config.yml'
results = PROJECT_DIR / 'Results'
survey_responses = PROJECT_DIR / 'survey-responses'
default_checkpoint_path = models / 'checkpoints'

class SurveyResponses:
    root = PROJECT_DIR / 'survey-responses'
    military = root / 'MilitaryAccuracy'
    natural_world = root / 'NaturalWorldAccuracy'
    urban = root / 'UrbanAccuracy'

class APIKeys:
    openai = apikeys / 'openai.apikey'
    openai_org = apikeys / 'openai.org'
    claude = apikeys / 'claude.apikey'
    claude_org = apikeys / 'claude.org'
    gemini = apikeys / 'gemini.apikey'
    
class ModelPath:
    llama2_7b = models/ 'llama2' / 'hf' / '7B'
    llama2_13b = models / 'llama2' / 'hf' / '13B'
    llama2_70b = models / 'llama2' / 'hf' / '70B'

    llava_next_7b_vicuna = models / 'llava-next' / '7B-Vicuna'
    llava_next_7b_mistral = models / 'llava-next' / '7B-Mistral'
    llava_next_13b_vicuna = models / 'llava-next' / '13B-Vicuna'
    llava_next_34b = models / 'llava-next' / '34B'
    llava_next_72b = models / 'llava-next' / '72B'
    llava_next_110b = models / 'llava-next' / '110B'
    
    pixtral_12b = models / 'pixtral' / '12B'

    qwen2_2b = models / 'qwen2' / '2B-Instruct'
    qwen2_7b = models / 'qwen2' / '7B-Instruct'
    
    phi3_5_vision = models / 'phi3' / 'Vision-Instruct'

    llama3_8b = models / 'llama3' / 'hf' / '8B'
    llama3_11b_vision_instruct = models/ 'llama3' / 'hf' / '11B-Vision-Instruct' 
    llama3_90b_vision_instruct = models / 'llama3' / 'hf' / '90B-Vision-Instruct'

class DatasetPath:
    commonsenseqa = PROJECT_DIR / datasets / 'CommonsenseQA'

class FineTunedAdapters:
    finetuned = PROJECT_DIR / 'finetuned_adapters'
    llama2_7b = finetuned / 'llama2_7b'
    llama3_8b = finetuned / 'llama3_8b'

