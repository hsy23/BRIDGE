import json
from pathlib import Path
from datasets import load_from_disk
from BRIDGE.utils.stable import seed_everything
from BRIDGE.utils.configure import Configure, Field as F

from datasets import load_dataset
if Path("data/10k_prompts_ranked").exists() is False:
    dataset = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
    dataset.save_to_disk("data/10k_prompts_ranked")


class Config(Configure):
    seed = F(int, default=42, help="Random seed")
    total = F(int, default=10, help="Total number of prompts to sample")
    output_file = F(str, default="datasets/prompts/prompts.json", help="Output file path")


config = Config()
config.parse_sys_args()

seed_everything(config.seed)
dataset = load_from_disk("data/10k_prompts_ranked")
prompts = []
for item in dataset.shuffle():
    if 100 <= len(item['prompt']) <= 300 and not any(char.isdigit() for char in item['prompt']):
        prompts.append({
            "query": item['prompt'],
        })
    if len(prompts) == config.total:
        break

with open(config.output_file, "w") as f:
    json.dump(prompts, f, indent=4)
