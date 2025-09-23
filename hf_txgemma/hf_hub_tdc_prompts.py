import json
from huggingface_hub import hf_hub_download

MODEL_VARIANT = "9b-predict"  # @param ["2b-predict", "9b-chat", "9b-predict", "27b-chat", "27b-predict"]

model_id = f"google/txgemma-{MODEL_VARIANT}"

tdc_prompts_filepath = hf_hub_download(
    repo_id=model_id,
    filename="tdc_prompts.json",
)

with open(tdc_prompts_filepath, "r") as f:
    tdc_prompts_json = json.load(f)