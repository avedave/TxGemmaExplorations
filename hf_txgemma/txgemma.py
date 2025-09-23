import os
from dotenv import load_dotenv

load_dotenv()

USE_PIPELINE = True


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_VARIANT = "9b-predict"  # @param ["2b-predict", "9b-chat", "9b-predict", "27b-chat", "27b-predict"]

model_id = f"google/txgemma-{MODEL_VARIANT}"

if MODEL_VARIANT == "2b-predict":
    additional_args = {}
else:
    additional_args = {
        #uncoment this line to use 8-bit quantization (requires bitsandbytes library)
        #"quantization_config": BitsAndBytesConfig(load_in_8bit=True)
        #"quantization_config": BitsAndBytesConfig(load_in_4bit=True)
    }

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    **additional_args,
)

# import json
# from huggingface_hub import hf_hub_download

# tdc_prompts_filepath = hf_hub_download(
#     repo_id=model_id,
#     filename="tdc_prompts.json",
# )

# with open(tdc_prompts_filepath, "r") as f:
#     tdc_prompts_json = json.load(f)

# BBB_Martins
#task_name = "BBB_Martins"
#input_type = "{Drug SMILES}"
#drug_smiles = "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21"
#prompt = tdc_prompts_json[task_name].replace(input_type, drug_smiles)


drug_smiles = "CC(=O)Nc1cccc(-n2c(=O)n(C3CC3)c(=O)c3c(Nc4ccc(I)cc4F)n(C)c(=O)c(C)c32)c1"
target_amino_acid_sequence = "MSSGKRRNPLGLSLPPTVNEQSESGEATAEEATATVPLEEQLKKLGLTEPQTQRLSEFLQVKEGIKELSEDMLQTEGELGHGNGGVVNKCVHRKTGVIMARKLVHLEIKPSVRQQIVKELAVLHKCNSPFIVGFYGAFVDNNDISICMEYMDGLSLDIVLKKVGRLPEKFVGRISVAVVRGLTYLKDEIKILHRDVKPSNMLVNSNGEIKLCDFGVSGMLIDSMANSFVGTRSYMAPERLTGSHYTISSDIWSFGLSLVELLIGRYPVPAPSQAEYATMFNVAENEIELADSLEEPNYHPPSNPASMAIFEMLDYIVNGPPPTLPKRFFTDEVIGFVSKCLRKLPSERATLKSLTADVFFTQYADHDDQGEFAVFVKGTINLPKLNP"

import tdc_prompt_composer as tdcpc
prompt = tdcpc.BindingDB_ic50_prompt(drug_smiles, target_amino_acid_sequence)

def run_directly(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def run_pipeline(model, tokenizer, prompt):
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    outputs = pipe(prompt, max_new_tokens=8)
    response = outputs[0]["generated_text"]
    return response

if USE_PIPELINE:
    response = run_pipeline(model, tokenizer, prompt)
else:
    response = run_directly(model, tokenizer, prompt)

print(response)



