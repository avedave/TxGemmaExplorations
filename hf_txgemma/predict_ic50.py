
import os
import sys
import csv
import numpy as np
from dotenv import load_dotenv

# Add hf_txgemma to path
#sys.path.append(os.path.join(os.path.dirname(__file__), 'hf_txgemma'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import tdc_prompt_composer as tdcpc

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_VARIANT = "9b-predict"
MODEL_ID = f"google/txgemma-{MODEL_VARIANT}"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "data", "MEK_Inhibitors.csv")
TARGET_SEQUENCE_PATH = os.path.join(SCRIPT_DIR, "data", "mek2_target.txt")
OUTPUT_CSV_PATH = "ic50_predictions.csv"
NUM_RUNS = 2
MAX_COMPOUNDS = 2

def get_model_and_tokenizer():
    """Loads and returns the model and tokenizer."""
    additional_args = {}
    if MODEL_VARIANT != "2b-predict":
        # To use 8-bit quantization (requires bitsandbytes library):
        # additional_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # To use 4-bit quantization:
        # additional_args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        **additional_args,
    )
    return model, tokenizer

def get_target_sequence(filepath):
    """Reads the target amino acid sequence from a file."""
    with open(filepath, "r") as f:
        return f.read().strip()

def predict_ic50(pipe, smiles, target_sequence):
    """Generates a prompt and runs the prediction."""
    prompt = tdcpc.BindingDB_ic50_prompt(smiles, target_sequence)
    outputs = pipe(prompt, max_new_tokens=8)
    response = outputs[0]["generated_text"]
    
    # Extract the numeric value from the response
    try:
        # The model output is the number after "Answer:"
        answer_part = response.split("Answer:")[1]
        # The value might have leading/trailing spaces
        result_value = answer_part.strip()
        return float(result_value)
    except (IndexError, ValueError) as e:
        print(f"Could not parse ic50 value from response: {response}. Error: {e}")
        return None

def main():
    """Main function to run the prediction and analysis."""
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("Reading target sequence...")
    target_sequence = get_target_sequence(TARGET_SEQUENCE_PATH)

    print(f"Reading compounds from {INPUT_CSV_PATH}...")
    
    results = []
    compounds_processed = 0
    
    with open(INPUT_CSV_PATH, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            if MAX_COMPOUNDS and compounds_processed >= MAX_COMPOUNDS:
                print(f"Reached MAX_COMPOUNDS limit of {MAX_COMPOUNDS}. Stopping.")
                break

            chembl_id = row.get("ChEMBL ID")
            name = row.get("Name")
            smiles = row.get("Smiles")

            if chembl_id and name and smiles:
                print(f"Processing {name} ({chembl_id})...")
                predictions = []
                for i in range(NUM_RUNS):
                    print(f"  Run {i+1}/{NUM_RUNS}...")
                    prediction = predict_ic50(pipe, smiles, target_sequence)
                    if prediction is not None:
                        predictions.append(prediction)
                
                if predictions:
                    avg_ic50 = np.mean(predictions)
                    min_ic50 = np.min(predictions)
                    max_ic50 = np.max(predictions)
                    
                    results.append({
                        "ChEMBL ID": chembl_id,
                        "Name": name,
                        "Smiles": smiles,
                        "Average IC50": avg_ic50,
                        "Min IC50": min_ic50,
                        "Max IC50": max_ic50,
                    })
                    print(f"  Results: Avg={avg_ic50:.2f}, Min={min_ic50:.2f}, Max={max_ic50:.2f}")
                
                compounds_processed += 1

    print(f"Writing results to {OUTPUT_CSV_PATH}...")
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as outfile:
        if results:
            writer = csv.DictWriter(outfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("Script finished successfully.")

if __name__ == "__main__":
    main()
