
import os
import sys
import csv
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time
import time

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
ANTI_TARGET_SEQUENCE_PATH = os.path.join(SCRIPT_DIR, "data", "mek1_anti_target.txt")
NUM_RUNS = 5
MAX_COMPOUNDS = None  # Set to None to process all compounds

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

def get_sequence(filepath):
    """Reads the amino acid sequence from a file."""
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
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_csv_file_name = f"{timestamp}-ic50_predictions.csv"
    output_csv_path = os.path.join(SCRIPT_DIR, "predictions", output_csv_file_name)

    # Create predictions directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("Reading target sequence...")
    target_sequence = get_sequence(TARGET_SEQUENCE_PATH)
    print("Reading anti-target sequence...")
    anti_target_sequence = get_sequence(ANTI_TARGET_SEQUENCE_PATH)

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
                anti_predictions = []
                for i in range(NUM_RUNS):
                    print(f"  Run {i+1}/{NUM_RUNS}...")
                    prediction = predict_ic50(pipe, smiles, target_sequence)
                    anti_prediction = predict_ic50(pipe, smiles, anti_target_sequence)
                    if prediction is not None:
                        predictions.append(prediction)
                    if anti_prediction is not None:
                        anti_predictions.append(anti_prediction)
                
                if predictions and anti_predictions:
                    avg_ic50 = np.mean(predictions)
                    min_ic50 = np.min(predictions)
                    max_ic50 = np.max(predictions)
                    var_ic50 = np.var(predictions)
                    std_ic50 = np.std(predictions)
                    anti_avg_ic50 = np.mean(anti_predictions)
                    anti_min_ic50 = np.min(anti_predictions)
                    anti_max_ic50 = np.max(anti_predictions)
                    anti_var_ic50 = np.var(anti_predictions)
                    anti_std_ic50 = np.std(anti_predictions)

                    aTrg_Trg_ratio = anti_avg_ic50 / avg_ic50 if avg_ic50 != 0 else "Trg IC50=0"

                    result_row = {
                        "ChEMBL ID": chembl_id,
                        "Name": name,
                        "Smiles": smiles,
                        "aTrg/Trg": aTrg_Trg_ratio,
                        "Trg MEAN": avg_ic50,
                        "Trg MIN": min_ic50,
                        "Trg MAX": max_ic50,
                        "Trg VAR": var_ic50,
                        "Trg STD": std_ic50,
                        "aTrg MEAN": anti_avg_ic50,
                        "aTrg MIN": anti_min_ic50,
                        "aTrg MAX": anti_max_ic50,
                        "aTrg VAR": anti_var_ic50,
                        "aTrg STD": anti_std_ic50,
                    }

                    for i, p in enumerate(predictions):
                        result_row[f"Trg P{i+1}"] = p
                    
                    for i, p in enumerate(anti_predictions):
                        result_row[f"aTrg P{i+1}"] = p
                    
                    results.append(result_row)

                    print(f"  Results: aTrg_mean/Trg_mean={aTrg_Trg_ratio:.2f}")
                
                compounds_processed += 1

    print(f"Writing results to {output_csv_path}...")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        if results:
            # Define fieldnames based on the first result, ensuring consistent order
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hms_elapsed = time.strftime('%Hh:%Mm:%Ss', time.gmtime(elapsed_time))
    print(f"Script finished successfully in {hms_elapsed}.")

if __name__ == "__main__":
    main()
