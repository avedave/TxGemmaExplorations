
import pandas as pd
from typing import List, Optional

def load_molecules(file_path: str, fields_to_keep: List[str]) -> Optional[pd.DataFrame]:
    """
    Loads a semicolon-delimited CSV file  from CHEMBL into a pandas DataFrame and filters it to keep specified columns.

    Args:
        file_path: The absolute path to the CSV file.
        fields_to_keep: A list of column names (strings) to keep in the final DataFrame.

    Returns:
        A pandas DataFrame containing only the specified columns, or None if an error occurs.
    """
    try:
        # Attempt to load the CSV file using a semicolon delimiter.
        df = pd.read_csv(file_path, delimiter=';')

        # Verify that all requested columns exist in the loaded DataFrame.
        missing_fields = [field for field in fields_to_keep if field not in df.columns]
        if missing_fields:
            print(f"Error: The following requested columns were not found in the file: {missing_fields}")
            return None

        # Filter the DataFrame to keep only the specified columns.
        filtered_df = df[fields_to_keep]
        
        print("DataFrame loaded and filtered successfully.")
        return filtered_df

    except FileNotFoundError:
        print(f"Error: File not found at the specified path: {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file. Please ensure it is a valid semicolon-delimited CSV: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

