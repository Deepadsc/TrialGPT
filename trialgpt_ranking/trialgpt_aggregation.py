__author__ = "qiao"

"""
TrialGPT-Ranking main functions.
"""

import json
import os
import sys
import re

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import generate_response

def fix_json_string(json_str, nct_id='unknown'):
    """
    Comprehensive function to fix malformed JSON strings.
    
    This function applies multiple cleaning and repair strategies to handle
    various JSON formatting issues, including missing commas, unbalanced braces,
    and other common syntax errors.
    
    Args:
        json_str (str): The potentially malformed JSON string to fix
        nct_id (str): Trial ID for context (used in debug info)
        
    Returns:
        dict: The parsed JSON object, or an empty dict if parsing fails
    """
    # Return empty dict for empty strings
    if not json_str or not json_str.strip():
        return {}
    
    # Initial cleaning
    clean_str = json_str.replace('\ufeff', '')  # Remove BOM
    clean_str = clean_str.strip()
    
    # Remove leading/trailing newlines
    while clean_str.startswith('\n'):
        clean_str = clean_str[1:]
    while clean_str.endswith('\n'):
        clean_str = clean_str[:-1]
    
    # Try standard parsing first
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        pass
    
    # Fix common JSON formatting issues
    # Fix missing commas between objects
    clean_str = clean_str.replace('}\n  "', '},\n  "')
    clean_str = clean_str.replace('}\n"', '},\n"')
    clean_str = clean_str.replace('}\n {', '}, {')
    clean_str = clean_str.replace('}\n{', '},{')
    clean_str = clean_str.replace('} "', '}, "')
    clean_str = clean_str.replace('}"', '},')
    
    # Fix missing commas after values before next key
    clean_str = re.sub(r'"\s*\n\s*"', '",\n"', clean_str)
    clean_str = re.sub(r']\s*\n\s*"', '],\n"', clean_str)
    clean_str = re.sub(r'"\s*\n\s*\{', '",\n{', clean_str)
    clean_str = re.sub(r'(\d+|true|false|null)\s*\n\s*"', '\\1,\n"', clean_str)
    
    # Try parsing with initial fixes
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError as e:
        # More aggressive cleaning for specific error types
        if "Expecting ',' delimiter" in str(e):
            # Find the position mentioned in the error
            match = re.search(r'char (\d+)', str(e))
            if match:
                pos = int(match.group(1))
                if pos < len(clean_str):
                    # Insert a comma at the position
                    clean_str = clean_str[:pos] + ',' + clean_str[pos:]
        
        # Fix unbalanced braces
        if "Expecting property name enclosed in double quotes" in str(e):
            if not clean_str.startswith('{'):
                clean_str = '{' + clean_str
            if not clean_str.endswith('}'):
                clean_str = clean_str + '}'
    
    # Try parsing with more aggressive fixes
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # Try removing escape characters and fixing quotes
        clean_str = clean_str.replace('\\', '')
        clean_str = clean_str.replace('"{', '{').replace('}"', '}')
        
        # Ensure the string starts and ends with braces
        if not clean_str.startswith('{'):
            clean_str = '{' + clean_str
        if not clean_str.endswith('}'):
            clean_str = clean_str + '}'
    
    # Try parsing after escape character fixes
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # Last resort: try to extract individual criterion entries
        try:
            # Extract key-value pairs using regex
            pattern = r'"(\d+)"\s*:\s*\[(.*?)\]'
            matches = re.findall(pattern, json_str, re.DOTALL)
            
            if matches:
                result = {}
                for key, value in matches:
                    # Try to parse the value as a list
                    try:
                        result[key] = json.loads('[' + value + ']')
                    except:
                        # If parsing fails, use a placeholder
                        result[key] = ["Error parsing", [], "Error parsing"]
                return result
        except:
            pass
    
    # If all parsing attempts fail, return an empty dict
    return {}

def convert_criteria_pred_to_string(
        prediction: dict,
        trial_info: dict,
) -> str:
    """
    Convert TrialGPT prediction to a linear string of criteria.

    This function takes the structured prediction data and trial information,
    and converts it into a more readable, linear string format. It organizes
    the information by inclusion and exclusion criteria, and for each criterion,
    it provides the full text along with the relevance, evident sentences (if any),
    and eligibility predictions.

    Args:
        prediction (dict): A dictionary containing the TrialGPT predictions.
        trial_info (dict): A dictionary containing information about the clinical trial.

    Returns:
        str: A formatted string containing the criteria and their predictions.
    """
    output = ""
    nct_id = trial_info.get('nct_id', 'unknown')

    # Process both inclusion and exclusion criteria
    for inc_exc in ["inclusion", "exclusion"]:
        # Create a dictionary to map indices to criteria
        idx2criterion = {}
        # Check if criteria exists in trial_info
        criteria_key = inc_exc + "_criteria"
        if criteria_key not in trial_info:
            # Silently continue if criteria not found
            continue
            
        # Split the criteria text by double newlines
        criteria = trial_info[criteria_key].split("\n\n")

        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()

            # Skip headers or invalid criteria
            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue
            if len(criterion) < 5:
                continue

            # Add valid criterion to the dictionary
            idx2criterion[str(idx)] = criterion
            idx += 1
                # Check if the prediction contains data for this criteria type
        if inc_exc not in prediction:
            # Silently continue if criteria not found in prediction
            continue

        # Get the prediction data
        pred_data = prediction[inc_exc]
        
        # If it's a string (Format 2), try to parse it as JSON
        if isinstance(pred_data, str):
            # Use our comprehensive JSON fixing function
            pred_data = fix_json_string(pred_data, nct_id)
        
        # Ensure pred_data is a dictionary
        if not isinstance(pred_data, dict):
            # If not a dictionary, skip silently
            continue
        
        # Process predictions for each criterion
        for idx, (criterion_idx, preds) in enumerate(pred_data.items()):

            # Skip criteria not in our dictionary
            if criterion_idx not in idx2criterion:
                continue

            criterion = idx2criterion[criterion_idx]
              # Handle different formats of preds
            if isinstance(preds, list):
                # Format 1: preds is a list with 3 elements
                # Skip predictions without exactly 3 elements
                if len(preds) != 3:
                    continue

                # Build the output string for this criterion
                output += f"{inc_exc} criterion {idx}: {criterion}\n"
                output += f"\tPatient relevance: {preds[0]}\n"
                # Add evident sentences if they exist
                if len(preds[1]) > 0:
                    output += f"\tEvident sentences: {preds[1]}\n"
                output += f"\tPatient eligibility: {preds[2]}\n"
            elif isinstance(preds, dict):
                # Format 2: preds is a dictionary with specific keys
                # Extract the relevant information from the dictionary
                relevance = preds.get(0, "")
                evidence = preds.get(1, [])
                eligibility = preds.get(2, "")
                
                # Build the output string for this criterion
                output += f"{inc_exc} criterion {idx}: {criterion}\n"
                output += f"\tPatient relevance: {relevance}\n"
                # Add evident sentences if they exist
                if evidence and len(evidence) > 0:
                    output += f"\tEvident sentences: {evidence}\n"
                output += f"\tPatient eligibility: {eligibility}\n"
            else:
                # Unknown format, skip
                continue

    return output


def convert_pred_to_prompt(patient: str, pred: dict, trial_info: dict) -> tuple[str, str]:
    """Generate improved system and user prompts for clinical trial relevance and eligibility scoring."""

    # Construct trial information string
    trial = (
        f"Title: {trial_info['brief_title']}\n"
        f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
        f"Summary: {trial_info['brief_summary']}"
    )
    
        # Ensure pred has the right format before conversion
    processed_pred = {}
    for key in ["inclusion", "exclusion"]:
        if key in pred:
            if isinstance(pred[key], str):
                try:
                    processed_pred[key] = json.loads(pred[key])
                except json.JSONDecodeError:
                    # If parsing fails, use empty dict to avoid errors
                    processed_pred[key] = {}
            else:
                processed_pred[key] = pred[key]
        else:
            processed_pred[key] = {}


    # Convert prediction to string
    pred_string = convert_criteria_pred_to_string(pred, trial_info)

    # System prompt
    system_prompt = """You are a clinical trial recruitment specialist. Your task is to assess patient-trial relevance and eligibility based on a patient note, clinical trial description, and criterion-level eligibility predictions.

### Instructions:

1. **Relevance Score (R)**:
   - Provide a detailed explanation of why the patient is relevant (or irrelevant) to the clinical trial.
   - Predict a relevance score (R) between `0` and `100`:
     - `R=0`: The patient is completely irrelevant to the clinical trial.
     - `R=100`: The patient is perfectly relevant to the clinical trial.

2. **Eligibility Score (E)**:
   - Provide a detailed explanation of the patient’s eligibility for the clinical trial.
   - Predict an eligibility score (E) between `-R` and `R`:
     - `E=-R`: The patient is ineligible (meets no inclusion criteria or is excluded by all exclusion criteria).
     - `E=0`: Neutral (no relevant information for any criteria is found).
     - `E=R`: The patient is fully eligible (meets all inclusion criteria and no exclusion criteria).

Prioritize accuracy, use standardized medical terminology in your analysis, and ensure that your scores are within the specified ranges (`0 ≤ R ≤ 100` and `-R ≤ E ≤ R`)."""

    # User prompt
    user_prompt = f"""Analyze the following information:

### Patient Note:
{patient}

### Clinical Trial:
{trial}

### Criterion-level Eligibility Predictions:
{pred_string}

### Output Instructions:
- **Provide ONLY a valid JSON object** with the exact structure below:
```json
{{
  "relevance_explanation": "Your detailed reasoning for the relevance score",
  "relevance_score_R": float_value_between_0_and_100,
  "eligibility_explanation": "Your detailed reasoning for the eligibility score",
  "eligibility_score_E": float_value_between_negative_R_and_positive_R
}}
```

- **Critical Rules:**
  1. **Do NOT include any text outside of the JSON object.**
  2. All reasoning and additional context **MUST** be included in the `relevance_explanation` and `eligibility_explanation` fields.
  3. Ensure that all values are valid:
     - The `relevance_score_R` must be a float between `0` and `100`.
     - The `eligibility_score_E` must be a float in the range of `[-relevance_score_R, +relevance_score_R]`.

### Example Output:
```json
{{
  "relevance_explanation": "The patient has a condition explicitly mentioned in the trial criteria, making this trial highly relevant.",
  "relevance_score_R": 85.0,
  "eligibility_explanation": "The patient meets most inclusion criteria but is disqualified by one exclusion criterion (hypertension).",
  "eligibility_score_E": -20.0
}}
```

- **Additional Notes**:
  - Populate all fields, even if explanations are brief.
  - If there’s no additional information to provide, use an empty string `""` for the explanation fields.
  - **Any text outside the JSON structure will be considered invalid output.**
"""

    return system_prompt, user_prompt


def trialgpt_aggregation(patient: str, trial_results: dict, trial_info: dict, model: str, model_type: str,
                         model_instance: any):
    debug_data = []
    debug_filename = f"results/messages_trialgpt_aggregation.json"
    try:
        # Create a copy of trial_results to avoid modifying the original
        processed_results = {}
        for key, value in trial_results.items():
            if key in ["inclusion", "exclusion"]:
                if isinstance(value, str):
                    try:
                        processed_results[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # If parsing fails, use empty dict to avoid errors
                        processed_results[key] = {}
                else:
                    processed_results[key] = value
            else:
                processed_results[key] = value
                
        system_prompt, user_prompt = convert_pred_to_prompt(
            patient,
            processed_results,
            trial_info
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = generate_response(model_type, model_instance, messages, model)
        result = result.strip("`").strip("json")

        # Prepare the new debug entry
        debug_entry = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": result
        }

        # Read existing debug data or create an empty list
        if os.path.exists(debug_filename):
            with open(debug_filename, 'r') as f:
                try:
                    debug_data = json.load(f)
                except json.JSONDecodeError:
                    debug_data = []
        else:
            debug_data = []

        # Append the new entry
        debug_data.append(debug_entry)

        #TODO: make debug optional, would be slow for production datasets
        # Write the updated debug data back to the file
        with open(debug_filename, 'w') as f:
            json.dump(debug_data, f, indent=4)

        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {result}")
            result = {
                "relevance_explanation": "Error parsing JSON",
                "relevance_score_R": 0,
                "eligibility_explanation": "Error parsing JSON",
                "eligibility_score_E": 0
            }

        return result
    except Exception as e:
        print(f"Error in trialgpt_aggregation: {e}")
        return {
            "relevance_explanation": f"Error processing data: {str(e)}",
            "relevance_score_R": 0,
            "eligibility_explanation": f"Error processing data: {str(e)}",
            "eligibility_score_E": 0
        }
