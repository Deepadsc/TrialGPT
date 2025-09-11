#!/usr/bin/env python3

"""
Generate search keywords for patient descriptions using specified model and corpus.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root) 
from common.utils import setup_model, generate_response

def parse_arguments_kg():
    """
    Parse command-line arguments for the keyword generation script.

    This function sets up the argument parser and defines the required and optional
    arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate search keywords for patient descriptions.")

    # Required arguments
    parser.add_argument("-c", "--corpus", required=True, help="The corpus to process: trec_2021, trec_2022, or sigir")
    parser.add_argument("-m", "--model", required=True, help="The model to use for generating keywords")
    # Optional arguments

    return parser.parse_args()


def get_keyword_generation_messages(note):
    """
    Prepare the messages for keyword generation based on a patient note.

    Args:
        note (str): The patient description.

    Returns:
        list: A list of message dictionaries for the AI model.
    """
    system = """You are a medical research assistant specializing in clinical trial matching. Your task is to analyze patient descriptions to identify key medical conditions and assist in finding suitable clinical trials. Prioritize accuracy and relevance in your analysis."""

    prompt = f"""Please analyze the following patient description for clinical trial matching:

    ## {note}

    ### Instructions:
    1. Summarize the patient's main medical issues in 3-5 sentences.
    2. Generate At least 27-32 key conditions, ranked by relevance for clinical trial matching.
    3. Use standardized medical terminology (e.g., "Type 2 Diabetes" instead of "high blood sugar").
    4. Include conditions only if explicitly mentioned or strongly implied in the description.

    ### Output a JSON object in this format:
    **Provide ONLY a valid JSON object** with the following structure:
    {{
      "summary": "Brief patient summary",
      "conditions": ["Condition 1", "Condition 2", ...]
    }}

    ### Important Notes:
    - **MANDATORY**: Generate at least 30 keywords - clinical trial matching requires comprehensive coverage
    - If you are unsure about a condition, include it only if it is explicitly mentioned or strongly implied in the description.
    - **Do NOT include any text outside of the JSON object.** This means no notes, explanations, headers, or footers outside the JSON.

    Now, please process the patient description and respond with the JSON object.
    """

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]


def main(args):
    """
    Generate search keywords for patient descriptions using specified model and corpus.

    This function processes patient descriptions from a given corpus using either GPT-4o-mini or Claude models
    to generate relevant medical keywords. It saves the results to a JSON file.
    """
    outputs = {}
    failed_outputs = {}

    model_type, model_instance = setup_model(args.model)

    # Count total lines in the input file for progress tracking
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        total_lines = sum(1 for _ in f)

    # Process each query in the input file
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        for line in tqdm(f, total=total_lines, desc=f"Processing {args.corpus} queries"):
            try:
                entry = json.loads(line)
                messages = get_keyword_generation_messages(entry["text"])
                entry_id = entry["_id"]
                
                print(f"\nProcessing entry {entry_id}")
                print(f"Using model: {args.model}")
                
                try:
                    if model_type == 'gpt':
                        output = generate_response(model_type, model_instance, messages, args.model)
                    else:
                        output = generate_response(model_type, model_instance, messages)
                    
                    print(f"Raw API response: {output[:200]}..." if output else "Empty API response")
                    
                    if not output:
                        raise ValueError("Empty response from API")
                        
                    try:
                        parsed_output = json.loads(output)
                        if not isinstance(parsed_output, dict) or 'conditions' not in parsed_output:
                            raise ValueError("Response missing required 'conditions' field")
                        outputs[entry_id] = parsed_output
                        print(f"Successfully processed entry {entry_id}")
                        
                    except json.JSONDecodeError as je:
                        print(f"JSON decode error for entry {entry_id}: {str(je)}")
                        print(f"Raw response: {output}")
                        failed_outputs[entry_id] = {
                            "error": f"Failed to parse JSON: {str(je)}",
                            "raw_output": output
                        }
                        
                except Exception as e:
                    print(f"Error generating response for entry {entry_id}: {str(e)}")
                    failed_outputs[entry_id] = {
                        "error": f"Generation error: {str(e)}",
                        "raw_output": output if 'output' in locals() else "No output generated"
                    }
            except Exception as e:
                print(f"Error processing entry {entry['_id']}: {str(e)}")
                failed_outputs[entry["_id"]] = {
                    "error": str(e),
                    "raw_entry": line
                }

    # Save successful outputs
    output_file = f"results/retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)
    print(f"Results saved to {output_file}")

    # Save failed outputs
    failed_output_file = f"results/failed_retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(failed_output_file, "w") as f:
        json.dump(failed_outputs, f, indent=4)
    print(f"Failed results saved to {failed_output_file}")

    # Print summary
    print(f"Total entries processed: {len(outputs) + len(failed_outputs)}")
    print(f"Successful entries: {len(outputs)}")
    print(f"Failed entries: {len(failed_outputs)}")


if __name__ == "__main__":
    args = parse_arguments_kg()
    main(args)