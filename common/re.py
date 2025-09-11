import json
import os
import anthropic
import openai

def load_corpus_details(corpus_path: str) -> dict:
    """
    Load corpus details into a dictionary for quick lookup.

    Args:
        corpus_path (str): Path to the corpus JSONL file.

    Returns:
        dict: A dictionary containing corpus details, keyed by NCT ID.
    """
    corpus_details = {}
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            nct_id = entry["_id"]
            metadata = entry.get("metadata", {})
            corpus_details[nct_id] = {
                "brief_title": metadata.get("brief_title", entry.get("title", "")),
                "phase": metadata.get("phase", ""),
                "drugs": metadata.get("drugs", ""),
                "drugs_list": metadata.get("drugs_list", []),
                "diseases": metadata.get("diseases", ""),
                "diseases_list": metadata.get("diseases_list", []),
                "enrollment": metadata.get("enrollment", ""),
                "inclusion_criteria": metadata.get("inclusion_criteria", ""),
                "exclusion_criteria": metadata.get("exclusion_criteria", ""),
                "brief_summary": metadata.get("brief_summary", entry.get("text", "")),
                "NCTID": nct_id
            }
    return corpus_details

def setup_model(model_name):
    """
    Set up the model based on the model name.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        tuple: (model_type, model_instance)
            model_type is either 'claude' or 'gpt'
            model_instance is either an Anthropic client or an OpenAI client
    """
    if model_name.startswith('gpt'):
        client = openai.OpenAI()
        return 'gpt', client
    elif model_name.startswith('claude'):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        return 'claude', client
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def generate_response(model_type, model_instance, messages, model_name=None):
    """
    Generate a response using either Claude or GPT models.

    Args:
        model_type (str): Either 'claude' or 'gpt'.
        model_instance: Either an Anthropic client or an OpenAI client.
        messages (list): The input messages for the model.
        model_name (str, optional): The name of the model to use.

    Returns:
        str: The generated response.
    """
    try:
        if model_type == 'claude':
            system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
            user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
            if not user_messages:
                raise ValueError("At least one user message is required for Claude API")
            claude_messages = [{"role": "user", "content": "\n".join(user_messages)}]

            if model_name:
                if 'sonnet' in model_name.lower():
                    selected_model = "claude-3-5-sonnet-20241022"
                elif 'haiku' in model_name.lower():
                    selected_model = "claude-3-haiku-20240307"
                elif 'opus' in model_name.lower():
                    selected_model = "claude-3-opus-20240229"
                else:
                    selected_model = model_name
            else:
                selected_model = "claude-3-haiku-20240307"

            response = model_instance.messages.create(
                model=selected_model,
                max_tokens=4000,
                system=system_message,
                messages=claude_messages
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise ValueError("Empty response from Claude API")

        elif model_type == 'gpt':
            response = model_instance.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip("`").strip("json")

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except Exception as e:
        error_msg = f"Error generating response with {model_type} model: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Messages: {messages}")
        if model_type == 'claude':
            print(f"Model: {model_name}")
        raise Exception(error_msg) from e
