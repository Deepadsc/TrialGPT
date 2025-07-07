import subprocess
import argparse
import os
from dotenv import load_dotenv
import openai
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)
PYTHON_PATH = sys.executable

project_root = r"D:\Deepa_Nexturn\Task_Details\TrialGPT_Test_Cursor\Test_WF_Sigir"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add your OpenAI API key to the .env file.")
if not os.getenv('ANTHROPIC_API_KEY'):
    raise ValueError("ANTHROPIC_API_KEY not found in .env file. Please add your Anthropic API key to the .env file.")

def get_user_input():
    """
    Prompt user for required inputs interactively.
    """
    print("\n=== TrialGPT Configuration ===")

    # Ask which steps to run/skip first
    skip_keyword_gen = input("Skip keyword generation? (true/false, default: false): ").strip().lower() == 'true'
    skip_hybrid_fusion = input("Skip hybrid fusion retrieval? (true/false, default: false): ").strip().lower() == 'true'
    skip_matching = input("Skip matching? (true/false, default: false): ").strip().lower() == 'true'
    skip_aggregation = input("Skip aggregation? (true/false, default: false): ").strip().lower() == 'true'

    # Prompt for core corpus/model (needed for all steps)
    print("\nAvailable corpuses:")
    print("1. sigir")
    print("2. trec_2021")
    print("3. trec_2022")
    corpus_choice = input("Select corpus (1-3): ").strip()
    corpus_map = {'1': 'sigir', '2': 'trec_2021', '3': 'trec_2022'}
    corpus = corpus_map.get(corpus_choice, 'sigir')

    print("\nAvailable models:")
    print("1. gpt-4o-mini")
    print("2. gpt-4-turbo")
    print("3. gpt-4")
    print("4. gpt-3.5-turbo")
    print("5. claude-3-5-sonnet-20241022")
    print("6. claude-3-haiku-20240307")
    print("7. claude-3-opus-20240229")
    model_choice = input("Select model (1-7): ").strip()
    model_map = {'1': 'gpt-4o-mini', '2': 'gpt-4-turbo', '3': 'gpt-4', '4': 'gpt-3.5-turbo', '5': 'claude-3-5-sonnet-20241022', '6': 'claude-3-haiku-20240307', '7': 'claude-3-opus-20240229'}
    model = model_map.get(model_choice, 'gpt-4o-mini')

    # Only prompt for keyword generation query type if not skipping
    q_type = None
    if not skip_keyword_gen:
        print("\nAvailable query types:")
        print("1. raw (original queries)")
        print("2. gpt-4o-mini")
        print("3. gpt-4-turbo")
        print("4. gpt-4")
        print("5. gpt-3.5-turbo")
        print("6. claude-3-5-sonnet-20241022")
        print("7. claude-3-haiku-20240307")
        print("8. claude-3-opus-20240229")
        print("9. Clinician_A")
        print("10. Clinician_B")
        print("11. Clinician_C")
        print("12. Clinician_D")
        qtype_choice = input("Select query type (1-12): ").strip()
        qtype_map = {
            '1': 'raw', '2': 'gpt-4o-mini', '3': 'gpt-4-turbo', '4': 'gpt-4',
            '5': 'gpt-3.5-turbo',   '6': 'claude-3-5-sonnet-20241022', '7': 'claude-3-haiku-20240307',
            '8': 'claude-3-opus-20240229', '9': 'Clinician_A', '10': 'Clinician_B',
            '11': 'Clinician_C', '12': 'Clinician_D'
        }
        q_type = qtype_map.get(qtype_choice, 'raw')

    # Only prompt for hybrid fusion params if not skipping
    # Step-specific overwrite flags
    q_type = None
    topk = bm25_weight = medcpt_weight = rrf_k = batch_size = eligibility_threshold = exclusion_threshold = None
    overwrite_hybrid = overwrite_matching = overwrite_aggregation = None

    if not skip_hybrid_fusion:
        topk = int(input("Enter top-k value (number of results to retrieve, e.g., 20): "))
        bm25_weight = float(input("Enter BM25 weight (e.g., 1.0): "))
        medcpt_weight = float(input("Enter MedCPT weight (e.g., 1.0): "))
        rrf_k = int(input("Enter RRF k value (e.g., 20): "))
        overwrite_hybrid = input("Overwrite existing hybrid fusion results? (true/false, default: false): ").strip().lower()
        overwrite_hybrid = 'true' if overwrite_hybrid == 'true' else 'false'
        batch_size = int(input("Enter batch size (e.g., 32): "))
        eligibility_threshold = float(input("Enter eligibility threshold (0-1, e.g., 0.5): "))
        exclusion_threshold = float(input("Enter exclusion threshold (0-1, e.g., 0.3): "))
    if not skip_matching:
        overwrite_matching = input("Overwrite existing matching results? (true/false, default: false): ").strip().lower()
        overwrite_matching = 'true' if overwrite_matching == 'true' else 'false'
    if not skip_aggregation:
        overwrite_aggregation = input("Overwrite existing aggregation results? (true/false, default: false): ").strip().lower()
        overwrite_aggregation = 'true' if overwrite_aggregation == 'true' else 'false'

    return {
        'corpus': corpus,
        'model': model,
        'q_type': q_type,
        'topk': topk,
        'bm25_weight': bm25_weight,
        'medcpt_weight': medcpt_weight,
        'rrf_k': rrf_k,
        'batch_size': batch_size,
        'eligibility_threshold': eligibility_threshold,
        'exclusion_threshold': exclusion_threshold,
        'skip_keyword_gen': skip_keyword_gen,
        'skip_hybrid_fusion': skip_hybrid_fusion,
        'skip_matching': skip_matching,
        'skip_aggregation': skip_aggregation,
        'skip_ranking': False,     # Never skip ranking as it's the final step
        'overwrite_hybrid': overwrite_hybrid,
        'overwrite_matching': overwrite_matching,
        'overwrite_aggregation': overwrite_aggregation
    }

def run_step(description, command):
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        exit(1)
    else:
        print(f"✅ Completed: {description}")
    return result

def run_keyword_generation(corpus, model):
    print("\n=== Step 1: Keyword Generation ===")
    run_step("Running Keyword Generation", [
        "python", "trialgpt_retrieval/keyword_generation.py",
        "-c", corpus,
        "-m", model
    ])

def run_hybrid_fusion_retrieval(args):
    print("\n=== Step 2: Hybrid Fusion Retrieval ===")
    command = [
        "python", "trialgpt_retrieval/hybrid_fusion_retrieval.py",
        args['corpus'],  # corpus (required)
        args['q_type'],   # query type (required)
        str(args['rrf_k']),  # k parameter for RRF calculation
        str(args['bm25_weight']),  # BM25 weight
        str(args['medcpt_weight']),  # MedCPT weight
        "--overwrite", args['overwrite'],
        "--top_k", str(args['topk']),
        "--batch_size", str(args['batch_size']),
        "--eligibility_threshold", str(args['eligibility_threshold']),
        "--exclusion_threshold", str(args['exclusion_threshold'])
    ]
    run_step("Running Hybrid Fusion Retrieval", command)

def run_matching(corpus, model, overwrite):
    print("\n=== Step 3: Running Matching ===")
    run_step("Running Matching", [
        "python", "trialgpt_matching/run_matching.py",
        corpus,
        model,
        overwrite
    ])

def run_aggregation(corpus, model, matching_results_path, overwrite):
    print("\n=== Step 4: Running Aggregation ===")
    run_step("Running Aggregation", [
        "python", "trialgpt_ranking/run_aggregation.py",
        corpus,
        model,
        matching_results_path,
        overwrite
    ])

def run_ranking(corpus, model, overwrite=False):
    print("\n=== Step 5: Running Ranking ===")
    matching_results_path = f"results/matching_results_{corpus}_{model}.json"
    agg_results_path = f"results/aggregation_results_{corpus}_{model}.json"
    
    # Ask user if they want to overwrite existing ranking results
    if os.path.exists(f"results/ranking_results_{corpus}_{model}.json"):
        while True:
            overwrite_ranking = input("Ranking results already exist. Overwrite? (y/n): ").strip().lower()
            if overwrite_ranking in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'.")
        if overwrite_ranking != 'y':
            print("Skipping ranking as requested.")
            return
    
    run_step("Running Ranking", [
        "python", "trialgpt_ranking/rank_results.py",
        matching_results_path,  # matching_results_path
        agg_results_path,       # agg_results_path
        str(overwrite).lower(), # overwrite
        corpus,                 # corpus
        model                  # model
    ])

if __name__ == "__main__":
    # Get user input interactively
    args = get_user_input()
    
    # Show final configuration
    print("\n=== Final Configuration ===")
    print(f"Corpus: {args['corpus']}")
    print(f"Model: {args['model']}")
    print(f"Query Type: {args['q_type']}")
    print(f"Top-k: {args['topk']}")
    print(f"BM25 Weight: {args['bm25_weight']}")
    print(f"MedCPT Weight: {args['medcpt_weight']}")
    print(f"RRF k: {args['rrf_k']}")
    print(f"Batch Size: {args['batch_size']}")
    print(f"Eligibility Threshold: {args['eligibility_threshold']}")
    print(f"Exclusion Threshold: {args['exclusion_threshold']}")
    print(f"Skip Keyword Generation: {args['skip_keyword_gen']}")
    print(f"Skip Hybrid Fusion: {args['skip_hybrid_fusion']}")
    print(f"Skip Matching: {args['skip_matching']}")
    print(f"Skip Aggregation: {args.get('skip_aggregation', False)}")
    print(f"Skip Ranking: {args.get('skip_ranking', False)}")
    print("=" * 25 + "\n")
    
    # Ask for confirmation
    while True:
        proceed = input("Proceed with these settings? (y/n): ").strip().lower()
        if proceed in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'.")
    
    if proceed != 'y':
        print("Operation cancelled by user.")
        exit(0)
    
    # Run the pipeline
    if not args['skip_keyword_gen']:
        run_keyword_generation(args['corpus'], args['model'])
    
    if not args.get('skip_hybrid_fusion', False):
        # Use the correct overwrite flag for hybrid fusion
        args['overwrite'] = args.get('overwrite_hybrid', 'false')
        run_hybrid_fusion_retrieval(args)
    else:
        print("\n=== Skipping Hybrid Fusion Retrieval as requested. ===")
    if not args.get('skip_matching', False):
        run_matching(args['corpus'], args['model'], args.get('overwrite_matching', 'false'))
    else:
        print("\n=== Skipping Matching as requested. ===")

    # Aggregation step
    if not args.get('skip_aggregation', False):
        # Use the default matching results path for aggregation
        matching_results_path = f"results/matching_results_{args['corpus']}_{args['model']}.json"
        run_aggregation(args['corpus'], args['model'], matching_results_path, args.get('overwrite_aggregation', 'false'))
    else:
        print("\n=== Skipping Aggregation as requested. ===")

    # Ranking step
    if not args.get('skip_ranking', False):
        run_ranking(args['corpus'], args['model'], args.get('overwrite_ranking', 'false'))
    else:
        print("\n=== Skipping Ranking as requested. ===")
    
    print("\n=== Retrieval completed successfully! ===")
    print("\n🎉 TrialGPT Retrieval Complete!")

# then use openai.ChatCompletion.create(...) or openai.Completion.create(...)