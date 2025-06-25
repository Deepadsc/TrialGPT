import streamlit as st
import json
import os
import re
import streamlit as st
from pathlib import Path

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="TrialGPT - Clinical Trial Matching",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIG ---

# Base results directory
RESULTS_DIR = "results"

# Find the most recent matching results file
def find_matching_results_file():
    """Find the most recent matching results file in the results directory."""
    try:
        results_path = Path(RESULTS_DIR)
        if not results_path.exists():
            return None
            
        # Look for files matching the pattern 'matching_results_*.json'
        matching_files = list(results_path.glob('matching_results_*.json'))
        
        if not matching_files:
            return None
            
        # Sort by modification time (newest first) and return the first one
        matching_files.sort(key=os.path.getmtime, reverse=True)
        return str(matching_files[0])
    except Exception as e:
        st.warning(f"Error finding matching results file: {str(e)}")
        return None

# Set paths
MATCHING_RESULTS_PATH = find_matching_results_file()
RANKING_RESULTS_DIR = os.path.join(RESULTS_DIR, "trial_rankings")
CORPUS_PATH = os.path.join(RESULTS_DIR, "bm25_corpus_trec_2022.json")

if not MATCHING_RESULTS_PATH:
    st.warning("⚠️ No matching results file found. Some features may be limited.")
if not os.path.exists(RANKING_RESULTS_DIR):
    st.warning("⚠️ No ranking directory found. Ranking features will not be available.")

# --- LOAD DATA ---
@st.cache_data
def load_matching_results():
    with open(MATCHING_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_ranking_results(query_id):
    ranking_file = Path(RANKING_RESULTS_DIR) / f"trialranking_{query_id}.txt"
    if not ranking_file.exists():
        return None
    
    trials = []
    current_trial = {}
    
    with open(ranking_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('NCT'):
                if current_trial:  # Save previous trial
                    trials.append(current_trial)
                # Extract NCT ID and scores
                parts = line.split(':')
                current_trial = {
                    'nctid': parts[0],
                    'scores': {}
                }
                # Parse scores
                score_parts = parts[1].split(',')
                for part in score_parts:
                    if '=' in part:
                        key, value = part.split('=')
                        current_trial['scores'][key.strip()] = float(value.strip())
                i += 1
                # Get trial details
                current_trial['brief_summary'] = lines[i].replace('Brief Summary:', '').strip()
                i += 1
                current_trial['relevance'] = lines[i].replace('Relevance Explanation:', '').strip()
                i += 1
                current_trial['eligibility'] = lines[i].replace('Eligibility Explanation:', '').strip()
                i += 1  # Skip the empty line
            i += 1
        if current_trial:  # Add the last trial
            trials.append(current_trial)
    return trials

def get_trial_by_nctid(nctid, corpus_data):
    if nctid in corpus_data.get("corpus_nctids", []):
        return {"NCTID": nctid, "status": "In corpus"}
    return {"NCTID": nctid, "status": "Not found in corpus"}

@st.cache_data
def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """Remove strikethrough patterns from text"""
    if not text:
        return ""
    # Remove strikethrough patterns like '~text~' or '~~text~~'
    text = re.sub(r'~+([^~]+)~+', r'\1', text)
    return text

@st.cache_data
def load_patient_queries():
    queries_file = Path("dataset/trec_2022/queries.jsonl")
    queries = {}
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line)
                if '_id' in query:
                    queries[query['_id']] = query
    except Exception as e:
        st.error("Error loading patient data. Please check the data files.")
    return queries

def get_qrels_status(qrels_score):
    """Convert qrels score to status text and color."""
    if qrels_score == 2:
        return 'Eligible', '#4CAF50'  # Green
    elif qrels_score == 1:
        return 'Excluded', '#FFC107'  # Yellow
    else:  # 0 or any other value
        return 'Not relevant', '#F44336'  # Red

def get_score_color(score, score_type='trial'):
    if score_type == 'trial':
        # For trial scores, use the qrels score if available
        status, color = get_qrels_status(score)
        return color
    else:  # For matching and agg scores
        if score > 0.5:
            return '#4CAF50'
        elif score > 0:
            return '#8BC34A'
        elif score > -0.5:
            return '#FFC107'
        else:
            return '#F44336'

# --- UI ---

# Clear the initialized flag on page load
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.query_id = ""

# Custom CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        max-width: 1000px;
        padding: 1rem;
        margin: 0 auto;
    }
    
    /* Page title */
    h1 {
        color: #1e40af;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Patient card */
    .patient-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #4a6fa5;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Trial card */
    .trial-card {
        background: white;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 3rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
    }
    
    /* Add space and visual separator between trial cards */
    .trial-card:not(:last-child) {
        margin-bottom: 4rem;
    }
    
    .trial-card:not(:last-child)::after {
        content: '';
        position: absolute;
        bottom: -2rem;
        left: 5%;
        right: 5%;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            #e2e8f0, 
            #4CC9F0, 
            #7209B7, 
            #e2e8f0, 
            transparent);
        opacity: 0.7;
    }
    

    .trial-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Trial header */
    .trial-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #eee;
    }
    
    /* Trial number */
    .trial-number {
        background: linear-gradient(135deg, #4a6fa5 0%, #3a5a80 100%);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Trial title */
    .trial-title {
        font-weight: 700;
        color: #1e40af;
        font-size: 1.2rem;
        flex-grow: 1;
        letter-spacing: -0.3px;
    }
    
    /* Section headers */
    h2, h3, h4 {
        color: #2E8B8B;  /* Sophisticated teal - professional and calming */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.3rem;
    }
    
    h2 {
        font-size: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.3rem;
    }
    
    /* Trial score */
    .trial-score {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565c0;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 20px;
        font-weight: 500;
        color: #64748b;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #f1f5f9;
        color: #1e40af;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
    }
    
    /* Content cards */
    .content-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        line-height: 1.6;
        color: #334155;
        border-left: 3px solid #e2e8f0;
    }
    
    /* Score cards */
    .score-card {
        background: white;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-label {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .score-value {
        font-weight: 600;
        color: #1e40af;
        font-size: 1.1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .trial-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .trial-score {
            align-self: flex-start;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main title with icon and styling
st.markdown("""
    <h1 style='color: #1e4b8b; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>
        🔍 TrialGPT Matching
    </h1>
""", unsafe_allow_html=True)

# Load data
try:
    # Show loading state
    with st.spinner('Loading data...'):
        # Load corpus data (required)
        corpus_data = load_corpus()
        if not corpus_data:
            st.error(f"❌ Failed to load corpus data from {os.path.basename(CORPUS_PATH) if CORPUS_PATH else 'N/A'}")
            st.error("Please make sure the corpus file exists and is a valid JSON file.")
            st.stop()
        
        # Load patient queries (required)
        patient_queries = load_patient_queries()
        if not patient_queries:
            st.error("❌ No patient queries found. Please check the queries file.")
            st.stop()
        
        # Load matching results (optional)
        matching_results = load_matching_results()
        if matching_results is None:
            st.warning("⚠️ Matching results not found. Some features may be limited.")
        
        # Get available query IDs
        query_ids = list(patient_queries.keys())
        if not query_ids:
            st.error("❌ No query IDs found in patient queries")
            st.stop()
        
        # Initialize or get query_id from session state
        if 'query_id' not in st.session_state:
            st.session_state.query_id = query_ids[0] if query_ids else None
        
        # Create query selector
        query_id = st.selectbox("Select a query ID:", 
                              query_ids, 
                              index=query_ids.index(st.session_state.query_id) if st.session_state.query_id in query_ids else 0)
        
        # Update session state
        st.session_state.query_id = query_id
        
        # Load ranking results for the selected query
        ranking_results = load_ranking_results(query_id) if RANKING_RESULTS_DIR and os.path.exists(RANKING_RESULTS_DIR) else None
        
        # Show success message
        st.success("✅ Data loaded successfully!")
        
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.error("Please check the error message above and make sure all required files are in the correct location.")
    st.stop()

# Initialize sidebar for quick view
with st.sidebar:
    st.markdown("<h3 style='color: #1e4b8b;'>Quick View</h3>", unsafe_allow_html=True)
    st.markdown(f"**Patient ID:** {query_id}")
    
    # Add custom CSS for status badges
    st.markdown("""
        <style>
            .status-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
                margin-left: 8px;
            }
            .status-eligible { background-color: #e8f5e9; color: #2e7d32; }
            .status-excluded { background-color: #fff8e1; color: #ff8f00; }
            .status-not-relevant { background-color: #ffebee; color: #c62828; }
        </style>
    """, unsafe_allow_html=True)
    
    # Add a placeholder for trial suggestions that will be filled later
    trial_placeholder = st.empty()

if not query_id:
    st.info("👈 Please enter a patient query ID in the sidebar to begin.")
    st.stop()

# Load data
matching_results = load_matching_results()
ranking_results = load_ranking_results(query_id)
corpus_data = load_corpus()
patient_queries = load_patient_queries()

if query_id not in matching_results and not ranking_results:
    st.error(f"❌ No results found for query ID: {query_id}")
    st.stop()

# Display patient details
if not query_id:
    st.info("👈 Enter a patient ID in the sidebar to begin")
    st.stop()

# Load patient data
try:
    patient_data = None
    with open('dataset/trec_2022/queries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('_id') == query_id:
                patient_data = data
                break
    
    if not patient_data:
        st.error(f"❌ No patient found with ID: {query_id}")
        st.stop()
    
    # Extract patient text
    patient_text = patient_data.get('text', '')
    
    # Display patient information with full width
    st.markdown("<h3 style='color: #1e4b8b;'>📋 Patient Information</h3>", unsafe_allow_html=True)
    newline = '\n'
    st.markdown(f"""
        <div class="content-card">
            {patient_text.replace(newline, '<br>')}
        </div>
    """, unsafe_allow_html=True)
            
except Exception as e:
    st.error("Unable to load patient data. Please check the patient ID and try again.")
    st.stop()

# Display ranking results if available
if ranking_results:
    # Sort trials by trial_score in descending order and take top 2
    sorted_trials = sorted(ranking_results, 
                         key=lambda x: x['scores'].get('trial_score', -100), 
                         reverse=True)[:2]  # Only take top 2 trials
    
    # Update the sidebar with trial suggestions
    with trial_placeholder.container():
        st.markdown("**Suggested Trials:**")
        for i, trial in enumerate(sorted_trials, 1):
            qrels_score = trial['scores'].get('qrels_score', 0)  # Default to 0 if not found
            status, _ = get_qrels_status(qrels_score)
            # Format status for CSS class
            status_class = status.lower().replace(' ', '-')
            st.markdown(
                f"{i}. {trial.get('nctid', 'N/A')} "
                f"<span class='status-badge status-{status_class}'>{status}</span>",
                unsafe_allow_html=True
            )
    
    if not sorted_trials:
        st.warning("No matching trials found for this patient.")
        st.stop()
        
    st.markdown("<h2 style='color: #1e4b8b; margin-top: 2rem;'>💊 Suggested Clinical Trials</h2>", unsafe_allow_html=True)
    
    for idx, trial in enumerate(sorted_trials, 1):
        scores = trial['scores']
        qrels_score = scores.get('qrels_score', 0)  # Default to 0 (Not relevant) if not found
        status, badge_color = get_qrels_status(qrels_score)
        trial_score = scores.get('trial_score', 0)  # Get the trial score for the score cards
        
        with st.container():
            # Format status for CSS class
            status_class = status.lower().replace(' ', '-')
            st.markdown(f"""
                <div class="trial-card" style="padding: 1.5rem;">
                    <div class="trial-header">
                        <div style="display: flex; align-items: center; flex-grow: 1;">
                            <div class="trial-number">{idx}</div>
                            <div class="trial-title">{trial.get('nctid', 'NCT ID not available')}</div>
                        </div>
                        <div class="status-badge status-{status_class}" style="color: {badge_color};">{status}</div>
                    </div>
            """, unsafe_allow_html=True)
            
            # Trial details in tabs with enhanced UI
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Relevance", "✅ Eligibility", "📊 Scores"])
            
            with tab1:
                summary = clean_text(trial.get('brief_summary', 'No summary available'))
                st.markdown(f"""
                    <div class="content-card">
                        <div style="font-size: 0.95rem; line-height: 1.7; color: #334155;">
                            {summary if summary else 'No summary available'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                relevance = clean_text(trial.get('relevance', 'No relevance explanation available.'))
                st.markdown(f"""
                    <div class="content-card">
                        <div style="font-size: 0.95rem; line-height: 1.7; color: #334155;">
                            {relevance if relevance else 'No relevance explanation available.'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with tab3:
                eligibility = clean_text(trial.get('eligibility', 'No eligibility explanation available.'))
                st.markdown(f"""
                    <div class="content-card">
                        <div style="font-size: 0.95rem; line-height: 1.7; color: #334155;">
                            {eligibility if eligibility else 'No eligibility explanation available.'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with tab4:
                st.markdown("""
                    <div style="font-size: 1rem; color: #FF9E00; font-weight: 600; margin-bottom: 1rem; background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 6px; border-left: 4px solid #FF9E00;">
                        📊 Detailed Scoring Metrics
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                        <div class="score-card">
                            <div class="score-label">Trial Score</div>
                            <div class="score-value">{trial_score:.2f}</div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">Overall match quality</div>
                        </div>
                        <div class="score-card">
                            <div class="score-label">Matching Score</div>
                            <div class="score-value">{matching_score:.2f}</div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">Criteria alignment</div>
                        </div>
                        <div class="score-card">
                            <div class="score-label">Aggregation Score</div>
                            <div class="score-value">{agg_score:.2f}</div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">Composite metric</div>
                        </div>
                    </div>
                """.format(
                    trial_score=scores.get('trial_score', 0),
                    matching_score=scores.get('matching_score', 0),
                    agg_score=scores.get('agg_score', 0)
                ), unsafe_allow_html=True)
                
                # Close the trial card div
                st.markdown('</div>', unsafe_allow_html=True)
                # Close the container
                st.markdown('</div>', unsafe_allow_html=True)
                    
# Display matching results if ranking is not available
elif query_id in matching_results:
    st.warning("⚠️ Full ranking not available. Showing basic matching results:")
    all_trials = {}
    for label in matching_results[query_id]:
        for nctid, trial_result in matching_results[query_id][label].items():
            all_trials[nctid] = trial_result
    
    for nctid, trial_result in all_trials.items():
        with st.expander(f"Trial: {nctid}", expanded=False):
            if isinstance(trial_result, dict):
                for crit_type in ["inclusion", "exclusion"]:
                    if crit_type in trial_result:
                        st.markdown(f"**{crit_type.capitalize()}**")
                        for k, v in trial_result[crit_type].items():
                            st.markdown(f"- **Criterion {k}:** {v[0]}")
                            st.markdown(f"  - **Sentences:** {v[1]}")
                            st.markdown(f"  - **Status:** {v[2]}")
            trial_info = get_trial_by_nctid(nctid, corpus_data)
            st.markdown("**Trial Info:**")
            st.json(trial_info)

# Add some custom CSS
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
        }
        .stButton>button:hover {
            background-color: #3a5a80;
            color: white;
        }
        .stMarkdown h3 {
            color: #2c3e50;
        }
        .stMarkdown h2 {
            color: #2c3e50;
            border-bottom: 2px solid #4a6fa5;
            padding-bottom: 0.3em;
        }
    </style>
""", unsafe_allow_html=True)
