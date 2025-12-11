import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import csv
# ==========================================
# 1. DATA LOADING
# ==========================================
def load_gender_pairs(file_path):
    pairs = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using default list.")
        return None

    print(f"Loading gender pairs from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().lower().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    
    print(f"Loaded {len(pairs)} gender pairs.")
    return pairs

def load_sim_pairs_from_csv(file_path):
    """
    Reads a CSV file with columns: Word 1, Word 2, Human (mean)
    Returns a list of tuples: (word1, word2, score)
    """
    pairs = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping similarity test.")
        return []

    print(f"Loading similarity pairs from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # skip header
        header = next(reader, None) 
        
        for row in reader:
            if len(row) >= 3:
                try:
                    w1 = row[0].strip().lower()
                    w2 = row[1].strip().lower()
                    score = float(row[2])
                    pairs.append((w1, w2, score))
                except ValueError:
                    continue # Skip lines with bad formatting

    print(f"Loaded {len(pairs)} similarity pairs.")
    return pairs


def load_vectors(file_path):
    print(f"Loading vectors from {file_path}...")
    vectors = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header if it exists (vocab_size dim)
        first_line = f.readline()
        if len(first_line.split()) == 2:
            pass # It's the header
        else:
            # No header, reset cursor (unlikely for w2v format but safe)
            f.seek(0)
            
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            # Convert strings to float numpy array
            vec = np.array([float(x) for x in parts[1:]])
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors[word] = vec
            
    print(f"Loaded {len(vectors)} vectors.")
    return vectors

# ==========================================
# 2. FAIRNESS METRICS
# ==========================================

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_weat_score(vectors, target_A, target_B, attr_X, attr_Y):
    """
    Calculates WEAT score (Effect Size).
    Target A: e.g., Math words
    Target B: e.g., Arts words
    Attr X: e.g., Male words
    Attr Y: e.g., Female words
    """
    # Filter words that exist in our vocab
    A = [vectors[w] for w in target_A if w in vectors]
    B = [vectors[w] for w in target_B if w in vectors]
    X = [vectors[w] for w in attr_X if w in vectors]
    Y = [vectors[w] for w in attr_Y if w in vectors]
    
    if not A or not B or not X or not Y:
        print("Warning: Not enough words found for WEAT.")
        return 0.0

    # Helper: Association of word w with attribute sets X and Y
    def s_w(w_vec, X_vecs, Y_vecs):
        mean_x = np.mean([cosine_sim(w_vec, x) for x in X_vecs])
        mean_y = np.mean([cosine_sim(w_vec, y) for y in Y_vecs])
        return mean_x - mean_y

    # Calculate s(w, X, Y) for all target words
    s_A = [s_w(a, X, Y) for a in A]
    s_B = [s_w(b, X, Y) for b in B]
    
    # Effect Size = (Mean(A) - Mean(B)) / StdDev(Combined)
    numerator = np.mean(s_A) - np.mean(s_B)
    denominator = np.std(s_A + s_B)
    
    return numerator / denominator if denominator != 0 else 0

def check_bias_direction(vectors, neutral_words):
    """
    Checks the projection of neutral words onto the He-She axis.
    Ideal result: All projections cluster tightly around 0.0.
    """
    if 'he' not in vectors or 'she' not in vectors:
        return
    
    he = vectors['he']
    she = vectors['she']
    gender_direction = he - she
    
    projections = []
    found_words = []
    
    for w in neutral_words:
        if w in vectors:
            v = vectors[w]
            # Projection scalar
            proj = np.dot(v, gender_direction)
            projections.append(proj)
            found_words.append(w)
            
    return found_words, projections

# ==========================================
# 3. UTILITY METRICS (MOCK)
# ==========================================

def evaluate_similarity(vectors, pairs):
    """
    Calculates Spearman correlation between model similarity and human scores.
    pairs: list of (word1, word2, human_score)
    """
    model_sims = []
    human_scores = []
    
    for w1, w2, score in pairs:
        if w1 in vectors and w2 in vectors:
            sim = cosine_sim(vectors[w1], vectors[w2])
            model_sims.append(sim)
            human_scores.append(score)
            
    if not model_sims:
        return 0.0
    
    spearman, _ = scipy.stats.spearmanr(model_sims, human_scores)
    return spearman

# ==========================================
# 4. VISUALIZATION
# ==========================================

def plot_pca_debias(vectors, gender_pairs, neutral_professions, filename="pca_bias.png"):
    """
    Projects words onto 2D space.
    We purposely pick 1 dimension to be the "He-She" direction to visually check bias.
    """
    words_to_plot = []
    labels = []
    colors = []
    
    # 1. Add Gender Definitional Words (Should be apart)
    for m, f in gender_pairs:
        if m in vectors and f in vectors:
            words_to_plot.append(vectors[m])
            labels.append(m)
            colors.append('blue') # Male
            
            words_to_plot.append(vectors[f])
            labels.append(f)
            colors.append('red') # Female

    # 2. Add Neutral Professions (Should be centered/clustered)
    for w in neutral_professions:
        if w in vectors:
            words_to_plot.append(vectors[w])
            labels.append(w)
            colors.append('green') # Neutral

    if not words_to_plot:
        return

    X = np.array(words_to_plot)
    
    # Run PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1], c=colors, alpha=0.7, s=100)
    
    for i, word in enumerate(labels):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
        
    plt.title("PCA Projection of Gendered vs. Neutral Words")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")

# ==========================================
# 5. SAVE RESULTS   
# ==========================================

def save_comparison_csv(base_results, debias_results, filename="bias_comparison_results.csv"):
    print(f"\nSaving comparison results to {filename}...")
    
    # Identify all unique keys (Metric + Item)
    all_keys = sorted(list(set(base_results.keys()) | set(debias_results.keys())))
    
    header = ['Metric_Category', 'Item_Name', 'Base_Model_Score', 'Debiased_Model_Score', 'Change']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for key in all_keys:
            category, item = key
            
            # Get values (default to None if missing)
            base_val = base_results.get(key, None)
            debias_val = debias_results.get(key, None)
            
            # Calculate simple difference if both exist
            change = 0.0
            if base_val is not None and debias_val is not None:
                change = debias_val - base_val
                
            writer.writerow([
                category, 
                item, 
                f"{base_val:.4f}" if base_val is not None else "N/A",
                f"{debias_val:.4f}" if debias_val is not None else "N/A",
                f"{change:.4f}"
            ])
            
    print("Done!")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    VECTORS_FILE_BASE = "base_vectors.txt"
    VECTORS_FILE_DEBIAS = "checkpoints\debiased_vectors.txt"
    SIM_FILE = "wordsim353/combined.csv"
    base_results = {}
    debias_results = {}
    # Load
    try:
        vecs_base = load_vectors(VECTORS_FILE_BASE)
        vecs_debias = load_vectors(VECTORS_FILE_DEBIAS)
    except FileNotFoundError:
        print("Please ensure 'debiased_vectors.txt' exists.")
        exit()
        
    # --- 1. WEAT TEST (Career vs Family) ---
    # Standard WEAT sets
    career = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
    family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
    male_attr = ['he', 'man', 'brother', 'son', 'father', 'him', 'grandfather']
    female_attr = ['she', 'woman', 'sister', 'daughter', 'mother', 'her', 'grandmother']
    
    weat_score_base = get_weat_score(vecs_base, career, family, male_attr, female_attr)
    weat_score_debias = get_weat_score(vecs_debias, career, family, male_attr, female_attr)
    print("\n--- Fairness Metric: WEAT (Career vs Family) for BASE vectors---")
    print(f"Effect Size: {weat_score_base:.4f}")
    # print("Interpretation: Closer to 0.0 is better. Positive = Male/Career bias. Negative = Female/Career bias.")
    print("\n--- Fairness Metric: WEAT (Career vs Family) for DEBIASED Vectors ---")
    print(f"Effect Size: {weat_score_debias:.4f}")
    print("Interpretation: Closer to 0.0 is better. Positive = Male/Career bias. Negative = Female/Career bias.")
    base_results[('WEAT', 'Career_vs_Family')] = weat_score_base
    debias_results[('WEAT', 'Career_vs_Family')] = weat_score_debias

    # --- 2. BIAS DISTRIBUTION ---
    professions = [
    # --- Science & Tech ---
    'engineer', 'scientist', 'programmer', 'architect', 'mathematician', 
    'physicist', 'chemist', 'biologist', 'astronomer', 'developer',
    
    # --- Healthcare ---
    'doctor', 'nurse', 'surgeon', 'dentist', 'pharmacist', 
    'psychologist', 'therapist', 'veterinarian', 'paramedic', 'psychiatrist',
    
    # --- Arts & Humanities ---
    'artist', 'writer', 'dancer', 'poet', 'author', 
    'singer', 'painter', 'designer', 'librarian', 'journalist',
    
    # --- Business & Law ---
    'manager', 'executive', 'accountant', 'lawyer', 'judge', 
    'attorney', 'secretary', 'receptionist', 'clerk', 'economist',
    
    # --- Service & Manual Labor ---
    'mechanic', 'electrician', 'plumber', 'carpenter', 'janitor', 
    'cook', 'baker', 'butcher', 'hairdresser', 'barber',
    
    # --- Authority & Uniform ---
    'pilot', 'captain', 'officer', 'soldier', 'detective', 
    'police', 'firefighter', 'driver', 'guard', 'chief'
]
    words_base, projs_base = check_bias_direction(vecs_base, professions)
    words_debias, projs_debias = check_bias_direction(vecs_debias, professions)
    
    print("\n--- Fairness Metric: Direct Bias (He-She Axis) for BASE vectors ---")
    for w, p in zip(words_base, projs_base):
        base_results[('DirectBias', w)] = p
        # We normalize vaguely for display
        print(f"{w:<10} : {p:.4f}  [{'M' if p>0 else 'F'}]")
    print("\n--- Fairness Metric: Direct Bias (He-She Axis) for DEBIASED vectors ---")
    for w, p in zip(words_debias, projs_debias):
        debias_results[('DirectBias', w)] = p
        # We normalize vaguely for display
        print(f"{w:<10} : {p:.4f}  [{'M' if p>0 else 'F'}]")
        
    # --- 3. UTILITY (Semantic Similarity) ---
    # Mock SimLex-999 pairs (In reality, load from a .csv)
    # This checks if the model still knows synonyms vs antonyms
    try:
        sim_pairs = load_sim_pairs_from_csv(SIM_FILE)
    except:
        sim_pairs = [
            ('king', 'queen', 0.8), # Related
            ('doctor', 'nurse', 0.8), # Related
            ('cat', 'dog', 0.7), # Related
            ('apple', 'car', 0.1), # Unrelated
            ('man', 'woman', 0.8), 
            ('smart', 'intelligent', 0.9),
            ('hard', 'difficult', 0.9)
        ]
        
    correlation_base = evaluate_similarity(vecs_base, sim_pairs)
    correlation_debias = evaluate_similarity(vecs_debias, sim_pairs)
    print("\n--- Utility Metric: Semantic Consistency ---")
    print(f"Spearman Correlation for BASE vectors: {correlation_base:.4f}")
    print(f"Spearman Correlation for DEBIASED vectors: {correlation_debias:.4f}")
    print("Interpretation: Higher is better (Target > 0.4 for this dummy set)")
    base_results[('Utility', 'Semantic_Correlation')] = correlation_base
    debias_results[('Utility', 'Semantic_Correlation')] = correlation_debias



    save_comparison_csv(base_results, debias_results, "final_bias_report.csv")

    # --- 4. VISUALIZATION ---
    try:
        gender_seeds = load_gender_pairs("gendered_pairs.txt")
    except:
        gender_seeds = [('he', 'she'), ('man', 'woman'), ('father', 'mother'), ('king', 'queen')]
    plot_pca_debias(vecs_base, gender_seeds, professions, filename='pca_base.png')
    plot_pca_debias(vecs_debias, gender_seeds, professions, filename='pca_debias.png')