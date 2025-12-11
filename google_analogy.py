import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
VECTORS_FILE = "checkpoints\debiased_vectors.txt"
ANALOGY_FILE = "questions-words.txt"  # Download this file first
TOP_K = 1  # We usually check if the exact word is #1

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_vectors(file_path):
    print(f"Loading vectors from {file_path}...")
    vectors = {}
    words_list = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header if present
        first_line = f.readline().split()
        if len(first_line) == 2:
            print(f"Header found. Vocab: {first_line[0]}, Dim: {first_line[1]}")
        else:
            f.seek(0)
            
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            
            # Normalize immediately for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
                
            vectors[word] = vec
            words_list.append(word)
            
    print(f"Loaded {len(vectors)} vectors.")
    # Return dictionary and matrix for fast operations
    vector_matrix = np.array([vectors[w] for w in words_list])
    return vectors, words_list, vector_matrix

# ==========================================
# 3. ANALOGY EVALUATION
# ==========================================
def evaluate_analogies(vectors, idx2word, vector_matrix, analogy_path):
    if not os.path.exists(analogy_path):
        print(f"Error: {analogy_path} not found.")
        print("Download it: https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt")
        return

    word2idx = {w: i for i, w in enumerate(idx2word)}
    
    total = 0
    correct = 0
    sections = {}
    current_section = "Unknown"
    
    print("\nStarting Analogy Evaluation...")
    
    with open(analogy_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Detect section headers (e.g., : capital-common-countries)
            if line.startswith(':'):
                current_section = line[1:].strip()
                sections[current_section] = {'total': 0, 'correct': 0}
                print(f"--> Processing section: {current_section}")
                continue
                
            # Parse analogy: a b c d (a:b :: c:d)
            # Example: Athens Greece Baghdad Iraq
            parts = line.lower().split()
            if len(parts) != 4: continue
            
            a, b, c, d = parts
            
            # Skip if any word is missing from vocab
            if a not in vectors or b not in vectors or c not in vectors or d not in vectors:
                continue
            
            # Vector Arithmetic: target = b - a + c
            # (Greece - Athens + Baghdad = Iraq)
            vec_a = vectors[a]
            vec_b = vectors[b]
            vec_c = vectors[c]
            
            target_vec = vec_b - vec_a + vec_c
            
            # Normalize target
            norm = np.linalg.norm(target_vec)
            if norm > 0:
                target_vec /= norm
                
            # Find closest vectors (Matrix multiplication for speed)
            # Scores = dot product of (VocabMatrix . Target)
            scores = np.dot(vector_matrix, target_vec)
            
            # Get Top K indices
            # We want to ignore a, b, and c in the results (standard protocol)
            sorted_indices = np.argsort(-scores)
            
            found_word = None
            for idx in sorted_indices[:4]: # Check top 4 candidates
                candidate = idx2word[idx]
                if candidate not in [a, b, c]:
                    found_word = candidate
                    break
            
            # Check accuracy
            is_correct = (found_word == d)
            
            total += 1
            sections[current_section]['total'] += 1
            if is_correct:
                correct += 1
                sections[current_section]['correct'] += 1

    # ==========================================
    # 4. REPORTING
    # ==========================================
    print("\n" + "="*40)
    print("RESULTS BY SECTION")
    print("="*40)
    
    syntactic_correct = 0
    syntactic_total = 0
    semantic_correct = 0
    semantic_total = 0
    
    for section, stats in sections.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"{section:<30}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
            
            # Rough categorization of standard Google sections
            if 'gram' in section: # gram1-adjective-to-adverb, etc.
                syntactic_correct += stats['correct']
                syntactic_total += stats['total']
            else:
                semantic_correct += stats['correct']
                semantic_total += stats['total']

    print("="*40)
    if semantic_total > 0:
        print(f"Semantic Accuracy: {semantic_correct/semantic_total*100:.2f}%")
    if syntactic_total > 0:
        print(f"Syntactic Accuracy: {syntactic_correct/syntactic_total*100:.2f}%")
    
    if total > 0:
        print(f"\nOVERALL ACCURACY: {correct/total*100:.2f}% ({correct}/{total})")
    else:
        print("No analogies could be evaluated (Words missing from vocab).")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        vecs, words, mat = load_vectors(VECTORS_FILE)
        evaluate_analogies(vecs, words, mat, ANALOGY_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")