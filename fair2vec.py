import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import Counter
import re
import random
import os

# ==========================================
# 1. PREPROCESSING & VOCABULARY
# ==========================================

def get_file_iterator(file_path, max_sentences=None):
    """
    Generator that yields lines from a file.
    Args:
        file_path: Path to the corpus text file.
        max_sentences: Integer limit. If None, reads the whole file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find corpus file: {file_path}")
        
    print(f"Reading corpus from: {file_path} (Limit: {max_sentences if max_sentences else 'None'})")
    
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
                count += 1
                if max_sentences is not None and count >= max_sentences:
                    print(f"Reached limit of {max_sentences} sentences.")
                    break

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

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def export_embeddings(model, idx2word, filename="vectors.txt"):
    print(f"=> Exporting embeddings to {filename}")
    model.eval()
    with torch.no_grad():
        weights = model.in_embed.weight.cpu().numpy()
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(idx2word)} {weights.shape[1]}\n")
        for idx, word in idx2word.items():
            vec = weights[idx]
            vec_str = " ".join([f"{v:.6f}" for v in vec])
            f.write(f"{word} {vec_str}\n")

class TrainingMonitor:
    def __init__(self, log_dir="runs/debias_experiment"):
        self.writer = SummaryWriter(log_dir)
        
    def log_losses(self, epoch, sem_loss, adv_loss, debias_loss):
        self.writer.add_scalar('Loss/Semantic', sem_loss, epoch)
        self.writer.add_scalar('Loss/Adversary', adv_loss, epoch)
        self.writer.add_scalar('Loss/Debias', debias_loss, epoch)
        
    def check_bias(self, model, word2idx, device, epoch, test_words=['doctor', 'nurse', 'engineer', 'homemaker']):
        model.eval()
        if 'he' not in word2idx or 'she' not in word2idx:
            return

        he_idx = torch.tensor([word2idx['he']], device=device)
        she_idx = torch.tensor([word2idx['she']], device=device)
        
        with torch.no_grad():
            he_vec = model.get_embedding(he_idx)
            she_vec = model.get_embedding(she_idx)
            
            for word in test_words:
                if word in word2idx:
                    w_idx = torch.tensor([word2idx[word]], device=device)
                    w_vec = model.get_embedding(w_idx)
                    
                    sim_he = F.cosine_similarity(w_vec, he_vec).item()
                    sim_she = F.cosine_similarity(w_vec, she_vec).item()
                    
                    bias_score = sim_he - sim_she
                    
                    self.writer.add_scalar(f'Bias_Score/{word}', bias_score, epoch)
                    print(f"   [Probe] {word}: Bias Score = {bias_score:.4f} (Target: 0.0)")

    def close(self):
        self.writer.close()

def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-\s]', '', text) 
    return text.split()

def build_vocabulary(corpus_iterator, min_freq=5, max_vocab_size=150000):
    print("Building vocabulary...")
    token_counts = Counter()
    for text_chunk in corpus_iterator:
        tokens = simple_tokenizer(text_chunk)
        token_counts.update(tokens)
    
    vocab_freqs = token_counts.most_common(max_vocab_size)
    vocab_freqs = [pair for pair in vocab_freqs if pair[1] >= min_freq]
    
    word2idx = {word: idx for idx, (word, count) in enumerate(vocab_freqs)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Vocab size: {len(word2idx)}")
    return word2idx, idx2word, vocab_freqs

def get_subsampling_weights(vocab_freqs, threshold=1e-5):
    total_count = sum(count for word, count in vocab_freqs)
    keep_probs = {}
    for idx, (word, count) in enumerate(vocab_freqs):
        freq_fraction = count / total_count
        p_keep = (np.sqrt(freq_fraction / threshold) + 1) * (threshold / freq_fraction)
        keep_probs[idx] = min(1.0, p_keep)
    return keep_probs

# ==========================================
# 2. DATA LOADERS (UPDATED TO ITERABLE)
# ==========================================

# --- NEW: Streaming Dataset Class ---
class StreamingWord2VecDataset(IterableDataset):
    def __init__(self, file_path, word2idx, window_size=3, subsampling_probs=None, max_sentences=None):
        self.file_path = file_path
        self.word2idx = word2idx
        self.window_size = window_size
        self.subsampling_probs = subsampling_probs
        self.max_sentences = max_sentences

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        count = 0
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if worker_info is not None:
                # Multi-processing: Split work among workers
                # Worker 0 gets lines 0, 4, 8... 
                # Worker 1 gets lines 1, 5, 9...
                for i, line in enumerate(f):
                    if i % worker_info.num_workers == worker_info.id:
                        yield from self.process_line(line)
                        count += 1
                        if self.max_sentences and count >= self.max_sentences // worker_info.num_workers:
                            break
            else:
                # Single-process: Read everything
                for line in f:
                    yield from self.process_line(line)
                    count += 1
                    if self.max_sentences and count >= self.max_sentences:
                        break

    def process_line(self, line):
        tokens = simple_tokenizer(line)
        token_ids = [self.word2idx[w] for w in tokens if w in self.word2idx]
        
        # Subsampling
        if self.subsampling_probs:
            token_ids = [t for t in token_ids if random.random() < self.subsampling_probs[t]]
            
        if len(token_ids) < 2:
            return

        for i, target in enumerate(token_ids):
            start = max(0, i - self.window_size)
            end = min(len(token_ids), i + self.window_size + 1)
            context_words = token_ids[start:i] + token_ids[i+1:end]
            
            for context in context_words:
                yield torch.tensor(target), torch.tensor(context)

# Old Map-style dataset kept just in case you fall back to dummy data
class Word2VecDataset(Dataset):
    def __init__(self, corpus, word2idx, window_size=3, subsampling_probs=None):
        self.pairs = []
        for text in corpus:
            tokens = simple_tokenizer(text)
            token_ids = [word2idx[w] for w in tokens if w in word2idx]
            if subsampling_probs:
                token_ids = [tid for tid in token_ids if random.random() < subsampling_probs[tid]]
            for i, target in enumerate(token_ids):
                start = max(0, i - window_size)
                end = min(len(token_ids), i + window_size + 1)
                context_words = token_ids[start:i] + token_ids[i+1:end]
                for context in context_words:
                    self.pairs.append((target, context))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return torch.tensor(target), torch.tensor(context)

class GenderDataset(Dataset):
    def __init__(self, gender_pairs, word2idx):
        self.samples = []
        for m_word, f_word in gender_pairs:
            if m_word in word2idx and f_word in word2idx:
                self.samples.append((word2idx[m_word], 1)) 
                self.samples.append((word2idx[f_word], 0)) 
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_infinite_gender_loader(gender_pairs, word2idx, batch_size=4):
    dataset = GenderDataset(gender_pairs, word2idx)
    if len(dataset) == 0:
        raise ValueError("No gender pairs found in vocabulary!")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    while True:
        for batch in dataloader:
            yield batch

class NeutralSampler:
    def __init__(self, word2idx, all_gender_words, vocab_freqs, n_words_to_debias=20000):
        self.neutral_ids = []
        for word, _ in vocab_freqs[:n_words_to_debias]:
            if word not in all_gender_words:
                self.neutral_ids.append(word2idx[word])
        self.neutral_ids = torch.LongTensor(self.neutral_ids)
        self.n_samples = len(self.neutral_ids)

    def get_batch(self, batch_size, device):
        indices = torch.randint(0, self.n_samples, (batch_size,), device=device)
        return self.neutral_ids.to(device)[indices]

# ==========================================
# 3. MODELS
# ==========================================

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2VecModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim) 
        self.out_embed = nn.Embedding(vocab_size, embed_dim) 
        nn.init.uniform_(self.in_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.out_embed.weight, -0.1, 0.1)

    def forward(self, target_ids, context_ids):
        in_vecs = self.in_embed(target_ids)
        out_vecs = self.out_embed(context_ids)
        return torch.sum(in_vecs * out_vecs, dim=1)
    
    def get_embedding(self, word_ids):
        return self.in_embed(word_ids)

class GenderAdversary(nn.Module):
    def __init__(self, embed_dim):
        super(GenderAdversary, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, embeddings):
        return self.net(embeddings)

# ==========================================
# 4. TRAINING LOGIC 
# ==========================================

def train_step_with_refresh(w2v_model, adversary, opt_w2v, opt_adv, 
                            w2v_batch, gender_loader, neutral_batch, 
                            vocab_size, device, lambda_param=0.8, n_adversary_repeats=3):
    
    target_ids, context_ids = w2v_batch
    target_ids = target_ids.to(device)
    context_ids = context_ids.to(device)
    
    neg_ids = torch.randint(0, vocab_size, target_ids.shape, device=device)

    # --- PHASE 1: Semantic Update ---
    opt_w2v.zero_grad()
    
    pos_score = w2v_model(target_ids, context_ids)
    pos_loss = -torch.mean(F.logsigmoid(pos_score))
    
    neg_score = w2v_model(target_ids, neg_ids)
    neg_loss = -torch.mean(F.logsigmoid(-neg_score))
    
    semantic_loss = pos_loss + neg_loss
    semantic_loss.backward() 
    
    if lambda_param == 0:
        opt_w2v.step()
        return semantic_loss.item(), 0.0, 0.0
    
    # --- PHASE 2: Train Adversary ---
    total_adv_loss = 0
    for _ in range(n_adversary_repeats):
        try:
            g_words, g_labels = next(gender_loader)
        except StopIteration:
            break
        
        g_words = g_words.to(device)
        g_labels = g_labels.to(device)
            
        opt_adv.zero_grad()
        
        g_embeddings = w2v_model.get_embedding(g_words).detach()
        g_preds = adversary(g_embeddings)
        adv_loss = F.binary_cross_entropy_with_logits(g_preds, g_labels.float().unsqueeze(1))
        
        adv_loss.backward()
        opt_adv.step()
        total_adv_loss += adv_loss.item()

    # --- PHASE 3: Debiasing ---
    neutral_embeddings = w2v_model.get_embedding(neutral_batch)
    neutral_logits = adversary(neutral_embeddings)
    debias_loss = torch.mean(neutral_logits ** 2)
    
    (lambda_param * debias_loss).backward()
    opt_w2v.step()
    
    return semantic_loss.item(), (total_adv_loss/n_adversary_repeats), debias_loss.item()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    EMBED_DIM = 128
    BATCH_SIZE = 8192*2
    LAMBDA = 1.0 
    CORPUS_FILE = "wikipedia_en_20231101.txt"
    GENDER_FILE = "gendered_pairs.txt"
    MAX_SENTENCES = 2000000
    
    # 1. BUILD VOCABULARY (Linear Pass)
    if os.path.exists(CORPUS_FILE):
        vocab_iter = get_file_iterator(CORPUS_FILE, max_sentences=MAX_SENTENCES)
        word2idx, idx2word, vocab_freqs = build_vocabulary(vocab_iter, min_freq=5)
        
        # 2. SETUP STREAMING DATASET
        # NOTE: We do NOT use 'get_file_iterator' here. The dataset class handles file opening itself.
        w2v_dataset = StreamingWord2VecDataset(
            CORPUS_FILE, 
            word2idx, 
            window_size=2,
            subsampling_probs=get_subsampling_weights(vocab_freqs),
            max_sentences=MAX_SENTENCES
        )
        
        # 3. SETUP DATALOADER
        # num_workers=4 is safe now because we are streaming (no RAM explosion)
        # shuffle=False is required for IterableDataset (shuffling happens in buffer if impl, or sequential)
        w2v_loader = DataLoader(
            w2v_dataset, 
            batch_size=BATCH_SIZE, 
            pin_memory=True, 
            num_workers=4,          
            persistent_workers=True,
            prefetch_factor=4
        )
        
    else:
        print(f"Warning: '{CORPUS_FILE}' not found. Using Synthetic Dummy Data.")
        dummy_corpus = [
            "the king is a man", "the queen is a woman",
            "he is a brother", "she is a sister",
            "the doctor is a genius", "the nurse is a helper", 
            "the engineer fixed the car", "the receptionist answered the phone",
            "he is strong", "she is beautiful",
            "father is a parent", "mother is a parent"
        ] * 500
        word2idx, idx2word, vocab_freqs = build_vocabulary(dummy_corpus, min_freq=1)
        # Fallback to map-style for dummy data
        w2v_dataset = Word2VecDataset(dummy_corpus, word2idx, subsampling_probs=get_subsampling_weights(vocab_freqs))
        w2v_loader = DataLoader(w2v_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    # Setup Aux Data
    loaded_pairs = load_gender_pairs(GENDER_FILE)
    if loaded_pairs:
        gender_pairs = loaded_pairs
    else:
        gender_pairs = [('man','woman'), ('he','she'), ('king','queen'), ('brother','sister'), ('father','mother')]

    all_gender_words = set([w for p in gender_pairs for w in p])
    
    gender_loader_iter = get_infinite_gender_loader(gender_pairs, word2idx, batch_size=256)
    neutral_sampler = NeutralSampler(word2idx, all_gender_words, vocab_freqs)
    
    # Init Models
    w2v_model = Word2VecModel(len(word2idx), EMBED_DIM).to(device)
    adversary = GenderAdversary(EMBED_DIM).to(device)
    
    opt_w2v = optim.Adam(w2v_model.parameters(), lr=0.005)
    opt_adv = optim.Adam(adversary.parameters(), lr=0.001)
    
    # This reduces the LR by 50% every 15 epochs
    scheduler_w2v = optim.lr_scheduler.StepLR(opt_w2v, step_size=15, gamma=0.5)
    scheduler_adv = optim.lr_scheduler.StepLR(opt_adv, step_size=15, gamma=0.5)

    monitor = TrainingMonitor(log_dir="runs/w2v_debias_v2")
    
    print("\nStarting GPU Training with Monitoring...")
    
    EPOCHS = 75
    WARMUP_EPOCHS = 10

    for epoch in range(EPOCHS):
        w2v_model.train()
        adversary.train()
        
        if epoch < WARMUP_EPOCHS:
            current_lambda = 0.0
            print(f"--> Warm-up Phase (Epoch {epoch+1}): Semantic Training Only")
        else:
            current_lambda = LAMBDA
            print(f"--> Debiasing Phase (Epoch {epoch+1}): Lambda = {current_lambda}")

        total_sem = 0
        total_adv = 0
        total_debias = 0
        batch_count = 0  # <--- CRITICAL: Manually count batches for IterableDataset
        
        for i, batch in enumerate(w2v_loader):
            neutral_batch = neutral_sampler.get_batch(len(batch[0]), device)
            
            s_loss, a_loss, d_loss = train_step_with_refresh(
                w2v_model, adversary, opt_w2v, opt_adv, 
                batch, gender_loader_iter, neutral_batch, 
                len(word2idx), device, lambda_param=current_lambda
            )
            
            total_sem += s_loss
            total_adv += a_loss
            total_debias += d_loss
            batch_count += 1
        
        # Avoid division by zero if dataset is empty
        if batch_count == 0: batch_count = 1
            
        avg_sem = total_sem / batch_count
        avg_adv = total_adv / batch_count
        avg_debias = total_debias / batch_count

        # LR decay
        scheduler_w2v.step()
        scheduler_adv.step()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"   Losses: Semantic={avg_sem:.4f} | Adv={avg_adv:.4f} | Debias={avg_debias:.4f}")
        
        monitor.log_losses(epoch, avg_sem, avg_adv, avg_debias)
        monitor.check_bias(w2v_model, word2idx, device, epoch)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': w2v_model.state_dict(),
            'optimizer': opt_w2v.state_dict(),
            'adversary': adversary.state_dict(),
        }, filename=f"checkpoint_fair2vec.pth.tar")

    export_embeddings(w2v_model, idx2word, filename="debiased_vectors.txt")
    monitor.close()
    print("\nTraining Complete. To view logs run: tensorboard --logdir=runs")

    print("\n--- Evaluation ---")
    test_words = ['doctor', 'nurse', 'king', 'queen']
    for w in test_words:
        if w in word2idx:
            idx = torch.tensor([word2idx[w]], device=device)
            emb = w2v_model.get_embedding(idx)
            gender_prob = torch.sigmoid(adversary(emb)).item()
            print(f"Word: {w:10} | Gender Prob (1=Male, 0=Female): {gender_prob:.4f}")