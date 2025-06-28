# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import esm
import os
import random
from sklearn.ensemble import RandomForestRegressor
from Bio.PDB import PDBParser, Selection
import re
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import logging

warnings.filterwarnings('ignore')

# --- Configure logging ---
logging.basicConfig(
    filename='gfp_design.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Constants ---
DATA_DIR = r"C:\Users\28333\Desktop\proteindesign\.venv\2025Protein Design"
TRAIN_DATA_FILE = r"C:\Users\28333\Desktop\proteindesign\.venv\2025Protein Design\GFP_data.xlsx"
WT_SEQ_FILE = r"C:\Users\28333\Desktop\proteindesign\.venv\2025Protein Design\AAseqs of 4 GFP proteins.txt"
EXCLUSION_FILE = r"C:\Users\28333\Desktop\proteindesign\.venv\2025Protein Design\Exclusion_List.csv"
PDB_FILE = r"C:\Users\28333\Desktop\proteindesign\.venv\2025Protein Design\GFP Protein structures\4 kinds of GFP Protein structures\avGFP__2wur.pdb"

# Model parameters
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
MAX_MUTATIONS = 6
N_CANDIDATES = 1000
TOP_N = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Self-attention parameters
NUM_HEADS = 4
EMBED_DIM = 640

# Candidate position pool
CANDIDATE_POSITION_POOL = [
    7,10,11,24,30,39,44,50,51,61,64,65,66,70,71,72,86,99,101,105,109,128,
    145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
    176,195,203,205,212,221,222,226,231,232,233,235
]
CANDIDATE_POSITION_POOL_0BASED = [p - 1 for p in CANDIDATE_POSITION_POOL]

BENEFICIAL_MUTATIONS = [
    'F64L', 'S65T', 'Y66H','Y66W','Y145F','M153T', 'V163A', 'I167T', 'N149K',
    'C48S', 'F99S', 'A206K', 'T203Y', 'E222Q', 'S72A', 'Q69L','C70S'
]

# --- Initialize random seeds ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Resource monitoring ---
def print_resource_usage():
    msg = f"Running on {DEVICE}"
    logging.info(msg)
    print(msg)


# --- Data loading and preprocessing ---
def load_data():
    logging.info("Starting data loading...")
    try:
        df = pd.read_excel(TRAIN_DATA_FILE)
        logging.info(f"Loaded Excel file, data size: {len(df)}")
        logging.info(f"Data columns: {list(df.columns)}")
        print(f"Loaded Excel file, data size: {len(df)}")
        print(f"Data columns: {list(df.columns)}")

        if not pd.api.types.is_numeric_dtype(df['Brightness']):
            raise ValueError("Brightness column contains non-numeric values")
        if df['aaMutations'].isna().all():
            raise ValueError("aaMutations column has no valid mutation data")

        wt_seq = get_wildtype_sequence(WT_SEQ_FILE)
        logging.info(f"Loaded wild-type sequence, length: {len(wt_seq)}")
        print(f"Loaded wild-type sequence, length: {len(wt_seq)}")

        exclusion_df = pd.read_csv(EXCLUSION_FILE)
        possible_columns = ['sequences-not-submit', 'sequence', 'Sequence']
        seq_col = None
        for col in exclusion_df.columns:
            match, score = process.extractOne(col, possible_columns)
            if score > 80:
                seq_col = col
                break
        if seq_col:
            exclusion = set(exclusion_df[seq_col].astype(str))
            logging.info(f"Exclusion list contains {len(exclusion)} sequences, sample: {list(exclusion)[:5]}")
            print(f"Exclusion list contains {len(exclusion)} sequences, sample: {list(exclusion)[:5]}")
        else:
            raise ValueError(f"No sequence column found, available columns: {exclusion_df.columns}")

        return df, wt_seq, exclusion
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        raise


def get_wildtype_sequence(file_path):
    """Extract avGFP wild-type sequence, ensuring length 238 and valid format"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'avGFP' in line and i + 1 < len(lines):
                    seq = lines[i + 1].strip()
                    if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq) and len(seq) == 238:
                        logging.info("Successfully parsed avGFP sequence")
                        return seq
                    else:
                        raise ValueError(
                            f"Invalid avGFP sequence: length {len(seq)} != 238 or contains invalid characters")
            raise ValueError("avGFP wild-type sequence not found")
    except FileNotFoundError:
        logging.error(f"File {file_path} not found")
        raise ValueError(f"File {file_path} not found")
    except Exception as e:
        logging.error(f"Failed to parse {file_path}: {str(e)}")
        raise ValueError(f"Failed to parse {file_path}: {str(e)}")


# --- Position and mutation mining ---
def extract_high_score_positions(df, brightness_threshold=3.0):
    """Extract mutation positions from high-brightness sequences"""
    positions = set()
    for muts in df[df['Brightness'] > brightness_threshold]['aaMutations'].dropna():
        if pd.isna(muts) or not isinstance(muts, str):
            continue
        for mut in muts.split(':'):
            if re.match(r'[A-Z]\d+[A-Z]', mut):
                try:
                    pos = int(mut[1:-1])
                    if 1 <= pos <= 238:
                        positions.add(pos)
                    else:
                        logging.warning(f"Mutation {mut} position {pos} out of range (1-238)")
                except ValueError:
                    logging.warning(f"Invalid mutation format: {mut}")
                    continue
    return sorted(list(positions))


def extract_beneficial_mutations(df, brightness_threshold=3.0):
    """Extract beneficial mutations from high-brightness sequences"""
    mutations = set()
    for muts in df[df['Brightness'] > brightness_threshold]['aaMutations'].dropna():
        if pd.isna(muts) or not isinstance(muts, str):
            continue
        for mut in muts.split(':'):
            if re.match(r'[A-Z]\d+[A-Z]', mut):
                try:
                    pos = int(mut[1:-1])
                    if 1 <= pos <= 238:
                        mutations.add(mut)
                    else:
                        logging.warning(f"Mutation {mut} position {pos} out of range (1-238)")
                except ValueError:
                    logging.warning(f"Invalid mutation format: {mut}")
                    continue
    return sorted(list(mutations))


# --- Structural position selection ---
def select_structural_positions(pdb_file, max_distance=8.0):
    """Select positions within 8Å of the chromophore"""
    try:
        parser = PDBParser()
        structure = parser.get_structure('avGFP', pdb_file)
        chromophore = [r for r in structure[0].get_residues() if r.id[1] in [65, 66, 67]]
        if not chromophore:
            logging.error("Chromophore residues (65-67) not found in PDB")
            return []
        selected_positions = []
        for res in structure[0].get_residues():
            res_id = res.id[1]
            min_dist = float('inf')
            for chromo_res in chromophore:
                for atom1 in res:
                    for atom2 in chromo_res:
                        try:
                            dist = atom1 - atom2
                            min_dist = min(min_dist, dist)
                        except Exception as e:
                            logging.warning(f"Distance calculation failed for residue {res_id}: {str(e)}")
                            continue
            if min_dist < max_distance:
                selected_positions.append(res_id)
        logging.info(f"Selected {len(selected_positions)} structural positions")
        return sorted(selected_positions)
    except Exception as e:
        logging.error(f"Structural position selection failed: {str(e)}")
        return []


# --- Position-specific amino acid selection ---
def get_position_specific_amino_acids(pos, seq):
    """Select amino acids based on position type"""
    try:
        if pos + 1 in [64, 65, 66, 68, 69, 70]:
            return ['T', 'N', 'Q', 'F', 'Y', 'W', 'H', 'L']  # Chromophore-proximal
        elif pos + 1 in [10, 163, 167, 171, 187]:
            return ['V', 'I', 'L']  # Hydrophobic core
        elif pos + 1 in [101, 149, 153, 203, 206]:
            return ['R', 'K', 'D', 'E', 'N', 'Q']  # Surface
        else:
            return [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != seq[pos]]
    except IndexError:
        logging.error(f"Index error in amino acid selection for position {pos}")
        return ['A']  # Fallback to alanine


# --- Mutation sequence generation ---
def generate_mutations_with_fixed(wt_seq, max_mutations=6, exclusion=set(), position_pool=None, fixed_mutations=None):
    """Generate candidate mutation sequences"""
    candidates = []
    position_pool = position_pool or list(range(len(wt_seq)))
    fixed_mutations = fixed_mutations or []
    try:
        for _ in range(N_CANDIDATES):
            seq = list(wt_seq)
            applied_muts = []
            selected_fixed = random.sample(fixed_mutations, min(random.randint(1, 2), len(fixed_mutations)))
            for mut in selected_fixed:
                if mut in exclusion:
                    continue
                if re.match(r'[A-Z]\d+[A-Z]', mut):
                    try:
                        pos = int(mut[1:-1]) - 1
                        new_aa = mut[-1]
                        if 0 <= pos < len(seq) and seq[pos] == mut[0]:
                            seq[pos] = new_aa
                            applied_muts.append(pos)
                        else:
                            logging.warning(
                                f"Invalid fixed mutation: {mut}, pos {pos}, orig_aa {seq[pos] if 0 <= pos < len(seq) else 'out of range'}")
                    except ValueError:
                        logging.warning(f"Invalid fixed mutation format: {mut}")
                        continue
            n_extra = random.randint(1, 2)
            available_pos = [p for p in position_pool if p not in applied_muts]
            if n_extra > len(available_pos):
                continue
            mut_pos = random.sample(available_pos, n_extra)
            for pos in mut_pos:
                try:
                    new_aa = random.choice(get_position_specific_amino_acids(pos, seq))
                    seq[pos] = new_aa
                except IndexError:
                    logging.warning(f"Index error during mutation at position {pos}")
                    continue
            seq_str = ''.join(seq)
            if seq_str not in exclusion and seq_str != wt_seq:
                candidates.append(seq_str)
        candidates = list(set(candidates))
        logging.info(f"Generated {len(candidates)} unique candidate sequences")
        print(f"Generated {len(candidates)} unique candidate sequences")
        return candidates
    except Exception as e:
        logging.error(f"Mutation generation failed: {str(e)}")
        raise


# --- SASA calculation ---
def calculate_sasa(pdb_file, pos):
    """Calculate solvent-accessible surface area for a residue"""
    try:
        structure = PDBParser().get_structure('avGFP', pdb_file)
        model = structure[0]
        from Bio.PDB.DSSP import DSSP
        dssp = DSSP(model, pdb_file)
        for key in dssp.keys():
            if dssp[key][1] == str(pos):
                return float(dssp[key][3]) * 100  # Convert to Å²
        logging.warning(f"Residue {pos} not found in DSSP output, returning 0.0")
        return 0.0
    except Exception as e:
        logging.warning(f"SASA calculation failed for residue {pos}: {str(e)}, returning 0.0")
        return 0.0


# --- Stability prediction model ---
class StabilityPredictor:
    def __init__(self, pdb_file, df):
        try:
            self.structure = PDBParser().get_structure('avGFP', pdb_file)
            self.stable_mutations = set(BENEFICIAL_MUTATIONS)
            self.core_residues = [10, 64, 163, 167, 171, 187]
            self.surface_residues = [101, 149, 153, 203, 206, 221, 231, 235]
        except Exception as e:
            logging.error(f"StabilityPredictor initialization failed: {str(e)}")
            raise

    def predict(self, mutations):
        """Predict stability of a mutated sequence"""
        if not mutations:  # 无突变时返回基准分
            return 75.0
        score = 50.0  # 中等初始分数
        for mut in mutations:
            try:
                if mut in self.stable_mutations:
                    score += 15  # 已知有益突变大幅加分
                    continue
                pos = int(mut[1:-1])
                new_aa = mut[-1]
                orig_aa = mut[0]
                sasa = calculate_sasa(PDB_FILE, pos)
                # 更严格的评分规则
                if sasa > 50:  # 表面残基
                    if new_aa in ['R', 'K', 'D', 'E']:  # 带电氨基酸
                        score += 3
                    elif new_aa in ['A', 'V', 'I', 'L']:  # 疏水氨基酸
                        score -= 5
                    elif new_aa == 'P':  # 脯氨酸破坏二级结构
                        score -= 8
                else:  # 核心残基
                    if new_aa in ['V', 'I', 'L']:  # 适合核心的疏水氨基酸
                        score += 5
                    elif new_aa in ['S', 'T']:  # 小极性氨基酸
                        score -= 2
                    elif new_aa in ['R', 'K', 'D', 'E']:  # 带电氨基酸不适合核心
                        score -= 10
                    elif new_aa == 'P':  # 核心区脯氨酸严重破坏结构
                        score -= 15
                # 特殊残基类型处理
                if orig_aa == 'G':  # 甘氨酸突变
                    score -= 8  # 甘氨酸通常有特殊结构作用
                elif orig_aa == 'P' and new_aa != 'P':  # 脯氨酸突变
                    score += 5  # 通常改善结构
            except ValueError:
                logging.warning(f"Invalid mutation format: {mut}")
                score -= 5
        return max(0, min(100, score))  # 仍然限制在0-100范围


# --- ESM embeddings ---
def get_esm_embeddings(sequences, model, alphabet, device, batch_size=BATCH_SIZE):
    """Generate ESM-2 embeddings for sequences"""
    try:
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        embeddings = []
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        logging.info(
            f"Generating embeddings: {len(sequences)} sequences, {num_batches} batches, batch size {batch_size}, device {device}")
        print(f"Generating embeddings: {len(sequences)} sequences, {num_batches} batches")
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            data = [(f"seq{j}", seq) for j, seq in enumerate(batch)]
            try:
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                with torch.no_grad():
                    results = model(tokens, repr_layers=[30])
                    seq_emb = results["representations"][30][:, 1:-1].mean(dim=1).cpu().numpy()
                    embeddings.append(seq_emb)
                logging.info(f"Processed batch {i // batch_size + 1}/{num_batches}")
                print(f"Processed batch {i // batch_size + 1}/{num_batches}")
            except Exception as e:
                logging.error(f"Batch {i // batch_size + 1} failed: {str(e)}")
                print(f"Batch {i // batch_size + 1} failed: {str(e)}")
                continue
        if not embeddings:
            raise RuntimeError("No embeddings generated")
        return np.vstack(embeddings)
    except Exception as e:
        logging.error(f"Embedding generation failed: {str(e)}")
        raise


# --- Attention-enhanced random forest ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)
        output = self.out(context)
        return output


class AttentionEnhancedRF:
    def __init__(self, n_estimators=200, embed_dim=640, num_heads=4, random_state=SEED):
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.device = DEVICE

    def fit(self, X, y):
        try:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                X_transformed = self.attention(X_tensor).cpu().numpy()
            self.rf.fit(X_transformed, y)
        except Exception as e:
            logging.error(f"RF training failed: {str(e)}")
            raise

    def predict(self, X):
        try:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                X_transformed = self.attention(X_tensor).cpu().numpy()
            return self.rf.predict(X_transformed)
        except Exception as e:
            logging.error(f"RF prediction failed: {str(e)}")
            raise


# --- Visualize mutation positions ---
def plot_mutation_positions(submission_df, pdb_file):
    """Plot 3D distribution of mutation positions"""
    try:
        structure = PDBParser().get_structure('avGFP', pdb_file)
        positions = set()
        for muts in submission_df['Mutations']:
            for mut in muts.split(':'):
                if re.match(r'[A-Z]\d+[A-Z]', mut):
                    try:
                        positions.add(int(mut[1:-1]))
                    except ValueError:
                        logging.warning(f"Invalid mutation format: {mut}")
                        continue

        x, y, z = [], [], []
        for res in structure[0].get_residues():
            if res.id[1] in positions:
                for atom in res:
                    if atom.name == 'CA':
                        coord = atom.get_coord()
                        x.append(coord[0])
                        y.append(coord[1])
                        z.append(coord[2])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='red', s=50, label='Mutation Positions')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        plt.title('Mutation Positions in GFP Structure')
        plt.legend()
        plt.savefig('mutation_positions.png')
        plt.close()
        logging.info("Generated mutation positions plot: mutation_positions.png")
        print("Generated mutation positions plot: mutation_positions.png")
    except Exception as e:
        logging.error(f"Mutation position visualization failed: {str(e)}")
        print(f"Mutation position visualization failed: {str(e)}")


# --- Multi-objective optimization model ---
class GFPDesigner:
    def __init__(self, df, wt_seq, exclusion):
        try:
            self.df = df[df['GFP type'] == 'avGFP']
            logging.info(f"Filtered avGFP data, remaining {len(self.df)} records, columns: {list(self.df.columns)}")
            print(f"Filtered avGFP data, remaining {len(self.df)} records, columns: {list(self.df.columns)}")
            self.wt_seq = wt_seq
            self.exclusion = exclusion
            logging.info(f"Loading ESM-2 model: {ESM_MODEL_NAME} to device: {DEVICE}")
            print(f"Loading ESM-2 model: {ESM_MODEL_NAME} to device: {DEVICE}")
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
            self.esm_model.to(DEVICE)
            self.esm_model.eval()
            logging.info(f"Model loaded, located on device: {DEVICE}")
            print(f"Model loaded, located on device: {DEVICE}")
            self.stability_model = StabilityPredictor(PDB_FILE, self.df)
            self.brightness_model = self.train_brightness_model()
        except Exception as e:
            logging.error(f"GFPDesigner initialization failed: {str(e)}")
            raise

    def train_brightness_model(self):
        """Train brightness prediction model"""
        try:
            seqs = self.df['aaMutations'].apply(lambda x: generate_mutated_sequence(x, self.wt_seq))
            valid_seqs = seqs.dropna().tolist()
            logging.info(f"Original sequence count: {len(valid_seqs)}")
            print(f"Original sequence count: {len(valid_seqs)}")
            if len(valid_seqs) > 5000:
                valid_seqs = random.sample(valid_seqs, 5000)
            embeddings = get_esm_embeddings(valid_seqs, self.esm_model, self.alphabet, DEVICE)
            X = embeddings
            y = self.df[seqs.notna()]['Brightness'].values[:len(valid_seqs)]
            model = AttentionEnhancedRF(n_estimators=200, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, random_state=SEED)
            model.fit(X, y)
            logging.info("Brightness model trained successfully")
            return model
        except Exception as e:
            logging.error(f"Brightness model training failed: {str(e)}")
            raise

    def evaluate_candidate(self, seq):
        """Evaluate candidate sequence for brightness and stability"""
        try:
            mutations = self.get_mutations(seq)
            emb = get_esm_embeddings([seq], self.esm_model, self.alphabet, DEVICE, batch_size=1)
            brightness = self.brightness_model.predict(emb)[0]
            stability = self.stability_model.predict(mutations)
            return brightness, stability
        except Exception as e:
            logging.error(f"Candidate evaluation failed: {str(e)}")
            raise

    def get_mutations(self, seq):
        """Get list of mutations in a sequence"""
        try:
            mutations = []
            for i in range(len(seq)):
                if seq[i] != self.wt_seq[i]:
                    mutations.append(f"{self.wt_seq[i]}{i + 1}{seq[i]}")
            return mutations
        except IndexError:
            logging.error("Index error in get_mutations")
            raise

    def design_sequences(self):
        """Design optimized sequences using Pareto front"""
        try:
            candidates = generate_mutations_with_fixed(
                self.wt_seq, MAX_MUTATIONS, self.exclusion, CANDIDATE_POSITION_POOL_0BASED, BENEFICIAL_MUTATIONS
            )
            logging.info(f"Generated {len(candidates)} candidate sequences")
            print(f"Generated {len(candidates)} candidate sequences")
            results = []
            for idx, seq in enumerate(candidates, 1):
                try:
                    bright, stable = self.evaluate_candidate(seq)
                    results.append((seq, bright, stable))
                    if idx % 100 == 0 or idx == len(candidates):
                        logging.info(f"Evaluated {idx}/{len(candidates)} sequences")
                        print(f"Evaluated {idx}/{len(candidates)} sequences")
                except Exception as e:
                    logging.error(f"Evaluation of sequence {idx} failed: {str(e)}")
                    print(f"Evaluation of sequence {idx} failed: {str(e)}")
                    continue
            logging.info(f"Evaluation completed, {len(results)} valid sequences")
            print(f"Evaluation completed, {len(results)} valid sequences")

            def pareto_front(results):
                sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                pareto = []
                max_stability = -float('inf')
                seen_positions = set()
                for seq, bright, stable in sorted_results:
                    if stable >= max_stability:
                        muts = ':'.join(self.get_mutations(seq))
                        mut_positions = set(int(m[1:-1]) for m in muts.split(':') if re.match(r'[A-Z]\d+[A-Z]', m))
                        if not seen_positions.intersection(mut_positions):
                            pareto.append((seq, bright, stable))
                            max_stability = stable
                            seen_positions.update(mut_positions)
                return pareto

            top_seqs = pareto_front(results)
            logging.info(f"Pareto front contains {len(top_seqs)} sequences")
            print(f"Pareto front contains {len(top_seqs)} sequences")

            if len(top_seqs) < TOP_N:
                logging.warning(f"Pareto front has fewer than {TOP_N} sequences, supplementing")
                print(f"Warning: Pareto front has fewer than {TOP_N} sequences, supplementing")
                remaining = [r for r in results if r not in top_seqs]
                score_groups = {}
                for seq, bright, stable in remaining:
                    score_key = (round(bright, 4), round(stable, 4))
                    if score_key not in score_groups:
                        score_groups[score_key] = []
                    score_groups[score_key].append((seq, bright, stable))
                for score_key in sorted(score_groups.keys(), key=lambda x: x[0] * 0.6 + x[1] * 0.4, reverse=True):
                    top_seqs.extend(score_groups[score_key])
                    if len(top_seqs) >= TOP_N:
                        break
            logging.info(f"Final selection: {len(top_seqs)} sequences")
            print(f"Final selection: {len(top_seqs)} sequences")

            submission = []
            for idx, (seq, bright, stable) in enumerate(top_seqs, 1):
                muts = self.get_mutations(seq)
                submission.append({
                    'Sequence ID': f'Seq{idx}',
                    'Mutations': ':'.join(muts),
                    'Full Sequence': seq,
                    'Brightness Score': bright,
                    'Stability Score': stable
                })
            submission_df = pd.DataFrame(submission)
            logging.info(f"Generated submission DataFrame with {len(submission_df)} sequences")
            print(f"Generated submission DataFrame with {len(submission_df)} sequences")
            return submission_df
        except Exception as e:
            logging.error(f"Sequence design failed: {str(e)}")
            raise


# --- Helper function ---
def generate_mutated_sequence(mutation_str, wt_seq):
    """Generate mutated sequence from mutation string"""
    try:
        if pd.isna(mutation_str) or mutation_str.strip() == '':
            return wt_seq
        seq = list(wt_seq)
        mutations = mutation_str.split(':')
        for mut in mutations:
            if not re.match(r'[A-Z]\d+[A-Z]', mut):
                logging.warning(f"Skipping invalid mutation format: {mut}")
                continue
            try:
                orig_aa = mut[0]
                pos = int(mut[1:-1]) - 1
                new_aa = mut[-1]
                if 0 <= pos < len(seq) and seq[pos] == orig_aa:
                    seq[pos] = new_aa
                else:
                    logging.warning(
                        f"Mutation {mut} invalid: position {pos + 1} out of range or original amino acid mismatch")
            except ValueError:
                logging.warning(f"Mutation {mut} format error")
                continue
        return ''.join(seq)
    except Exception as e:
        logging.error(f"Mutated sequence generation failed: {str(e)}")
        raise


# --- Main function ---
if __name__ == "__main__":
    try:
        print_resource_usage()
        df, wt_seq, exclusion = load_data()
        structural_positions = select_structural_positions(PDB_FILE)
        CANDIDATE_POSITION_POOL.extend([p for p in structural_positions if p not in CANDIDATE_POSITION_POOL])
        CANDIDATE_POSITION_POOL_0BASED = [p - 1 for p in CANDIDATE_POSITION_POOL]
        logging.info(f"Updated candidate position pool with {len(CANDIDATE_POSITION_POOL)} structural positions")
        print(f"Updated candidate position pool with {len(CANDIDATE_POSITION_POOL)} structural positions")
        beneficial_muts = extract_beneficial_mutations(df)
        BENEFICIAL_MUTATIONS.extend([m for m in beneficial_muts if m not in BENEFICIAL_MUTATIONS])
        logging.info(f"Updated beneficial mutations list with {len(BENEFICIAL_MUTATIONS)} mutations")
        print(f"Updated beneficial mutations list with {len(BENEFICIAL_MUTATIONS)} mutations")
        designer = GFPDesigner(df, wt_seq, exclusion)
        final_submission = designer.design_sequences()
        final_submission.to_csv('submission.csv', index=False, mode='w')
        logging.info(f"Successfully generated {len(final_submission)} candidate sequences, saved to submission.csv")
        print(f"Successfully generated {len(final_submission)} candidate sequences, saved to submission.csv")
        print("\nFinal sequences and scores:")
        print(final_submission[
                  ['Sequence ID', 'Mutations', 'Full Sequence', 'Brightness Score', 'Stability Score']].to_string(
            index=False))
        plot_mutation_positions(final_submission, PDB_FILE)
        saved_df = pd.read_csv('submission.csv')
        logging.info(f"Verification: submission.csv contains {len(saved_df)} sequences")
        print(f"\nVerification: submission.csv contains {len(saved_df)} sequences")
    except Exception as e:
        logging.error(f"Design pipeline failed: {str(e)}")
        print(f"Design pipeline failed: {str(e)}")
        print("Please check GFP_data.xlsx columns, file paths, DSSP configuration, or gfp_design.log")