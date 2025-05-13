import pandas as pd
import numpy as np
import torch
import esm # fair-esm library
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import re # 用于解析突变字符串
import warnings
import time  # 用于计时
warnings.filterwarnings('ignore') # 忽略一些不影响结果的警告

# --- 常量定义 ---
DATA_DIR = 'D:\\2025Protein Design'
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'GFP_data.xlsx')
WT_SEQ_FILE = os.path.join(DATA_DIR, 'AAseqs of 4 GFP proteins.txt') # 假设avGFP序列在此文件中
EXCLUSION_FILE = os.path.join(DATA_DIR, 'Exclusion_List.csv')

# --- 模型与生成参数 ---
ESM_MODEL_NAME = "esm2_t6_8M_UR50D" # 选择esm2_t6_8M_UR50DESM模型
MAX_MUTATIONS = 6 # 比赛规则：最多6个突变
N_CANDIDATES_TO_GENERATE = 500 # 生成候选序列的数量（可调整）
TOP_N_SELECT = 6 # 最终选择的序列数量

# 检查是否有可用的 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 设置随机种子以便结果可复现（可选）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 2.1 加载训练数据 ---
print("Loading training data...")
try:
    gfp_df = pd.read_excel(TRAIN_DATA_FILE, sheet_name='brightness') # 假设亮度数据在名为 'brightness' 的 sheet
    print(f"Loaded {len(gfp_df)} rows from {TRAIN_DATA_FILE}")
except FileNotFoundError:
    print(f"Error: Training data file not found at {TRAIN_DATA_FILE}")
    # 在 Bohrium 中，您可能需要检查路径是否正确或数据是否已挂载
    # exit() # 如果文件不存在，可能需要停止执行

# --- 2.2 加载 avGFP 野生型序列 ---
print("Loading avGFP WT sequence...")
avGFP_WT_sequence = None
try:
    with open(WT_SEQ_FILE, 'r') as f:
        # 假设文件格式是 >Header \n Sequence \n >Header2...
        # 我们需要找到 avGFP 的序列
        print("\n")
        header = ""
        seq_lines = []
        for line in f:
            if line.startswith('>'):
                # 如果找到了上一个序列，并且是avGFP，保存它
                if "avGFP" in header and seq_lines:
                    avGFP_WT_sequence = "".join(seq_lines).strip()
                    break # 找到后退出循环
                # 开始新的序列记录
                header = line.strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        # 处理文件最后一个序列的情况
        if avGFP_WT_sequence is None and "avGFP" in header and seq_lines:
             avGFP_WT_sequence = "".join(seq_lines).strip()

    if avGFP_WT_sequence:
        print(f"Found avGFP WT sequence (Length: {len(avGFP_WT_sequence)}).")
        # print(avGFP_WT_sequence) # 可以取消注释查看序列
    else:
        print("Error: avGFP WT sequence not found in", WT_SEQ_FILE)
        # 手动设置一个默认值或停止执行
        # avGFP_WT_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" # 示例
        # print("Using default WT sequence.")
        # exit()

except FileNotFoundError:
    print(f"Error: WT sequence file not found at {WT_SEQ_FILE}")
    # exit()

# --- 2.3 加载排除列表 ---
print("Loading exclusion list...")
try:
    exclusion_df = pd.read_csv(EXCLUSION_FILE)
    # 假设排除序列在名为 'sequences-not-submit' 的列中
    exclusion_sequences = set(exclusion_df['sequences-not-submit'].astype(str))
    print(f"Loaded {len(exclusion_sequences)} sequences into exclusion list.")
except FileNotFoundError:
    print(f"Error: Exclusion list file not found at {EXCLUSION_FILE}")
    exclusion_sequences = set() # 如果文件不存在，创建一个空集合
    print("Warning: Proceeding without an exclusion list.")
except KeyError:
    print(f"Error: Column 'sequences-not-submit' not found in {EXCLUSION_FILE}")
    exclusion_sequences = set()
    print("Warning: Proceeding without an exclusion list.")


# --- 2.4 预处理训练数据 ---
print("Preprocessing training data...")
# 筛选 avGFP 数据
avGFP_train_df = gfp_df[gfp_df['GFP type'] == 'avGFP'].copy()
print(f"Filtered down to {len(avGFP_train_df)} avGFP entries.")

# 定义函数：根据突变字符串生成完整序列
def generate_mutated_sequence(mutation_str, wt_sequence):
    """
    根据突变描述字符串和野生型序列生成突变后的完整序列。
    mutation_str: e.g., "WT", "G101A", "A12B:C34D"
    wt_sequence: 野生型氨基酸序列字符串
    """
    if not isinstance(mutation_str, str) or not wt_sequence:
        return None
    if mutation_str.strip().upper() == 'WT':
        return wt_sequence

    sequence = list(wt_sequence)
    mutations = mutation_str.split(':') # 支持多个突变，以冒号分隔
    valid_mutation_count = 0
    try:
        for mut in mutations:
            match = re.match(r'([A-Z])(\d+)([A-Z*.])$', mut.strip(), re.IGNORECASE) # 匹配 G101A, T203*, V163.
            if match:
                original_aa, pos, new_aa = match.groups()
                pos = int(pos) - 1 # 转换为 0-based index

                # 检查位置是否有效
                if pos < 0 or pos >= len(sequence):
                    # print(f"Warning: Invalid position {pos+1} in mutation '{mut}' for sequence length {len(sequence)}. Skipping mutation.")
                    continue # 跳过无效位置的突变

                # 检查原始氨基酸是否匹配 (可选，但建议)
                if sequence[pos].upper() != original_aa.upper():
                    # print(f"Warning: Original AA mismatch at position {pos+1} in mutation '{mut}'. Expected {sequence[pos]}, got {original_aa}. Applying mutation anyway.")
                    pass # 允许不匹配，但打印警告

                # 处理特殊字符
                if new_aa == '*': # 终止密码子 - 通常不希望出现在中间
                    # print(f"Warning: Stop codon '*' mutation '{mut}' encountered. Treating as deletion or invalid sequence for this tutorial.")
                    # 对于亮度预测，终止密码子通常导致无功能蛋白，可以返回None或特殊标记
                    return None # 或者根据需要处理
                elif new_aa == '.': # 表示与原氨基酸相同 (无突变)
                    new_aa = sequence[pos] # 保持不变

                sequence[pos] = new_aa.upper()
                valid_mutation_count += 1
            else:
                # print(f"Warning: Could not parse mutation '{mut}'. Skipping.")
                pass # 跳过无法解析的突变格式
        # 如果没有成功应用任何突变（可能是格式问题），返回None
        # if valid_mutation_count == 0 and mutations:
        #     return None
        return "".join(sequence)
    except Exception as e:
        # print(f"Error processing mutation string '{mutation_str}': {e}")
        return None # 返回 None 表示序列生成失败

# 应用函数生成序列
avGFP_train_df['full_sequence'] = avGFP_train_df['aaMutations'].apply(
    lambda x: generate_mutated_sequence(x, avGFP_WT_sequence)
)

# 清理数据：移除序列生成失败或亮度无效的行
original_len = len(avGFP_train_df)
avGFP_train_df.dropna(subset=['full_sequence', 'Brightness'], inplace=True)
# 确保亮度是数值类型
avGFP_train_df['Brightness'] = pd.to_numeric(avGFP_train_df['Brightness'], errors='coerce')
avGFP_train_df.dropna(subset=['Brightness'], inplace=True)

print(f"Removed {original_len - len(avGFP_train_df)} rows due to invalid sequences or brightness.")
print(f"Final training set size: {len(avGFP_train_df)}")

# 查看处理后的数据
print("\nSample of processed training data:")
print(avGFP_train_df[['aaMutations', 'Brightness', 'full_sequence']].head())


# --- 优化常量 ---
# 根据设备选择合适的ESM模型
ESM_MODEL_NAME_CPU = "esm2_t6_8M_UR50D"
ESM_MODEL_NAME_GPU = "esm2_t30_150M_UR50D" # 可以选用更大的模型，例如 "esm2_t33_650M_UR50D"，取决于GPU内存

# 根据设备选择合适的批次大小
CPU_BATCH_SIZE = 8
GPU_BATCH_SIZE = 16 # GPU通常可以处理更大的批次，可以根据GPU内存调整

# 用于嵌入的最大训练样本数（可按需调整）
# 如果数据集小于此值，则使用所有样本
MAX_TRAIN_SAMPLES_FOR_EMBEDDING = 5000
SEED = 42 # 确保采样可复现

# --- 1. 设备检测 ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    ESM_MODEL_NAME = ESM_MODEL_NAME_GPU
    BATCH_SIZE = GPU_BATCH_SIZE
    print("检测到CUDA GPU，将使用GPU进行计算。")
else:
    DEVICE = torch.device("cpu")
    ESM_MODEL_NAME = ESM_MODEL_NAME_CPU
    BATCH_SIZE = CPU_BATCH_SIZE
    print("未检测到CUDA GPU或CUDA不可用，将使用CPU进行计算。")

print(f"正在使用设备: {DEVICE}")
print(f"将加载的ESM模型: {ESM_MODEL_NAME}")
print(f"将使用的批次大小: {BATCH_SIZE}")


# --- 3.0（优化）必要时对训练数据进行采样 ---
print("\n正在检查训练数据大小以进行采样...")
# 直接使用已加载的 avGFP_train_df
if len(avGFP_train_df) > MAX_TRAIN_SAMPLES_FOR_EMBEDDING:
    print(f"训练数据大小 ({len(avGFP_train_df)}) 超过限制 ({MAX_TRAIN_SAMPLES_FOR_EMBEDDING})。正在采样...")
    # 对DataFrame进行采样以减少用于嵌入的序列数量
    sampled_train_df = avGFP_train_df.sample(n=MAX_TRAIN_SAMPLES_FOR_EMBEDDING, random_state=SEED)
    print(f"使用 {len(sampled_train_df)} 个采样序列进行嵌入。")
else:
    print(f"训练数据大小 ({len(avGFP_train_df)}) 在限制范围内。使用所有序列。")
    # 使用完整的DataFrame
    sampled_train_df = avGFP_train_df.copy()  # 使用副本以避免修改原始数据

# --- 3.1 加载适合选定设备的ESM模型 ---
print(f"\n正在加载ESM模型: {ESM_MODEL_NAME} 到设备 {DEVICE}...")
start_time = time.time()
try:
    # 加载模型和字母表
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    esm_model.eval()  # 设置为评估模式
    esm_model = esm_model.to(DEVICE)  # 将模型移动到选定的设备 (GPU or CPU)
    batch_converter = alphabet.get_batch_converter()
    print(f"ESM模型在 {time.time() - start_time:.2f} 秒内成功加载。")
except Exception as e:
    print(f"加载ESM模型时出错: {e}")
    print("请确保已安装 'fair-esm' 库 (pip install fair-esm) 且模型名称正确。")
    print("对于GPU使用，请确保已安装与CUDA兼容的PyTorch版本。")
    exit()  # 无法加载模型则退出

# --- 3.2 定义嵌入函数（适用于CPU和GPU） ---
def get_esm_embeddings(sequences, model, alphabet, batch_converter, device, batch_size):
    """
    在指定设备(CPU或GPU)上生成ESM嵌入（平均池化）。
    将输入数据和模型都移动到 `device`。
    """
    embeddings = []
    num_sequences = len(sequences)
    num_batches = (num_sequences + batch_size - 1) // batch_size
    model.eval() # 确保模型在评估模式
    model = model.to(device) # 确保模型在目标设备

    print(f"正在为 {num_sequences} 个序列生成嵌入，共 {num_batches} 个批次（批次大小: {batch_size}，设备: {device}）...")
    start_time_embed = time.time()

    with torch.no_grad():  # 对推理速度和内存至关重要
        for i in range(0, num_sequences, batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_labels = [f"seq_{j + i}" for j in range(len(batch_seqs))]  # 每个批次项的唯一标签
            data = list(zip(batch_labels, batch_seqs))

            try:
                # 1. 准备批次
                _, _, batch_tokens = batch_converter(data)
                # 将令牌移动到目标设备 (GPU or CPU)
                batch_tokens = batch_tokens.to(device)

                # 2. 获取表示（只需要最后一层）
                # results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False) # 旧版写法
                # 适配新版 fair-esm (>=2.0.0)
                results = model(batch_tokens, repr_layers=[model.num_layers])
                token_representations = results["representations"][model.num_layers]

                # 3. 对序列长度进行平均池化（忽略CLS/EOS/PAD令牌）
                # token_representations形状: [batch_size, seq_len+2, embed_dim]
                seq_repr_list = []
                for j, seq in enumerate(batch_seqs):
                    # +1 是因为 batch_converter 添加了 <cls> token
                    actual_len = len(seq)
                    # 从索引 1 开始到 actual_len 结束 (不包括 <eos> 和 padding)
                    # 注意：需要确保batch_converter没有添加其他的特殊token影响长度计算
                    # 对于标准esm模型，通常是<cls>...seq...<eos><pad>...
                    # 所以 token_representations[j, 1 : actual_len + 1, :] 是正确的
                    seq_tokens_repr = token_representations[j, 1 : actual_len + 1, :]
                    seq_repr = seq_tokens_repr.mean(dim=0) # 对实际序列长度的表示进行平均
                    seq_repr_list.append(seq_repr)

                batch_seq_repr = torch.stack(seq_repr_list, dim=0) # [batch_size, embed_dim]

                # 4. 存储结果 (移回CPU以聚合和后续处理，如转Numpy)
                embeddings.append(batch_seq_repr.cpu())

                if (i // batch_size + 1) % 10 == 0 or (i // batch_size + 1) == num_batches:  # 每10个批次或最后一个批次打印进度
                    elapsed_time = time.time() - start_time_embed
                    print(f"  已处理批次 {i // batch_size + 1}/{num_batches}... (耗时: {elapsed_time:.2f} 秒)")

            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and device == torch.device("cuda"):
                    print(f"\n处理批次 {i // batch_size + 1} 时发生CUDA内存不足错误: {e}")
                    print(f"当前批次大小: {batch_size}。尝试减小批次大小或使用更小的模型。")
                    # 记录错误并跳过，但会导致数据丢失
                    embed_dim = model.embed_dim
                    error_placeholder = torch.full((len(batch_seqs), embed_dim), float('nan'), device='cpu') # 存放在CPU
                    embeddings.append(error_placeholder)
                    torch.cuda.empty_cache() # 尝试释放一些内存
                else:
                    print(f"处理批次 {i // batch_size + 1} 时发生运行时错误: {e}")
                    embed_dim = model.embed_dim
                    error_placeholder = torch.full((len(batch_seqs), embed_dim), float('nan'), device='cpu')
                    embeddings.append(error_placeholder)
            except Exception as e:
                print(f"处理批次 {i // batch_size + 1} 时发生未知错误: {e}")
                # 处理错误，例如跳过批次或用NaN填充
                embed_dim = model.embed_dim  # 获取预期维度
                error_placeholder = torch.full((len(batch_seqs), embed_dim), float('nan'), device='cpu')
                embeddings.append(error_placeholder)


    total_embed_time = time.time() - start_time_embed
    print(f"嵌入生成在 {total_embed_time:.2f} 秒内完成。")
    if num_sequences > 0:
      print(f"平均每个序列耗时: {total_embed_time / num_sequences:.4f} 秒")


    if not embeddings:
        return torch.tensor([])  # 返回空张量

    # 连接所有批次的结果
    try:
        full_embeddings = torch.cat(embeddings, dim=0)
    except RuntimeError as e:
        print(f"连接嵌入批次时出错: {e}")
        print("这可能发生在批次处理出错导致维度不匹配时。请检查之前的错误信息。")
        # 尝试找出有效批次并连接，或者返回错误
        try:
            embed_dim = model.embed_dim # 获取模型维度
            valid_embeddings = [emb for emb in embeddings if isinstance(emb, torch.Tensor) and emb.ndim == 2 and emb.shape[1] == embed_dim and not torch.isnan(emb).all()]
            if valid_embeddings:
                print("尝试仅连接有效的嵌入批次...")
                full_embeddings = torch.cat(valid_embeddings, dim=0)
            else:
                print("没有有效的嵌入批次可以连接。")
                return torch.tensor([]) # 返回空张量
        except Exception as concat_err:
             print(f"尝试连接有效嵌入时再次出错: {concat_err}")
             return torch.tensor([]) # 返回空张量


    return full_embeddings  # 作为单个张量返回 (在 CPU 上)

# --- 3.3 为（可能采样的）训练数据生成嵌入 ---
train_sequences_to_embed = sampled_train_df['full_sequence'].tolist()

X = None # 初始化 X
y = None # 初始化 y

if train_sequences_to_embed:
    # 获取嵌入作为PyTorch张量
    train_embeddings_tensor = get_esm_embeddings(
        train_sequences_to_embed,
        esm_model,
        alphabet,
        batch_converter,
        DEVICE,            # 传递检测到的设备
        batch_size=BATCH_SIZE # 传递适合设备的批次大小
    )

    if train_embeddings_tensor.numel() > 0: # 检查张量是否为空
        print(f"生成的嵌入张量形状: {train_embeddings_tensor.shape}")

        # 将张量转换为numpy数组以与scikit-learn兼容
        # 因为上面 append 时已经 .cpu()，所以这里 tensor 已经在 CPU 上
        train_embeddings_np = train_embeddings_tensor.numpy()

        # --- 处理嵌入过程中可能出现的NaN值 ---
        nan_rows_mask = np.isnan(train_embeddings_np).any(axis=1)
        if np.any(nan_rows_mask):
            num_nan_rows = nan_rows_mask.sum()
            print(f"警告: 在嵌入中发现 {num_nan_rows} 行NaN值（可能是由于错误）。正在移除这些行及其对应的原始数据。")

            # 过滤嵌入数组和相应的DataFrame行
            valid_indices_bool = ~nan_rows_mask
            # 获取原始sampled_train_df中对应valid_indices_bool为True的索引
            original_indices = sampled_train_df.index[valid_indices_bool]

            train_embeddings_np = train_embeddings_np[valid_indices_bool]
            # 使用 .loc 和原始索引进行过滤，确保正确对齐
            sampled_train_df_filtered = sampled_train_df.loc[original_indices]

            print(f"过滤后的嵌入数据大小: {train_embeddings_np.shape[0]}")
            print(f"过滤后的DataFrame大小: {len(sampled_train_df_filtered)}")


            # 检查过滤后是否还有数据
            if train_embeddings_np.shape[0] == 0:
                 print("\n错误: 移除NaN值后没有剩余数据。无法继续。")
                 X, y = None, None
            else:
                 # --- 为模型训练准备最终的X和y ---
                 if train_embeddings_np.shape[0] == len(sampled_train_df_filtered):
                     X = train_embeddings_np
                     y = sampled_train_df_filtered['Brightness'].values
                     print(f"\n准备好的X（嵌入）形状: {X.shape}, y（亮度）形状: {y.shape}")
                     print("步骤3（嵌入生成）完成。")
                 else:
                     # 这个情况理论上不应发生，因为我们同时过滤了两者
                     print(f"\n错误: NaN过滤后最终嵌入 ({train_embeddings_np.shape[0]}) 和DataFrame行 ({len(sampled_train_df_filtered)}) 数量不匹配。")
                     print("无法进行训练。请检查过滤逻辑。")
                     X, y = None, None

        else:
             # 没有NaN值，直接使用
             print("嵌入中未检测到NaN值。")
             X = train_embeddings_np
             # 直接从 sampled_train_df 获取 y，因为没有过滤
             y = sampled_train_df['Brightness'].values
             print(f"\n准备好的X（嵌入）形状: {X.shape}, y（亮度）形状: {y.shape}")
             print("步骤3（嵌入生成）完成。")

    else:
        print("\n嵌入过程未能生成有效的嵌入张量（可能所有批次都出错或输入为空）。")
        X, y = None, None

else:
    print("\n采样/预处理后没有可用的序列进行嵌入。")
    X, y = None, None

# --- 清理（可选，有助于释放内存，特别是GPU内存） ---
print("\n正在清理内存...")
del esm_model, alphabet, batch_converter
# 删除可能已创建的张量和numpy数组
if 'train_embeddings_tensor' in locals() and isinstance(train_embeddings_tensor, torch.Tensor):
    del train_embeddings_tensor
if 'train_embeddings_np' in locals():
    del train_embeddings_np
# 删除采样或过滤后的DataFrame副本
if 'sampled_train_df' in locals():
    del sampled_train_df
if 'sampled_train_df_filtered' in locals():
     del sampled_train_df_filtered

# 如果使用了GPU，显式清空缓存
if DEVICE == torch.device("cuda"):
    print("清空CUDA缓存...")
    torch.cuda.empty_cache()
print("清理完成。")

# --- 现在X和y（如果成功创建）已准备好用于步骤4（模型训练） ---
if X is not None and y is not None:
     print(f"\n数据准备就绪，可以进行模型训练。X shape: {X.shape}, y shape: {y.shape}")
     # 这里可以接上你的模型训练代码，例如：
     # from sklearn.model_selection import train_test_split
     # from sklearn.ensemble import RandomForestRegressor
     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
     # model = RandomForestRegressor(n_estimators=100, random_state=SEED)
     # model.fit(X_train, y_train)
     # print("模型训练完成。")
     # score = model.score(X_test, y_test)
     # print(f"模型在测试集上的 R^2 分数: {score:.4f}")
else:
     print("\n数据准备失败，无法进行模型训练。请检查之前的日志输出。")

# 示例: 如果X不为None，则打印(X.shape, y.shape)，否则打印("X is None")
# print("\n--- 最终检查 ---")
# print((X.shape, y.shape) if X is not None else "X is None")

# --- 4.1 划分训练集和验证集 ---
# 如果数据量较少，可以考虑交叉验证，这里用简单的划分
if len(X) > 10: # 确保有足够数据划分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Split data into training ({len(X_train)}) and validation ({len(X_val)}) sets.")
else:
    print("Dataset too small for validation split, using all data for training.")
    X_train, y_train = X, y
    X_val, y_val = None, None # 没有验证集

# --- 4.2 初始化并训练随机森林模型 ---
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100, # 树的数量，可以调整
    random_state=SEED,
    n_jobs=-1, # 使用所有可用的 CPU 核心
    max_depth=20, # 限制树的深度，防止过拟合 (可调整)
    min_samples_leaf=3 # 叶节点最小样本数 (可调整)
)

rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# --- 4.3 (可选) 评估模型性能 ---
if X_val is not None:
    y_pred_val = rf_model.predict(X_val)
    r2 = r2_score(y_val, y_pred_val)
    print(f"\nModel Performance on Validation Set:")
    print(f"  R-squared (R²): {r2:.4f}")
    # R² 接近 1 表示模型拟合得较好，接近 0 或负数表示拟合很差
else:
    # 可以在训练集上评估，但这通常会过于乐观
    y_pred_train = rf_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print("\nModel Performance on Training Set (may be optimistic):")
    print(f"  R-squared (R²): {r2_train:.4f}")




# --- 5.1 定义候选位点池 (示例) ---
# !!! 关键步骤：这个列表应该基于您的研究 !!!
# 示例：包含一些文献中提到的稳定/亮度相关位点，以及靠近发色团的位点

candidate_position_pool = [
    # 靠近发色团 (at 65-67)
    64, 68, 69, 70, 71, 72,
    # 文献中提到的与稳定/亮度相关的 (示例)
    10, 30, 64, # F64L 是 Superfolder GFP 的关键突变之一
    101, 105, 109,
    145, 147, 153, # M153T 也是 Superfolder GFP 突变
    163, 167, # V163A 也是 Superfolder GFP 突变
    171, 187,
    203, 205, 221, 231, 232, 235
]
# 转换为 0-based index 用于代码处理
candidate_position_pool_0based = [p - 1 for p in candidate_position_pool]
print(f"\nUsing a candidate position pool of {len(candidate_position_pool)} sites (0-based index):")
# print(candidate_position_pool_0based)

# --- 5.2 定义生成候选序列的函数 ---
amino_acids = 'ACDEFGHIKLMNPQRSTVWY' # 20种标准氨基酸

def generate_single_candidate(wt_sequence, position_pool_0based, max_mutations):
    """生成一个随机组合突变的候选序列"""
    num_mutations = random.randint(1, max_mutations)
    # 从池中随机选择 num_mutations 个不同的位置
    positions_to_mutate = random.sample(position_pool_0based, num_mutations)

    mutated_sequence = list(wt_sequence)
    mutation_details = []

    for pos in positions_to_mutate:
        original_aa = wt_sequence[pos]
        # 随机选择一个不同于原始氨基酸的新氨基酸
        possible_new_aas = [aa for aa in amino_acids if aa != original_aa]
        new_aa = random.choice(possible_new_aas)
        mutated_sequence[pos] = new_aa
        mutation_details.append(f"{original_aa}{pos+1}{new_aa}") # 记录突变 (1-based)

    return "".join(mutated_sequence), ":".join(sorted(mutation_details, key=lambda x: int(re.search(r'\d+', x).group()))) # 按位置排序突变描述

# --- 5.3 生成大量候选序列 ---
print(f"\nGenerating {N_CANDIDATES_TO_GENERATE} candidate sequences...")
candidate_sequences = {} # 使用字典存储 {sequence: mutation_str} 以确保唯一性
generated_count = 0
attempts = 0
max_attempts = N_CANDIDATES_TO_GENERATE * 5 # 设置尝试上限，防止无限循环

while generated_count < N_CANDIDATES_TO_GENERATE and attempts < max_attempts:
    attempts += 1
    seq, mut_str = generate_single_candidate(avGFP_WT_sequence, candidate_position_pool_0based, MAX_MUTATIONS)
    if seq not in candidate_sequences and seq != avGFP_WT_sequence: # 确保不重复且不是野生型
        candidate_sequences[seq] = mut_str
        generated_count += 1
        if generated_count % (N_CANDIDATES_TO_GENERATE // 10) == 0:
            print(f"  Generated {generated_count}/{N_CANDIDATES_TO_GENERATE} unique candidates...")

if generated_count < N_CANDIDATES_TO_GENERATE:
    print(f"Warning: Could only generate {generated_count} unique candidates after {attempts} attempts.")

candidate_list = list(candidate_sequences.keys())
mutation_list = [candidate_sequences[seq] for seq in candidate_list]
print(f"Generated a total of {len(candidate_list)} unique candidate sequences.")


# --- 修正：定义用于预测的 ESM 模型名称 ---
# !!! 关键：这里必须使用与训练 rf_model 时相同的 ESM 模型 !!!
# 根据之前的日志，rf_model 是用 640 维嵌入训练的 (来自 esm2_t30_150M_UR50D)
PREDICTION_ESM_MODEL_NAME = "esm2_t6_8M_UR50D" # <--- 确认这个模型与训练时一致

print(f"尝试加载用于预测的 ESM 模型: {PREDICTION_ESM_MODEL_NAME}")
start_time = time.time()
try:
    # 加载指定用于预测的 ESM 模型和字母表
    esm_model_pred, alphabet_pred = esm.pretrained.load_model_and_alphabet(PREDICTION_ESM_MODEL_NAME)
    batch_converter_pred = alphabet_pred.get_batch_converter()
    # 重新确定设备 (优先使用 GPU)
    DEVICE_pred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model_pred.to(DEVICE_pred)
    esm_model_pred.eval() # 设置模型为评估模式 (关闭 dropout 等)
    print(f"用于预测的 ESM 模型 '{PREDICTION_ESM_MODEL_NAME}' 已加载到 {DEVICE_pred}，耗时 {time.time() - start_time:.2f} 秒。")
except Exception as e:
    print(f"加载 ESM 模型 {PREDICTION_ESM_MODEL_NAME} 时出错: {e}")
    print("请确保 'fair-esm' 已安装，模型名称正确，并且有足够的内存。")
    # 根据需要处理错误，例如退出
    exit()


# --- 6.1 为候选序列生成 ESM 嵌入 ---
print("\n使用预测模型为候选序列生成 ESM 嵌入...")
candidate_embeddings_np = np.array([]) # 初始化为空 numpy 数组

if candidate_list:
    # 确定批次大小 (如果使用GPU，可以尝试更大的批次)
    prediction_batch_size = CPU_BATCH_SIZE # 默认使用 CPU 批次大小 (来自之前的定义)
    if DEVICE_pred == torch.device("cuda"):
        # 如果是GPU，可以尝试更大的批次，例如 8, 16 或 32，取决于GPU内存和模型大小
        # 对于较大的模型如 150M，可能需要较小的批次大小
        prediction_batch_size = 8 # 示例值 (根据你的 GPU 内存调整，与训练时用的GPU_BATCH_SIZE可以不同)
        print(f"检测到 GPU，使用 GPU 批次大小: {prediction_batch_size}")
    else:
        print(f"使用 CPU 批次大小: {prediction_batch_size}")

    # *** 修复点：使用正确的函数名 'get_esm_embeddings' ***
    # 传递用于预测的模型、字母表、转换器和设备
    candidate_embeddings_tensor = get_esm_embeddings( # <-- 使用正确的函数名
        candidate_list,
        esm_model_pred,         # 使用预测模型
        alphabet_pred,        # 使用预测模型的字母表
        batch_converter_pred, # 使用预测模型的转换器
        DEVICE_pred,          # 使用预测模型所在的设备 (可能是 cuda 或 cpu)
        batch_size=prediction_batch_size # 使用调整后的批次大小
    )

    # 将嵌入结果转换为 NumPy 数组以用于 scikit-learn 模型
    # .cpu() 确保数据在 CPU 上，然后转换为 numpy
    candidate_embeddings_np = candidate_embeddings_tensor.cpu().numpy()

    print(f"候选序列嵌入的形状: {candidate_embeddings_np.shape}") # 检查维度是否正确 (应为 N x 640)

    # 检查是否有 NaN (如果嵌入过程中出错)
    if np.isnan(candidate_embeddings_np).any():
        print("警告：在候选序列嵌入中发现 NaN 值。将移除相应的候选序列。")
        nan_mask = np.isnan(candidate_embeddings_np).any(axis=1)
        # 需要同时过滤 candidate_list, mutation_list 和 embeddings
        # 使用列表推导式进行过滤
        original_count = len(candidate_list)
        candidate_list = [seq for i, seq in enumerate(candidate_list) if not nan_mask[i]]
        mutation_list = [mut for i, mut in enumerate(mutation_list) if not nan_mask[i]]
        candidate_embeddings_np = candidate_embeddings_np[~nan_mask]
        print(f"因 NaN 嵌入移除了 {original_count - len(candidate_list)} 个候选序列。")
        print(f"剩余候选序列数量: {len(candidate_list)}")
        print(f"过滤后的候选序列嵌入形状: {candidate_embeddings_np.shape}")

else:
    # candidate_embeddings_np 已经初始化为空数组
    print("上一步没有生成候选序列，跳过预测。")


# --- 6.2 预测亮度 ---
predicted_brightness = []
# 确保我们有有效的嵌入和候选列表来进行预测
# 并且嵌入的数量与候选列表的数量一致
if candidate_embeddings_np.shape[0] > 0 and len(candidate_list) == candidate_embeddings_np.shape[0]:
    print("\n正在为候选序列预测亮度...")
    try:
        # 使用加载的随机森林模型进行预测
        predicted_brightness = rf_model.predict(candidate_embeddings_np)
        print("预测完成。")
    except ValueError as ve: # 捕获更具体的 ValueError
        print(f"随机森林模型预测期间出错: {ve}")
        print("请确认用于预测的 ESM 模型生成的特征维度与训练 rf_model 时使用的维度一致。")
        # 如果预测失败，将结果列表清空，后续步骤会处理空结果
        predicted_brightness = []
    except Exception as e:
        print(f"随机森林模型预测期间发生未知错误: {e}")
        predicted_brightness = []
else:
    if not candidate_list:
         print("没有可用的候选序列进行预测。")
    elif candidate_embeddings_np.shape[0] == 0 and candidate_list:
         print("生成了候选序列，但未能生成有效的嵌入向量进行预测。")
    elif len(candidate_list) != candidate_embeddings_np.shape[0]:
         print("错误：经过 NaN 过滤后，候选序列数量与嵌入向量数量不匹配。")


# --- 6.3 组合结果并筛选 ---
final_candidates_formatted = pd.DataFrame() # 初始化为空 DataFrame

# 只有在成功生成预测并且数量与候选列表匹配时才继续
if len(candidate_list) > 0 and len(predicted_brightness) > 0 and len(candidate_list) == len(predicted_brightness):
    # 创建包含序列、突变和预测值的 DataFrame
    results_df = pd.DataFrame({
        'Sequence': candidate_list,
        'Mutations': mutation_list, # 确保 mutation_list 与 candidate_list 保持同步
        'PredictedBrightness': predicted_brightness
    })

    # 过滤掉排除列表中的序列
    print(f"\n根据排除列表 ({len(exclusion_sequences)} 个序列) 进行过滤...")
    initial_candidate_count = len(results_df)
    # 确保比较的是字符串类型
    results_df = results_df[~results_df['Sequence'].astype(str).isin(exclusion_sequences)]
    removed_count = initial_candidate_count - len(results_df)
    if removed_count > 0:
        print(f"移除了 {removed_count} 个在排除列表中的序列。")
    else:
        print("候选列表中的序列均不在排除列表中。")

    # 按预测亮度降序排序
    results_df = results_df.sort_values(by='PredictedBrightness', ascending=False)

    # 选择 Top N 个结果
    final_candidates = results_df.head(TOP_N_SELECT).copy() # 使用 .copy() 避免 SettingWithCopyWarning

    print(f"\n预测出的 Top {min(TOP_N_SELECT, len(final_candidates))} 个候选序列 (已排除):") # 显示实际选出的数量

    if not final_candidates.empty:
        # 为了更清晰地展示，可以添加一个 ID 列
        final_candidates.insert(0, 'Sequence ID', [f'Candidate_{i+1}' for i in range(len(final_candidates))])
        # 调整列顺序以符合提交格式要求
        final_candidates_formatted = final_candidates[['Sequence ID', 'Mutations', 'Sequence', 'PredictedBrightness']]
        # 使用 display 或 print 显示 DataFrame
        try:
            from IPython.display import display
            display(final_candidates_formatted) # 在 Jupyter 环境中友好显示
        except ImportError:
            print(final_candidates_formatted.to_string()) # 在非 IPython 环境中打印完整 DataFrame
    else:
        print("经过过滤和筛选后，没有剩余的候选序列。")

elif not candidate_list:
     print("\n没有生成候选序列或候选序列在预测前已被过滤掉。")
elif not predicted_brightness:
     print("\n预测步骤失败或没有产生结果。请检查之前的错误信息。")
else: # candidate_list 和 predicted_brightness 长度不匹配
     print("\n错误：最终候选序列数量与预测结果数量不匹配。无法继续处理。")

# --- 清理预测模型占用的内存 ---
print("\n清理预测模型内存...")
del esm_model_pred, alphabet_pred, batch_converter_pred
if 'candidate_embeddings_tensor' in locals():
    del candidate_embeddings_tensor
if 'candidate_embeddings_np' in locals():
    del candidate_embeddings_np
if DEVICE_pred == torch.device("cuda"):
    print("清空CUDA缓存...")
    torch.cuda.empty_cache()
print("预测模型清理完成。")


# --- 7.1 准备提交格式 (示例) ---
# 比赛要求提交 CSV，包含 'Sequence ID', 'Mutations', 'Full Sequence'
# 我们已经有了类似格式的 DataFrame 'final_candidates_formatted'

# 如果需要保存为 CSV 文件：
output_filename = "my_top_brightness_candidates.csv"
if not final_candidates_formatted.empty:
    # 选择需要的列
    submission_df = final_candidates_formatted[['Sequence ID', 'Mutations', 'Sequence']].copy()
    # 重命名列以完全匹配（如果需要）
    # submission_df.rename(columns={'Sequence': 'Full Sequence'}, inplace=True) # 假设需要 'Full Sequence' 列名
    submission_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved top {len(submission_df)} candidates to {output_filename}")
else:
    print("\nNo final candidates to save.")


# --- 7.2 后续步骤 ---
print("\n--- 教程已完成 ---")
print("后续可能的优化步骤：")
print("1.  **优化位置池：** 目前使用的数据量和批次非常小，您可以进行全面的文献/数据/结构分析，以创建更好的`候选位置池`。")
print("2.  **考虑稳定性：** 本教程仅关注亮度。你需要纳入预测/提高热稳定性的策略或模型（例如，使用PDB结构、已知的稳定突变）。这可能涉及多目标优化。")
print("3.  **改进模型：** 尝试不同的ESM模型、回归算法（例如梯度提升）、超参数调整，或更先进的技术，如微调ESM。")
print("4.  **代码打包：** 根据要求将代码整理成可运行的脚本或包（.zip）。包含一个README文件。")
print("5.  **设计原理：** 撰写一份清晰的文档，解释你的方法、选择特定位置/突变的原因以及所使用的方法。")
print("6.  **提交：** 准备最终的CSV文件，其中准确包含6条序列，确保它们不在排除列表中且符合突变限制。")