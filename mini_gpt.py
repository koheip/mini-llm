import math
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ハイパーパラメータ設定
# ==========================================
BATCH_SIZE = 32       # 1度に処理する文の数
BLOCK_SIZE = 64       # 最大の文脈長（何文字先まで見渡すか）
N_EMBD = 128          # 埋め込みベクトルの次元数
N_HEAD = 4            # Attentionヘッドの数
N_LAYER = 4           # Transformerブロックの層数
DROPOUT = 0.1         # ドロップアウト率
LEARNING_RATE = 1e-3  # 学習率
MAX_STEPS = 800       # 学習ステップ数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. トークナイザー（文字 ↔ 数値IDの変換）
# ==========================================
data_path = Path(__file__).parent / "sample_data.txt"
text = data_path.read_text(encoding="utf-8")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)

def get_batch():
    """学習データからランダムにバッチを切り出す（入力xと、1文字ずらした正解y）"""
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=DEVICE)
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# ==========================================
# 3. Causal Self-Attention（自己注意機構）
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        # Query, Key, Value を一括計算する線形層
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        # 未来の文字をカンニングしないための下三角マスク
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time(文字数), Channels(次元数)
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        head_dim = C // N_HEAD
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2)

        # Attentionスコア計算: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # 注目度に応じてValueを集約
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# ==========================================
# 4. Transformer Block & MiniGPT モデル全体
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差接続 + Attention
        x = x + self.mlp(self.ln2(x))   # 残差接続 + FeedForward
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. 埋め込み層（文字IDの埋め込み ＋ 位置情報の埋め込み）
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """プロンプト(idx)から1文字ずつ次の文字を予測してつなげる"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        self.train()
        return idx

# ==========================================
# 5. 学習（Training）と文章生成（Inference）
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"使用デバイス: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
    print(f"学習データ文字数: {len(text)} 文字 / 語彙数（文字種類数）: {vocab_size} 文字")

    model = MiniGPT().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {num_params:,} ({num_params/1e6:.2f}M)\n")

    # 学習前の状態で生成してみる（まだランダムな文字列になる）
    prompt = "吾輩は"
    start_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    print("【学習前の生成結果】")
    print(decode(model.generate(start_ids, max_new_tokens=50)[0].tolist()))
    print("-" * 50)

    # 学習ループ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, MAX_STEPS + 1):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == 1:
            print(f"Step {step:4d}/{MAX_STEPS} | Loss: {loss.item():.4f}")

    print("-" * 50)
    print("【学習後の生成結果】")
    for p in ["吾輩は", "メロスは", "ある日の"]:
        p_ids = torch.tensor([encode(p)], dtype=torch.long, device=DEVICE)
        out = decode(model.generate(p_ids, max_new_tokens=80, temperature=0.7)[0].tolist())
        print(f"▶ プロンプト「{p}」:\n{out}\n")
