import math
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ハイパーパラメータ設定
# ==========================================
BATCH_SIZE = 32
BLOCK_SIZE = 96       # 会話1往復が収まる文脈長
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.05
LEARNING_RATE = 2e-3
MAX_STEPS = 1000
STOP_CHAR = "◁"       # AIが「ここで返答終わり！」と判断する終了記号（Stop Token）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = Path(__file__).parent
data_path = base_dir / "chat_data.txt"
model_path = base_dir / "chat_model.pt"

text = data_path.read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
stop_token_id = stoi[STOP_CHAR]

def encode(s: str) -> list[int]:
    # 学習データにない未知の文字はスキップしてエラーを防ぐ
    return [stoi[c] for c in s if c in stoi]

def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=DEVICE)
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# ==========================================
# モデル定義（MiniGPTと同じTransformer構造）
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        head_dim = C // N_HEAD
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ChatMiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def reply(self, user_input: str, max_new_tokens: int = 80, temperature: float = 0.3) -> str:
        """ユーザーの質問を受け取り、終了記号(◁)が出るまでAIの返答を生成する"""
        self.eval()
        prompt = f"ユーザー: {user_input}\nAI: "
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
        generated_ids = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            token_id = next_id.item()
            if token_id == stop_token_id:
                break  # 終了記号(◁)が出たら発言を終了！
            generated_ids.append(token_id)
            idx = torch.cat((idx, next_id), dim=1)

        self.train()
        return decode(generated_ids).strip()

# ==========================================
# 学習・保存・対話モード
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    model = ChatMiniGPT().to(DEVICE)

    if "--chat" in sys.argv and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print("保存済みの学習モデル (chat_model.pt) を読み込みました！")
    else:
        print(f"会話データ ({len(text)}文字) の学習を開始します...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        for step in range(1, MAX_STEPS + 1):
            xb, yb = get_batch()
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == 1:
                print(f"Step {step:4d}/{MAX_STEPS} | Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), model_path)
        print(f"\n学習完了！モデルを {model_path.name} に保存しました。\n" + "=" * 50)

        # 自動テスト会話
        test_questions = [
            "こんにちは",
            "あなたは誰ですか？",
            "LLMって何？",
            "なぜ会話ができるの？",
            "疲れた",
            "ありがとう",
        ]
        print("【自作LLMとの会話テスト】")
        for q in test_questions:
            ans = model.reply(q)
            print(f"👤 ユーザー: {q}")
            print(f"🤖 ミニGPT : {ans}\n")

    if "--chat" in sys.argv:
        print("=" * 50)
        print("💬 対話モード開始！（終了するには q または exit と入力してください）")
        print("=" * 50)
        while True:
            try:
                user_msg = input("\n👤 あなた: ").strip()
            except EOFError:
                break
            if not user_msg or user_msg.lower() in ("q", "exit", "quit", "終了"):
                print("🤖 ミニGPT: さようなら！またお話ししましょう！")
                break
            print(f"🤖 ミニGPT: {model.reply(user_msg)}")
