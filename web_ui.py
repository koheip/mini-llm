import json
import math
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ハイパーパラメータ設定
# ==========================================
BATCH_SIZE = 32
BLOCK_SIZE = 96
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.05
LEARNING_RATE = 2e-3
STOP_CHAR = "◁"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 8000

base_dir = Path(__file__).parent
data_path = base_dir / "chat_data.txt"
model_path = base_dir / "chat_model.pt"

# ==========================================
# Transformer モデル定義
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
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
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
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ==========================================
# エンジンクラス（文字辞書・学習・推論を管理）
# ==========================================
class MiniLLMEngine:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.stop_token_id = 0
        self.model = None
        self.load_or_init()

    def build_vocab(self, text: str):
        if STOP_CHAR not in text:
            text += STOP_CHAR
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.stop_token_id = self.stoi[STOP_CHAR]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.itos[i] for i in ids if i in self.itos])

    def load_or_init(self):
        text = data_path.read_text(encoding="utf-8")
        self.build_vocab(text)
        self.model = ChatMiniGPT(self.vocab_size).to(DEVICE)
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location=DEVICE, weights_only=True)
                self.model.load_state_dict(state)
                print(f"学習済みモデル ({model_path.name}) を読み込みました。")
                return
            except Exception:
                pass
        print("初期モデルの学習を実行します...")
        self.train_model(steps=800)

    def train_model(self, steps: int = 800) -> dict:
        text = data_path.read_text(encoding="utf-8")
        self.build_vocab(text)
        self.model = ChatMiniGPT(self.vocab_size).to(DEVICE)
        data = torch.tensor(self.encode(text), dtype=torch.long, device=DEVICE)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        history = []
        self.model.train()
        for step in range(1, steps + 1):
            ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=DEVICE)
            xb = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
            yb = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
            _, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 200 == 0 or step == steps:
                history.append({"step": step, "loss": round(loss.item(), 4)})

        torch.save(self.model.state_dict(), model_path)
        return {"vocab_size": self.vocab_size, "history": history}

    @torch.no_grad()
    def chat(self, user_input: str, temperature: float = 0.3, max_new_tokens: int = 80) -> dict:
        self.model.eval()
        prompt = f"ユーザー: {user_input}\nAI: "
        encoded_prompt = self.encode(prompt)
        unknown_chars = [c for c in user_input if c not in self.stoi]

        idx = torch.tensor([encoded_prompt], dtype=torch.long, device=DEVICE)
        generated_ids = []
        first_token_candidates = []

        temp = max(0.05, float(temperature))
        for step_idx in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self.model(idx_cond)
            next_logits = logits[:, -1, :] / temp
            probs = F.softmax(next_logits, dim=-1)

            # 最初の1文字目を予測する際のトップ候補を記録（仕組みの可視化用）
            if step_idx == 0:
                top_probs, top_indices = torch.topk(probs[0], k=min(5, self.vocab_size))
                for p, tid in zip(top_probs.tolist(), top_indices.tolist()):
                    ch = self.itos[tid]
                    label = "[終了記号]" if ch == STOP_CHAR else ("\\n" if ch == "\n" else ch)
                    first_token_candidates.append({"char": label, "prob": round(p * 100, 1)})

            next_id = torch.multinomial(probs, num_samples=1)
            token_id = next_id.item()
            if token_id == self.stop_token_id:
                break
            generated_ids.append(token_id)
            idx = torch.cat((idx, next_id), dim=1)

        return {
            "reply": self.decode(generated_ids).strip(),
            "unknown_chars": unknown_chars,
            "first_candidates": first_token_candidates,
        }


engine = MiniLLMEngine()

# ==========================================
# ブラウザ用 HTML / CSS / JS (単一ファイル内蔵)
# ==========================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>自作ミニGPT ローカルWeb UI</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text: #f8fafc;
    --subtext: #94a3b8;
    --user-msg: #0284c7;
    --ai-msg: #334155;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", "Hiragino Sans", "Meiryo", sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 14px 24px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  header h1 { font-size: 1.2rem; display: flex; align-items: center; gap: 10px; }
  .badge {
    font-size: 0.78rem;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.3);
  }
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 420px;
    overflow: hidden;
  }
  /* 左側：チャット画面 */
  .chat-section {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    overflow: hidden;
  }
  .chat-log {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg-row { display: flex; flex-direction: column; max-width: 80%; }
  .msg-row.user { align-self: flex-end; align-items: flex-end; }
  .msg-row.ai { align-self: flex-start; align-items: flex-start; }
  .msg-label { font-size: 0.75rem; color: var(--subtext); margin-bottom: 4px; }
  .msg-bubble {
    padding: 12px 16px;
    border-radius: 14px;
    line-height: 1.5;
    font-size: 0.95rem;
    white-space: pre-wrap;
  }
  .msg-row.user .msg-bubble { background: var(--user-msg); color: #fff; border-bottom-right-radius: 4px; }
  .msg-row.ai .msg-bubble { background: var(--ai-msg); color: var(--text); border-bottom-left-radius: 4px; }
  .msg-meta {
    margin-top: 6px;
    font-size: 0.75rem;
    color: var(--subtext);
    background: rgba(255,255,255,0.04);
    padding: 6px 10px;
    border-radius: 8px;
  }
  .chips {
    padding: 10px 20px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    border-top: 1px solid var(--border);
    background: rgba(30, 41, 59, 0.5);
  }
  .chip {
    font-size: 0.8rem;
    background: var(--panel);
    color: var(--accent);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 999px;
    cursor: pointer;
    transition: 0.15s;
  }
  .chip:hover { background: var(--border); }
  .input-bar {
    padding: 14px 20px;
    background: var(--panel);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
  }
  .input-bar input {
    flex: 1;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: 0.95rem;
    outline: none;
  }
  .input-bar input:focus { border-color: var(--accent); }
  button {
    padding: 10px 18px;
    border-radius: 10px;
    border: none;
    background: var(--accent);
    color: #0f172a;
    font-weight: 700;
    cursor: pointer;
    transition: 0.15s;
  }
  button:hover { background: var(--accent-hover); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  /* 右側：設定・学習データ編集パネル */
  .side-panel {
    background: var(--panel);
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow-y: auto;
  }
  .card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .card h2 { font-size: 0.95rem; color: var(--accent); }
  .card p { font-size: 0.8rem; color: var(--subtext); line-height: 1.4; }
  .slider-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; }
  textarea {
    width: 100%;
    height: 260px;
    background: #0b1120;
    color: #e2e8f0;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-family: Consolas, monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    resize: vertical;
  }
  .status-box {
    font-size: 0.8rem;
    color: #4ade80;
    min-height: 1.2em;
    white-space: pre-line;
  }
</style>
</head>
<body>
<header>
  <h1>🤖 自作ミニGPT ローカルWeb UI</h1>
  <span class="badge">PyTorch フルスクラッチ実装 (約90万パラメータ)</span>
</header>

<main>
  <!-- 左側：チャットエリア -->
  <section class="chat-section">
    <div class="chat-log" id="chatLog">
      <div class="msg-row ai">
        <span class="msg-label">🤖 ミニGPT</span>
        <div class="msg-bubble">こんにちは！あなたのPCの中で動いている自作ミニGPTです。下のボタンをクリックするか、自由に話しかけてみてください！</div>
      </div>
    </div>

    <div class="chips">
      <span class="chip" onclick="sendPreset('こんにちは')">こんにちは</span>
      <span class="chip" onclick="sendPreset('あなたは誰ですか？')">あなたは誰ですか？</span>
      <span class="chip" onclick="sendPreset('LLMって何？')">LLMって何？</span>
      <span class="chip" onclick="sendPreset('Attentionって何？')">Attentionって何？</span>
      <span class="chip" onclick="sendPreset('疲れた')">疲れた</span>
      <span class="chip" onclick="sendPreset('おすすめのご飯は？')">おすすめのご飯は？</span>
    </div>

    <form class="input-bar" onsubmit="handleSend(event)">
      <input type="text" id="userInput" placeholder="メッセージを入力（例：こんにちは、LLMって何？ など）" autocomplete="off">
      <button type="submit" id="sendBtn">送信</button>
    </form>
  </section>

  <!-- 右側：パラメータ＆学習データ編集パネル -->
  <aside class="side-panel">
    <div class="card">
      <h2>⚙️ 生成パラメータ (Temperature)</h2>
      <div class="slider-row">
        <span>ランダム性（創造性）: <strong id="tempVal">0.3</strong></span>
        <input type="range" id="tempSlider" min="0.1" max="1.2" step="0.1" value="0.3"
               oninput="document.getElementById('tempVal').textContent = this.value">
      </div>
      <p>低いほど学習データに忠実に答え、高いほど色んな文字をランダムに組み合わせます。</p>
    </div>

    <div class="card" style="flex:1;">
      <h2>📚 学習データ編集 & その場で再学習</h2>
      <p>下の会話データ（<code>chat_data.txt</code>）に新しい会話や好きなキャラ口調を書き足して「保存して再学習」を押すと、約10秒で新しい会話を覚えます！（AIの返答の最後には必ず <code>◁</code> を付けてください）</p>
      <textarea id="dataEditor" spellcheck="false"></textarea>
      <button id="trainBtn" onclick="retrainModel()">🔄 保存して再学習する（約10秒）</button>
      <div class="status-box" id="trainStatus"></div>
    </div>
  </aside>
</main>

<script>
const chatLog = document.getElementById('chatLog');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const dataEditor = document.getElementById('dataEditor');
const trainBtn = document.getElementById('trainBtn');
const trainStatus = document.getElementById('trainStatus');

async function loadData() {
  const res = await fetch('/api/data');
  const data = await res.json();
  dataEditor.value = data.text;
}
loadData();

function sendPreset(text) {
  userInput.value = text;
  handleSend(new Event('submit'));
}

function appendMessage(role, text, metaHtml = '') {
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const label = document.createElement('span');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? '👤 あなた' : '🤖 ミニGPT';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  row.appendChild(label);
  row.appendChild(bubble);
  if (metaHtml) {
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.innerHTML = metaHtml;
    row.appendChild(meta);
  }
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function handleSend(e) {
  e.preventDefault();
  const msg = userInput.value.trim();
  if (!msg) return;
  userInput.value = '';
  appendMessage('user', msg);
  sendBtn.disabled = true;

  try {
    const temp = parseFloat(document.getElementById('tempSlider').value);
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, temperature: temp})
    });
    const data = await res.json();
    let meta = '';
    if (data.first_candidates && data.first_candidates.length > 0) {
      const cands = data.first_candidates.map(c => `「${c.char}」(${c.prob}%)`).join(' / ');
      meta += `🔍 最初の1文字目の予測確率トップ: ${cands}`;
    }
    if (data.unknown_chars && data.unknown_chars.length > 0) {
      meta += `<br>⚠️ 未学習の文字（無視されました）: ${data.unknown_chars.join(', ')}`;
    }
    appendMessage('ai', data.reply || '（返答が空でした）', meta);
  } catch (err) {
    appendMessage('ai', 'エラーが発生しました: ' + err);
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

async function retrainModel() {
  trainBtn.disabled = true;
  trainStatus.style.color = '#38bdf8';
  trainStatus.textContent = '⏳ 学習データを保存し、モデルを再学習しています（約10〜15秒お待ちください）...';
  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: dataEditor.value, steps: 800})
    });
    const data = await res.json();
    const lastLoss = data.history[data.history.length - 1].loss;
    trainStatus.style.color = '#4ade80';
    trainStatus.textContent = `✅ 再学習完了！ (文字種類数: ${data.vocab_size}文字 / 最終誤差 Loss: ${lastLoss})\n左のチャットからさっそく新しい会話を試せます！`;
  } catch (err) {
    trainStatus.style.color = '#f87171';
    trainStatus.textContent = 'エラー: ' + err;
  } finally {
    trainBtn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class WebUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # コンソール出力を 깔끔に保つ

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            text = data_path.read_text(encoding="utf-8")
            self._send_json({"text": text})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        req = json.loads(raw)

        if self.path == "/api/chat":
            msg = req.get("message", "")
            temp = float(req.get("temperature", 0.3))
            result = engine.chat(msg, temperature=temp)
            self._send_json(result)
        elif self.path == "/api/train":
            new_text = req.get("text", "").strip()
            steps = int(req.get("steps", 800))
            if new_text:
                data_path.write_text(new_text + "\n", encoding="utf-8")
            info = engine.train_model(steps=steps)
            self._send_json(info)
        else:
            self.send_error(404)


if __name__ == "__main__":
    server = None
    for candidate_port in (8765, 8080, 7860, 5000, 0):
        try:
            server = HTTPServer(("127.0.0.1", candidate_port), WebUIHandler)
            break
        except OSError:
            continue
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}"
    print("=" * 55, flush=True)
    print(f"🚀 ミニGPT Web UI サーバーを起動しました！", flush=True)
    print(f"🌐 ブラウザで開くURL: {url}", flush=True)
    print("（終了するにはこのウィンドウを閉じるか Ctrl+C を押してください）", flush=True)
    print("=" * 55, flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")
