---
language:
  - ja
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - gpt
  - transformer
  - from-scratch
  - japanese
  - educational
---

# 🤖 Mini-GPT (フルスクラッチ日本語ミニLLM)

Python と PyTorch だけでゼロから実装した、学習・実験用の小規模な日本語 GPT（Decoder-only Transformer）モデル（約90万パラメータ）です。

<p>
  <a href="https://huggingface.co/spaces/koheip/mini-gpt-ja-demo"><img alt="Open in Spaces" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg"></a>
  <a href="https://github.com/koheip/mini-llm"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-koheip%2Fmini--llm-181717?logo=github"></a>
</p>

- 🎮 **デモ（ブラウザで会話）**: https://huggingface.co/spaces/koheip/mini-gpt-ja-demo
- 💻 **GitHub**: https://github.com/koheip/mini-llm

## ✨ 特徴

1. **約150行で読める Transformer 実装 (`mini_gpt.py`)**
   - 文字単位トークナイザー（Character-level Tokenizer）
   - トークン埋め込み & 位置埋め込み（Token & Positional Embedding）
   - Causal Multi-Head Self-Attention（未来の文字をマスクする自己注意機構）
   - Transformer Block（LayerNorm + Attention + MLP + 残差接続）
   - 自己回帰による文章生成（Autoregressive Generation）
2. **会話チューニング＆対話モード (`chat_gpt.py`)**
   - `ユーザー: ...\nAI: ...◁` 形式の対話データと終了記号（Stop Token `◁`）を用いたチャットボット化
3. **ブラウザで遊べるローカル Web UI (`web_ui.py`)**
   - 追加ライブラリ不要（Python標準の `http.server` のみ）で動作
   - 最初の1文字目の予測確率トップ5の可視化
   - ブラウザ上での会話データ編集＆その場での再学習機能

---

## 🚀 使い方

### 1. インストール

```bash
pip install -r requirements.txt
```

### 2. 小説テキストの事前学習・文章生成を試す

```bash
python mini_gpt.py
```

### 3. 会話データの学習・ターミナル対話を試す

```bash
# 学習と会話テストの実行（chat_model.pt が保存されます）
python chat_gpt.py

# 対話モードで起動
python chat_gpt.py --chat
```

### 4. ローカル Web UI を起動する

```bash
python web_ui.py
```
起動後、ブラウザで `http://localhost:8765` が自動的に開きます（Windows環境では `start_webui.bat` のダブルクリックでも起動できます）。

---

## 📁 ファイル構成

- `mini_gpt.py` : 最小構成の日本語ミニGPT（小説テキストの事前学習・生成）
- `sample_data.txt` : 事前学習用の日本語文学サンプルテキスト
- `chat_gpt.py` : 会話データ学習・ターミナル対話スクリプト
- `chat_data.txt` : 一問一答の日本語会話データセット
- `chat_model.pt` : 学習済みのモデル重みファイル（PyTorch state_dict）
- `web_ui.py` : ローカル Web UI サーバー（チャット・確率可視化・データ編集＆再学習）

---

## 🔗 リンク

- 🎮 デモ Space: https://huggingface.co/spaces/koheip/mini-gpt-ja-demo
- 💻 GitHub: https://github.com/koheip/mini-llm
- 👤 作者: [koheip](https://huggingface.co/koheip) / [ポートフォリオ](https://huggingface.co/spaces/koheip/kohei-portfolio)

> 約90万パラメータの学習・実験用モデルです。学習データにない話題には的外れな返答になります。
