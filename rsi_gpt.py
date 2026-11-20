import json
import math
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch
import torch.nn as nn
import torch.nn.functional as F

from web_ui import BLOCK_SIZE, BATCH_SIZE, DEVICE, LEARNING_RATE, STOP_CHAR, MiniLLMEngine

base_dir = Path(__file__).parent
rsi_log_path = base_dir / "rsi_history.json"

# ==========================================
# RSI（再帰的自己改善）用の未知・応用プロンプト群
# ==========================================
# 既存の知識を組み合わせて答える応用質問（最初は高Temperatureだと破綻しやすい）
RSI_PROMPTS = [
    {"prompt": "こんにちは！元気？", "keywords": ["こんにちは", "元気", "話", "天気"]},
    {"prompt": "あなたの名前とできることを教えて", "keywords": ["ミニGPT", "会話", "質問", "AI", "ローカル"]},
    {"prompt": "LLMとAttentionについて教えて", "keywords": ["文字", "予測", "文脈", "関係", "言葉", "AI"]},
    {"prompt": "プログラミングで疲れたよ", "keywords": ["お疲れ", "無理", "休", "楽しく", "休憩"]},
    {"prompt": "お腹すいたし眠い", "keywords": ["お腹", "ラーメン", "カレー", "寝", "休", "布団"]},
    {"prompt": "Pythonで作られたAIなの？", "keywords": ["Python", "PyTorch", "AI", "ローカル", "ミニGPT"]},
    {"prompt": "今日も一日ありがとう！", "keywords": ["ありがとう", "どういたしまして", "お役に", "話しかけて"]},
]


class RSISelfImprover:
    """
    Recursive Self-Improvement (再帰的自己改善: STaR / Best-of-N Rejection Sampling方式)
    1. 自己生成 (Self-Generation): 1つの質問に対し複数の返答候補(N個)を探索生成
    2. 自己評価 (Self-Evaluation / Reward Model): 流暢さ・確信度・質問適合度で採点
    3. 厳選 (Selection): 閾値を超えた最高スコアの自己生成回答のみを合成データとして採用
    4. 再帰学習 (Recursive Fine-Tuning): 採用された合成データを加えて次世代モデルを学習
    """

    def __init__(self, engine: MiniLLMEngine):
        self.engine = engine

    @torch.no_grad()
    def generate_candidate(self, user_input: str, temperature: float = 0.75, max_new_tokens: int = 80) -> dict:
        """返答候補を1つ生成し、同時にモデルの自己確信度(平均確率)を計算する"""
        self.engine.model.eval()
        prompt = f"ユーザー: {user_input}\nAI: "
        idx = torch.tensor([self.engine.encode(prompt)], dtype=torch.long, device=DEVICE)

        generated_ids = []
        token_probs = []
        ended_cleanly = False

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self.engine.model(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            token_id = next_id.item()
            chosen_prob = probs[0, token_id].item()
            if token_id == self.engine.stop_token_id:
                ended_cleanly = True
                break
            generated_ids.append(token_id)
            token_probs.append(chosen_prob)
            idx = torch.cat((idx, next_id), dim=1)

        reply_text = self.engine.decode(generated_ids).strip()
        avg_confidence = sum(token_probs) / max(1, len(token_probs))
        return {
            "reply": reply_text,
            "avg_confidence": avg_confidence,
            "ended_cleanly": ended_cleanly,
        }

    def evaluate_candidate(self, prompt_info: dict, candidate: dict) -> dict:
        """
        自己評価関数（Reward / Critique Function）
        100点満点で生成候補の品質を採点し、理由（Critique）を返す
        """
        reply = candidate["reply"]
        score = 0.0
        reasons = []

        # 1. 完結性チェック（終了記号◁で正しく終わったか: 最大25点）
        if candidate["ended_cleanly"] and len(reply) > 0:
            score += 25.0
            reasons.append("完結性OK(+25)")
        else:
            reasons.append("未完結(0)")

        # 2. フォーマット・破綻チェック（「ユーザー:」や「AI:」が混入していないか、同じ文字の連続がないか: 最大25点）
        has_role_leak = ("ユーザー" in reply) or ("AI:" in reply) or ("\n" in reply)
        has_repetition = bool(re.search(r"(.)\1{3,}", reply))  # 同じ文字4連以上は減点
        if not has_role_leak and not has_repetition and len(reply) >= 12:
            score += 25.0
            reasons.append("文法・形式OK(+25)")
        else:
            if has_role_leak:
                reasons.append("役割混入(-)")
            if has_repetition:
                reasons.append("文字重複(-)")
            if len(reply) < 12:
                reasons.append("短すぎ(-)")

        # 3. 質問との意味的整合性（関連キーワードが含まれているか: 最大30点）
        matched_kw = [kw for kw in prompt_info["keywords"] if kw in reply]
        kw_score = min(30.0, len(matched_kw) * 15.0)
        score += kw_score
        if matched_kw:
            reasons.append(f"文脈適合:{','.join(matched_kw)}(+{int(kw_score)})")

        # 4. モデルの自己確信度（Confidence: 最大20点）
        conf_score = round(candidate["avg_confidence"] * 20.0, 1)
        score += conf_score
        reasons.append(f"確信度(+{conf_score})")

        return {
            "score": round(score, 1),
            "critique": " / ".join(reasons),
            "matched_keywords": matched_kw,
        }

    def run_rsi_cycle(
        self,
        generations: int = 3,
        candidates_per_prompt: int = 6,
        score_threshold: float = 75.0,
        steps_per_gen: int = 350,
    ) -> dict:
        """
        指定した世代数（Generations）だけ再帰的自己改善ループを回す
        """
        base_text = (base_dir / "chat_data.txt").read_text(encoding="utf-8").strip()
        synthetic_pairs = []  # 自己生成・厳選された新規会話ペア
        generation_logs = []

        for gen in range(1, generations + 1):
            gen_candidates_log = []
            new_accepted_in_gen = 0
            total_best_scores = []

            # 高温(探索温度)を世代が進むにつれて少しずつ落ち着かせる
            explore_temp = max(0.45, 0.85 - (gen - 1) * 0.12)

            for p_info in RSI_PROMPTS:
                prompt_str = p_info["prompt"]
                candidates = []
                for _ in range(candidates_per_prompt):
                    cand = self.generate_candidate(prompt_str, temperature=explore_temp)
                    eval_res = self.evaluate_candidate(p_info, cand)
                    cand.update(eval_res)
                    candidates.append(cand)

                # N個の候補からスコア最大の「Best-of-N」を選出
                candidates.sort(key=lambda x: x["score"], reverse=True)
                best = candidates[0]
                worst = candidates[-1]
                total_best_scores.append(best["score"])

                accepted = False
                qa_entry = f"ユーザー: {prompt_str}\nAI: {best['reply']}{STOP_CHAR}"
                if best["score"] >= score_threshold and qa_entry not in base_text:
                    # 同じ質問の古い合成データがあれば、より高スコアな回答で上書き進化させる
                    existing_idx = next(
                        (i for i, item in enumerate(synthetic_pairs) if item["prompt"] == prompt_str),
                        None,
                    )
                    if existing_idx is None:
                        synthetic_pairs.append(
                            {"prompt": prompt_str, "reply": best["reply"], "score": best["score"], "entry": qa_entry}
                        )
                        accepted = True
                        new_accepted_in_gen += 1
                    elif best["score"] >= synthetic_pairs[existing_idx]["score"]:
                        synthetic_pairs[existing_idx] = {
                            "prompt": prompt_str,
                            "reply": best["reply"],
                            "score": best["score"],
                            "entry": qa_entry,
                        }
                        accepted = True

                gen_candidates_log.append(
                    {
                        "prompt": prompt_str,
                        "best_reply": best["reply"],
                        "best_score": best["score"],
                        "best_critique": best["critique"],
                        "worst_reply": worst["reply"],
                        "worst_score": worst["score"],
                        "accepted": accepted,
                    }
                )

            avg_score = round(sum(total_best_scores) / len(total_best_scores), 1)

            # 厳選された自己生成データを元データに結合して次世代モデルを再帰学習！
            combined_text = base_text
            if synthetic_pairs:
                synth_block = "\n".join(item["entry"] for item in synthetic_pairs)
                combined_text = base_text + "\n" + synth_block

            (base_dir / "chat_data.txt").write_text(combined_text + "\n", encoding="utf-8")
            train_info = self.engine.train_model(steps=steps_per_gen)
            last_loss = train_info["history"][-1]["loss"]

            generation_logs.append(
                {
                    "generation": gen,
                    "avg_best_score": avg_score,
                    "new_accepted": new_accepted_in_gen,
                    "total_synthetic": len(synthetic_pairs),
                    "final_loss": last_loss,
                    "details": gen_candidates_log,
                }
            )

        summary = {
            "generations": generation_logs,
            "synthetic_data_count": len(synthetic_pairs),
            "synthetic_pairs": synthetic_pairs,
        }
        rsi_log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


if __name__ == "__main__":
    print("=" * 65)
    print("🧬 自作ミニGPT：RSI（Recursive Self-Improvement / 再帰的自己改善）開始")
    print("=" * 65)
    engine = MiniLLMEngine()
    improver = RSISelfImprover(engine)

    result = improver.run_rsi_cycle(generations=3, candidates_per_prompt=6, score_threshold=75.0, steps_per_gen=350)

    for g in result["generations"]:
        print(
            f"\n【第 {g['generation']} 世代 (Gen {g['generation']})】 "
            f"平均自己評価スコア: {g['avg_best_score']}点 / 100点 | "
            f"新規採用データ: +{g['new_accepted']}件 (累計 {g['total_synthetic']}件) | "
            f"再帰学習Loss: {g['final_loss']}"
        )
        print("-" * 65)
        for d in g["details"][:3]:  # 代表3件を表示
            mark = "✅採用" if d["accepted"] else "⏸見送り"
            print(f"❓ 質問: 「{d['prompt']}」 [{mark}]")
            print(f"   ❌ 棄却された悪い生成例 ({d['worst_score']}点): {d['worst_reply'][:45]}...")
            print(f"   🏆 厳選された最良生成例 ({d['best_score']}点): {d['best_reply']}")
            print(f"      └ 評価理由: {d['best_critique']}")

    print("\n" + "=" * 65)
    print(f"✨ RSI完了！合計 {result['synthetic_data_count']} 件の自己生成データが chat_data.txt に追加され、モデルが進化しました！")
    print("=" * 65)
