#!/usr/bin/env python3
"""
RAGAS 指标计算 — MiMo-V2.5-Pro 作为 judge
先跑 AnswerRelevancy + ContextPrecision + ContextRecall（较快）
Faithfulness 后续单独跑
"""
import json, os, sys, time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'

MIMO_KEY = "tp-c19uoslvrvgivdcr0qzj6hny880ztkl7zerhf33l4wbdijy6"
MIMO_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"

RESULTS_DIR = '/root/autodl-tmp/.autodl/小论文所需所有文件/results'

RAG_EXPERIMENTS = {
    'E2': 'exp2_bm25_300.json',
    'E3': 'exp3_naive_rag_300.json',
    'E4': 'exp4_hybrid_rag_300.json',
    'E5': 'exp5_iakg_rag_300.json',
    'E6': 'ablation_no_irir.json',
}


def main():
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import AnswerRelevancy, LLMContextPrecisionWithoutReference, ContextRecall
    from ragas import evaluate
    from sentence_transformers import SentenceTransformer

    # ── LLM: MiMo-V2.5-Pro (enable_thinking=false 关闭推理) ──
    openai_client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)
    _original_create = openai_client.chat.completions.create
    def patched_create(*args, **kwargs):
        kwargs.pop('n', None)
        kwargs['extra_body'] = {'enable_thinking': False}
        return _original_create(*args, **kwargs)
    openai_client.chat.completions.create = patched_create

    llm = llm_factory(MIMO_MODEL, client=openai_client)

    # ── Embeddings: 本地 BGE-large-zh ──
    class LocalBGE:
        def __init__(self):
            self.model = SentenceTransformer(
                '/root/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116'
            )
        def embed_query(self, text):
            return self.model.encode(text, normalize_embeddings=True).tolist()
        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()

    embeddings = LangchainEmbeddingsWrapper(LocalBGE())

    print("=" * 60)
    print("RAGAS 指标计算 (MiMo-V2.5-Pro, enable_thinking=false)")
    print("指标: AnswerRelevancy + ContextPrecision + ContextRecall")
    print("=" * 60)

    all_results = {}

    for exp_name, filename in RAG_EXPERIMENTS.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\n⚠️ {exp_name}: 不存在，跳过")
            continue

        with open(filepath) as f:
            data = json.load(f)
        details = data['details']

        print(f"\n{'─' * 40}")
        print(f"📊 {exp_name}: {len(details)} 条")

        samples = []
        skipped = 0
        for d in details:
            retrieved = d.get('retrieved_contexts') or []
            if not retrieved:
                skipped += 1
                continue
            if isinstance(retrieved[0], dict):
                retrieved = [r.get('content', str(r)) for r in retrieved]
            samples.append(SingleTurnSample(
                user_input=d.get('question', ''),
                response=d.get('full_answer', ''),
                retrieved_contexts=retrieved,
                reference=d.get('reference', ''),
                reference_contexts=d.get('reference_contexts', []),
            ))

        if skipped > 0:
            print(f"  跳过 {skipped} 条")
        if not samples:
            print(f"  ⚠️ 无有效样本")
            continue

        print(f"  有效样本: {len(samples)} 条")

        metrics = [
            AnswerRelevancy(),
            LLMContextPrecisionWithoutReference(),
            ContextRecall(),
        ]

        t0 = time.time()
        try:
            ds = EvaluationDataset(samples=samples)
            result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=embeddings)
            # 提取分数
            df = result.to_pandas()
            scores = {}
            for col in df.columns:
                if col not in ['user_input', 'response', 'retrieved_contexts', 'reference', 'reference_contexts']:
                    val = df[col].dropna().tolist()
                    if val and isinstance(val[0], (int, float)):
                        scores[col] = sum(val) / len(val)
            elapsed = time.time() - t0
            print(f"  ✅ 完成 ({elapsed:.0f}s):")
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")
            all_results[exp_name] = scores
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}

    # ── 保存 ──
    output_file = os.path.join(RESULTS_DIR, 'ragas_scores.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"{'实验':<8} {'Relevancy':>12} {'Precision':>12} {'Recall':>12}")
    print(f"{'─' * 60}")
    for exp, s in all_results.items():
        if 'error' in s:
            print(f"{exp:<8} {'ERROR':>12}")
        else:
            def g(k):
                for candidate in [k, k.lower()]:
                    v = s.get(candidate, None)
                    if v is not None and isinstance(v, (int, float)):
                        return f"{v:.4f}"
                return 'N/A'
            print(f"{exp:<8} {g('answer_relevancy'):>12} {g('llm_context_precision_without_reference'):>12} {g('context_recall'):>12}")

    print(f"\n✅ 完成，结果: {output_file}")


if __name__ == '__main__':
    main()
