"""
Hybrid RAG 基线：BM25 + 向量检索 加权融合
融合公式: final_score = α × norm(BM25_score) + β × norm(vector_score)
参数: α=0.5, β=0.5
"""
import json
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from embed_clauses import VectorRetriever

# ── 配置 ──
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"
TEST_SET = '/root/autodl-tmp/.autodl/测试集.json'
OUTPUT_FILE = '/root/autodl-tmp/.autodl/baseline_hybrid_rag_100.json'
INDEX_DIR = '/root/.openclaw/workspace/projects/kg-rag-solas/index'
CLAUSES_JSON = '/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json'


class HybridRetriever:
    """Hybrid RAG 检索器：BM25 + 向量 加权融合"""

    def __init__(self, alpha=0.5, beta=0.5):
        from rank_bm25 import BM25Okapi
        import jieba

        self.alpha = alpha
        self.beta = beta

        # 加载条款
        with open(CLAUSES_JSON, 'r', encoding='utf-8') as f:
            self.clauses = json.load(f)

        # 构建 BM25 索引
        print("构建 BM25 索引...")
        tokenized_corpus = []
        for c in self.clauses:
            text = f"{c['clause_id']} {c['title']} {c['content']}"
            tokens = list(jieba.cut(text))
            tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  BM25 索引: {len(self.clauses)} 条")

        # 向量检索器
        self.vector_ret = VectorRetriever()
        print(f"  向量检索器就绪")

        print("✅ Hybrid RAG 检索器就绪\n")

    def search(self, query: str, top_k: int = 15) -> list:
        import jieba

        # ── BM25 检索 ──
        query_tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(query_tokens)

        # 取 top 50 候选（比最终 top_k 多取一些用于融合）
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:50]
        bm25_results = {}
        bm25_max = float(bm25_scores[bm25_top_indices[0]]) if len(bm25_top_indices) > 0 else 1.0
        if bm25_max == 0:
            bm25_max = 1.0
        for idx in bm25_top_indices:
            cid = self.clauses[idx]['clause_id']
            bm25_results[cid] = {
                'clause_id': cid,
                'title': self.clauses[idx]['title'],
                'content': self.clauses[idx]['content'],
                'bm25_score': float(bm25_scores[idx]),
                'bm25_norm': float(bm25_scores[idx]) / bm25_max,  # Min-Max 归一化
            }

        # ── 向量检索 ──
        vector_results_raw = self.vector_ret.search(query, top_k=50, threshold=0.1)
        vector_results = {}
        vec_max = max([r['score'] for r in vector_results_raw], default=1.0)
        if vec_max == 0:
            vec_max = 1.0
        for r in vector_results_raw:
            cid = r['clause_id']
            vector_results[cid] = {
                'vector_score': r['score'],
                'vector_norm': r['score'] / vec_max,  # Min-Max 归一化
            }
            # 补充完整内容（向量检索有完整 content）
            if cid in bm25_results:
                if len(r.get('content', '')) > len(bm25_results[cid].get('content', '')):
                    bm25_results[cid]['content'] = r['content']

        # ── 融合 ──
        all_ids = set(bm25_results.keys()) | set(vector_results.keys())
        merged = []
        for cid in all_ids:
            item = bm25_results.get(cid, {
                'clause_id': cid,
                'title': '',
                'content': vector_results.get(cid, {}).get('content', ''),
                'bm25_score': 0.0,
                'bm25_norm': 0.0,
            })
            vec = vector_results.get(cid, {'vector_score': 0.0, 'vector_norm': 0.0})

            bm25_n = item.get('bm25_norm', 0.0)
            vec_n = vec.get('vector_norm', 0.0)

            # 加权融合
            final_score = self.alpha * bm25_n + self.beta * vec_n

            merged.append({
                'clause_id': cid,
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'bm25_score': item.get('bm25_score', 0.0),
                'bm25_norm': bm25_n,
                'vector_score': vec.get('vector_score', 0.0),
                'vector_norm': vec_n,
                'final_score': final_score,
            })

        # 按融合分数排序
        merged.sort(key=lambda x: x['final_score'], reverse=True)
        return merged[:top_k]


class HybridRAG:
    """Hybrid RAG 完整 pipeline"""

    def __init__(self):
        self.retriever = HybridRetriever(alpha=0.5, beta=0.5)
        self.llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print("✅ Hybrid RAG 基线就绪\n")

    def build_prompt(self, query: str, clauses: list) -> str:
        clause_context = ""
        for i, r in enumerate(clauses):
            clause_context += f"\n### 条款 {i+1}: {r['clause_id']}\n"
            clause_context += f"标题: {r['title']}\n"
            clause_context += f"内容: {r['content'][:800]}\n"

        prompt = f"""你是一个海事消防法规合规检查专家。请依据以下法规条款，判断给定场景是否合规。

## 场景描述
{query}

## 相关法规条款
{clause_context}

请按照以下格式回答：

### 第一步：条款分析
逐条分析相关条款的要求。

### 第二步：场景检查
将场景中的实际情况与条款要求进行对比。

### 第三步：最终结论
**合规** 或 **不合规**
依据：简要说明理由。
"""
        return prompt

    def judge(self, prompt: str) -> str:
        response = self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是海事消防法规合规检查专家，严格依据法规条款进行判断。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
            timeout=120,
        )
        return response.choices[0].message.content

    def query(self, question: str) -> dict:
        clauses = self.retriever.search(question, top_k=10)
        prompt = self.build_prompt(question, clauses)
        answer = self.judge(prompt)
        return {
            'retrieved_clauses': clauses,
            'prompt': prompt,
            'answer': answer,
        }


def normalize_answer(answer):
    if not answer:
        return 'unknown'
    answer = answer.strip()

    conclusion_positions = [m.start() for m in re.finditer(r'最终结论', answer)]
    for pos in reversed(conclusion_positions):
        after = answer[pos:pos+500]
        has_non = '不合规' in after or '不符合' in after or '违规' in after
        cleaned = after.replace('不合规', '').replace('不符合', '').replace('违规', '')
        has_com = '合规' in cleaned or '符合' in cleaned
        if has_non and not has_com:
            return '不合规'
        if has_com and not has_non:
            return '合规'

    bold_matches = list(re.finditer(r'\*\*(不?合规)\*\*', answer))
    if bold_matches:
        return bold_matches[-1].group(1)

    nc = answer.rfind('不合规')
    c = answer.replace('不合规', '').rfind('合规')
    if nc > c:
        return '不合规'
    if c > nc:
        return '合规'
    return 'unknown'


def extract_clause_refs(answer):
    refs = set()
    for m in re.finditer(r'SOLAS\s+II-2/[\d.]+(?:\.\d+)*', answer):
        refs.add(m.group(0).strip())
    for m in re.finditer(r'FSS\s+Code/[\d.]+(?:\.\d+)*', answer):
        refs.add(m.group(0).strip())
    return refs


def clause_topic_match(ref, expected):
    ref_parts = ref.replace('SOLAS II-2/', '').replace('FSS Code/', '').split('.')
    exp_parts = expected.replace('SOLAS II-2/', '').replace('FSS Code/', '').split('.')
    ref_prefix = '.'.join(ref_parts[:3]) if len(ref_parts) >= 3 else ref
    exp_prefix = '.'.join(exp_parts[:3]) if len(exp_parts) >= 3 else expected
    if ref_prefix == exp_prefix:
        return 'exact'
    ref_top = '.'.join(ref_parts[:2]) if len(ref_parts) >= 2 else ref
    exp_top = '.'.join(exp_parts[:2]) if len(exp_parts) >= 2 else expected
    if ref_top == exp_top:
        return 'topic'
    return 'none'


def main():
    print("=" * 60)
    print("🔬 Hybrid RAG 基线评估 (BM25 + Vector 融合)")
    print("=" * 60)

    # 加载测试集
    with open(TEST_SET) as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条\n")

    # 初始化
    rag = HybridRAG()
    results = []
    start_time = time.time()

    for i, case in enumerate(test_data):
        qid = case['id']
        question = case.get('question', '')
        answer_data = case.get('answer', {})
        expected_compliance = answer_data.get('compliance', '')
        expected_clause = answer_data.get('clause_reference', '')

        case_start = time.time()
        print(f"[{i+1}/{len(test_data)}] {qid}")

        try:
            result = rag.query(question)
            predicted = normalize_answer(result['answer'])

            compliance_correct = predicted == expected_compliance
            predicted_refs = extract_clause_refs(result['answer'])
            expected_refs = set()
            if expected_clause:
                for ref in expected_clause.replace('、', ',').replace('，', ',').split(','):
                    ref = ref.strip()
                    if ref:
                        expected_refs.add(ref)

            exact_matches = set()
            topic_matches = set()
            for ref in expected_refs:
                if ref in predicted_refs:
                    exact_matches.add(ref)
                else:
                    for p in predicted_refs:
                        if clause_topic_match(p, ref) != 'none':
                            topic_matches.add(ref)
                            break

            case_result = {
                'id': qid,
                'question': question[:200],
                'expected_compliance': expected_compliance,
                'predicted_compliance': predicted,
                'compliance_correct': compliance_correct,
                'expected_refs': list(expected_refs),
                'predicted_refs': list(predicted_refs),
                'exact_matches': list(exact_matches),
                'topic_matches': list(topic_matches),
                'retrieved_clauses': [{
                    'clause_id': c['clause_id'],
                    'title': c['title'],
                    'content': c['content'][:500],
                    'bm25_norm': c.get('bm25_norm', 0),
                    'vector_norm': c.get('vector_norm', 0),
                    'final_score': c.get('final_score', 0),
                } for c in result['retrieved_clauses']],
                'full_answer': result['answer'],
                'elapsed': time.time() - case_start,
            }
            status = "✅" if compliance_correct else "❌"
            print(f"  {status} 预测={predicted} 期望={expected_compliance} ({case_result['elapsed']:.1f}s)")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            case_result = {
                'id': qid,
                'error': str(e),
                'expected_compliance': expected_compliance,
            }

        results.append(case_result)

        # 每10条打印进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            correct_so_far = sum(1 for r in results if r.get('compliance_correct'))
            print(f"\n--- 进度: {i+1}/{len(test_data)}, 准确率: {correct_so_far}/{i+1}={correct_so_far/(i+1)*100:.1f}%, 耗时: {elapsed/60:.1f}min ---\n")

    # ── 汇总 ──
    total = len(results)
    correct = sum(1 for r in results if r.get('compliance_correct'))
    total_expected = sum(len(r.get('expected_refs', [])) for r in results)
    exact_hits = sum(len(r.get('exact_matches', [])) for r in results)
    topic_hits = sum(len(r.get('topic_matches', [])) + len(r.get('exact_matches', [])) for r in results)

    summary = {
        'system': 'HybridRAG',
        'fusion': 'alpha=0.5 × BM25_norm + beta=0.5 × Vector_norm',
        'total_cases': total,
        'compliance_accuracy': correct / total if total > 0 else 0,
        'clause_exact_match': exact_hits / total_expected if total_expected > 0 else 0,
        'clause_topic_match': topic_hits / total_expected if total_expected > 0 else 0,
        'errors': [
            {'id': r['id'], 'expected': r['expected_compliance'], 'predicted': r['predicted_compliance']}
            for r in results if not r.get('compliance_correct') and 'error' not in r
        ],
    }

    output = {'summary': summary, 'details': results}
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("📊 Hybrid RAG 最终结果:")
    print(f"  合规准确率:     {correct}/{total} = {summary['compliance_accuracy']:.1%}")
    print(f"  条款精确命中:   {summary['clause_exact_match']:.1%}")
    print(f"  条款主题命中:   {summary['clause_topic_match']:.1%}")
    print(f"  错误数:         {total - correct}")
    print(f"  💾 保存: {OUTPUT_FILE}")

    rag.retriever.vector_ret.model = None  # 避免序列化问题


if __name__ == '__main__':
    main()
