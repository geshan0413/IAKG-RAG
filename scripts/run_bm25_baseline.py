#!/usr/bin/env python3
"""BM25 + LLM baseline evaluation on 100 test cases."""
import json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baselines import BM25LLM
from eval_baselines import evaluate_domain_metrics

# Load test set
with open('/root/autodl-tmp/.autodl/测试集.json') as f:
    test_data = json.load(f)

print(f'共 {len(test_data)} 条测试数据', flush=True)

# Initialize BM25+LLM
print('初始化 BM25+LLM...', flush=True)
system = BM25LLM()
print('✅ 就绪，开始评估...', flush=True)

t_start = time.time()

def run_bm25(question: str) -> dict:
    return system.query(question, top_k=10)

result = evaluate_domain_metrics(test_data, 'BM25+LLM', run_bm25)

t_total = time.time() - t_start

s = result['summary']
print(f'\n========== BM25+LLM 基线结果 ==========', flush=True)
print(f'总数: {s["total_cases"]}', flush=True)
print(f'合规判断准确率: {s["compliance_accuracy"]*100:.1f}%', flush=True)
print(f'条款精确命中率: {s["clause_exact_match"]*100:.1f}%', flush=True)
print(f'条款主题命中率: {s["clause_topic_match"]*100:.1f}%', flush=True)
print(f'总耗时: {t_total/60:.1f} 分钟', flush=True)

# Show errors
errors = [r for r in result['details'] if not r.get('compliance_correct') and 'error' not in r]
print(f'\n错误 ({len(errors)} 题):', flush=True)
for r in errors:
    print(f'  {r["id"][:40]}: 预测={r["predicted_compliance"]} 期望={r["expected_compliance"]}', flush=True)

# Save
def make_serializable(obj):
    """Recursively convert sets to lists for JSON serialization."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

output = {
    'model': 'mimo-v2-flash',
    'baseline_type': 'bm25_llm',
    'summary': s,
    'total_time_min': round(t_total/60, 2),
    'details': make_serializable(result['details'])
}

outpath = '/root/autodl-tmp/.autodl/baseline_bm25_llm_qwen35_100.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f'\n结果已保存到 {outpath}', flush=True)
