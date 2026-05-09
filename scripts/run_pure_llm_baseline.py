#!/usr/bin/env python3
"""Pure LLM baseline evaluation on 100 test cases."""
import json, time, re, sys
from openai import OpenAI

# Config
API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://api.xiaomimimo.com/v1'
MODEL = 'mimo-v2-flash'

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Load test set
with open('/root/autodl-tmp/.autodl/测试集.json') as f:
    test_data = json.load(f)

print(f'共 {len(test_data)} 条测试数据', flush=True)

SYSTEM_PROMPT = """你是一位资深的海事消防安全法规合规审查专家。你需要根据你对 SOLAS（国际海上人命安全公约）第 II-2 章和 FSS Code（消防安全系统规则）的了解，判断给定的场景描述是否合规。

请按照以下格式输出你的判断：

【分析】
（简要分析场景涉及的法规要求和关键点）

【结论】
（合规 或 不合规）

【依据】
（列出你判断所依据的法规条款编号和内容）

注意：
1. 仅基于你对 SOLAS 和 FSS Code 的了解进行判断
2. 如果你的知识中没有相关信息，请说明无法确定
3. 结论必须明确为"合规"或"不合规" """

results = []
total = len(test_data)
t_start = time.time()

for i, item in enumerate(test_data):
    qid = item['id']
    question = item['question']
    expected = item['answer']

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ],
            max_tokens=1500,
            timeout=120
        )
        answer_text = resp.choices[0].message.content
        elapsed = time.time() - t0
    except Exception as e:
        answer_text = f'ERROR: {e}'
        elapsed = time.time() - t0

    # Extract compliance judgment
    compliance = '未知'
    if re.search(r'【结论】\s*\n?\s*不合规', answer_text):
        compliance = '不合规'
    elif re.search(r'【结论】\s*\n?\s*合规', answer_text):
        compliance = '合规'

    # Extract clause references
    clause_refs = re.findall(r'(?:SOLAS|FSS)\s*(?:II-2)?/?[\d\.]+(?:\.\d+)*', answer_text)

    # Check correctness
    expected_compliance = expected.get('compliance', '').strip()
    correct = (compliance == expected_compliance)

    # Check clause topic hit
    topic_hit = False
    expected_clause = expected.get('clause_reference', '')
    if expected_clause:
        topic_parts = re.findall(r'[\d]+', expected_clause)
        answer_clause_nums = []
        for ref in clause_refs:
            answer_clause_nums.extend(re.findall(r'[\d]+', ref))
        if len(topic_parts) >= 3:
            topic_prefix = topic_parts[:3]
            for j in range(len(answer_clause_nums) - 2):
                if answer_clause_nums[j:j+3] == topic_prefix:
                    topic_hit = True
                    break

    # Check exact clause hit
    exact_hit = False
    if expected_clause:
        for ref in clause_refs:
            if expected_clause.replace(' ', '') in ref.replace(' ', '') or ref.replace(' ', '') in expected_clause.replace(' ', ''):
                exact_hit = True
                break

    results.append({
        'id': qid,
        'question': question[:100],
        'expected_compliance': expected_compliance,
        'predicted_compliance': compliance,
        'correct': correct,
        'expected_clause': expected_clause,
        'predicted_clauses': clause_refs,
        'clause_exact_hit': exact_hit,
        'clause_topic_hit': topic_hit,
        'response_time': round(elapsed, 2),
        'answer_text': answer_text[:500]
    })

    correct_count = sum(1 for r in results if r['correct'])
    print(f'[{i+1}/{total}] {qid[:30]}... 预测={compliance} 期望={expected_compliance} {"✅" if correct else "❌"} 耗时={elapsed:.1f}s (累计: {correct_count}/{i+1}={correct_count/(i+1)*100:.1f}%)', flush=True)

# Summary
t_total = time.time() - t_start
correct_total = sum(1 for r in results if r['correct'])
exact_hits = sum(1 for r in results if r['clause_exact_hit'])
topic_hits = sum(1 for r in results if r['clause_topic_hit'])

print(f'\n========== 纯LLM基线结果 ==========', flush=True)
print(f'总数: {total}', flush=True)
print(f'合规判断准确率: {correct_total}/{total} = {correct_total/total*100:.1f}%', flush=True)
print(f'条款精确命中率: {exact_hits}/{total} = {exact_hits/total*100:.1f}%', flush=True)
print(f'条款主题命中率: {topic_hits}/{total} = {topic_hits/total*100:.1f}%', flush=True)
print(f'总耗时: {t_total/60:.1f} 分钟', flush=True)
print(f'平均响应时间: {sum(r["response_time"] for r in results)/total:.1f}s', flush=True)

# Save results
output = {
    'model': MODEL,
    'baseline_type': 'pure_llm',
    'total': total,
    'compliance_accuracy': correct_total/total,
    'clause_exact_hit_rate': exact_hits/total,
    'clause_topic_hit_rate': topic_hits/total,
    'total_time_min': round(t_total/60, 2),
    'avg_response_time': round(sum(r['response_time'] for r in results)/total, 2),
    'details': results
}

outpath = '/root/autodl-tmp/.autodl/baseline_pure_llm_qwen35_100.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f'结果已保存到 {outpath}', flush=True)
