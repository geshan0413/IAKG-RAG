#!/usr/bin/env python3
"""
消融实验 — 3组配置
A1: 去掉 Prompt
A2: 去掉 IRIR
A3: 去掉知识图谱
"""
import json, os, sys, time, re
from pathlib import Path
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

MIMO_KEY = "YOUR_API_KEY"
MIMO_URL = "https://api.xiaomimimo.com/v1"

TEST_FILE = '/root/autodl-tmp/.autodl/小论文所需所有文件/test_set.json'
RESULTS_DIR = '/root/autodl-tmp/.autodl/小论文所需所有文件/results'

# ── 工具函数（与 run_all_experiments.py 一致） ──
def normalize_answer(answer):
    if not answer: return 'unknown'
    answer = answer.strip()
    for m in re.finditer(r'最终结论', answer):
        section = answer[m.start():m.start()+800]
        bold = re.search(r'\*\*(不合规|合规)\*\*', section)
        if bold: return bold.group(1)
        cm = re.search(r'结论[：:]\s*(不合规|合规)', section)
        if cm: return cm.group(1)
    bold_all = list(re.finditer(r'\*\*(不合规|合规)\*\*', answer))
    if bold_all: return bold_all[-1].group(1)
    cm = re.search(r'【结论】\s*\n?\s*(不?合规)', answer)
    if cm: return cm.group(1)
    nc = [m.start() for m in re.finditer(r'不合规', answer)]
    c = [m.start() for m in re.finditer(r'(?<!不)合规', answer)]
    if nc and c: return '合规' if max(c) > max(nc) else '不合规'
    if nc: return '不合规'
    if c: return '合规'
    return 'unknown'

def extract_clause_refs(text):
    refs = set()
    for m in re.finditer(r'SOLAS\s+II-2[/.](\d+(?:\.\d+)*)', text):
        refs.add(f"SOLAS II-2/{m.group(1)}")
    for m in re.finditer(r'FSS\s*Code[/.](\d+(?:\.\d+)*)', text):
        refs.add(f"FSS Code/{m.group(1)}")
    return refs

def load_checkpoint(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                data = json.load(f)
            return data.get('details', [])
        except: pass
    return []

def save_checkpoint(details, exp_name, output_file):
    total = len(details)
    correct = sum(1 for d in details if d.get('accuracy_correct'))
    acc = correct / total if total else 0
    output = {
        'system': exp_name,
        'test_set': 'test_set.json',
        'timestamp': datetime.now().isoformat(),
        'summary': {'total': total, 'correct': correct, 'accuracy': round(acc, 4)},
        'details': details
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ── A1: 去掉 Prompt（无推理引导） ──
def build_plain_prompt(query, clauses):
    """无推理引导，只拼条款"""
    clause_ctx = ""
    for i, c in enumerate(clauses):
        clause_ctx += f"\n条款{i+1} [{c.get('clause_id','')}]: {c.get('content','')[:800]}\n"
    return f"""请根据以下法规条款判断场景是否合规。

场景：{query}

相关条款：
{clause_ctx}

请直接给出结论：合规或不合规，并引用相关条款编号。"""

def run_ablation(test_data, output_file, exp_name, use_irir=True, use_graph=True, use_prompt=True):
    """通用消融实验入口"""
    client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)

    details = load_checkpoint(output_file)
    start = len(details)
    if start > 0:
        print(f"[{exp_name}] 从第{start+1}条断点续跑")

    if not use_graph:
        # 不用图谱，只用向量检索
        from embed_clauses import VectorRetriever
        retriever = VectorRetriever()
    else:
        from compliance_rag_v7 import ComplianceRAG_V7
        if use_irir:
            rag = ComplianceRAG_V7()
        else:
            # 去掉 IRIR：monkey-patch 跳过意图识别
            import compliance_rag_v7 as crag
            _orig_ir = crag.intent_recognition
            crag.intent_recognition = lambda q, c: {'intent': 'positive', 'missing_entities': [], 'context_entities': [], 'fallback_level': 3}
            rag = ComplianceRAG_V7()

    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())

        t0 = time.time()
        try:
            if not use_graph:
                # 纯向量检索
                retrieved = retriever.search(q, top_k=15)
                prompt = build_plain_prompt(q, retrieved) if not use_prompt else build_rag_prompt(q, retrieved)
                resp = client.chat.completions.create(
                    model='mimo-v2-flash',
                    messages=[{'role':'system','content':'你是海事消防法规合规检查专家。'},{'role':'user','content':prompt}],
                    temperature=0.1, max_tokens=4000, timeout=180)
                answer = resp.choices[0].message.content
            else:
                raw = rag.query(q, top_k=15)
                answer = raw['answer']
                retrieved = raw.get('retrieved_clauses', [])

                if not use_prompt:
                    # 用 IAKG-RAG 检索但用 plain prompt
                    if isinstance(retrieved, list) and retrieved and isinstance(retrieved[0], dict):
                        plain = build_plain_prompt(q, retrieved)
                        resp = client.chat.completions.create(
                            model='mimo-v2-flash',
                            messages=[{'role':'system','content':'你是海事消防法规合规检查专家。'},{'role':'user','content':plain}],
                            temperature=0.1, max_tokens=4000, timeout=180)
                        answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"
            retrieved = []

        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]

        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved,
            'retrieved_contexts': [r.get('content','') if isinstance(r,dict) else str(r) for r in retrieved] if retrieved else [],
            'expected_compliance': exp_comp, 'expected_refs': list(exp_refs),
            'reference': f"{exp_comp}，依据：{case['answer'].get('clause_content','')[:500]}",
            'reference_contexts': gt_contexts,
            'accuracy_correct': pred_comp == exp_comp,
            'llm_time_ms': round((time.time()-t0)*1000)
        })

        status = "✅" if pred_comp == exp_comp else "❌"
        acc_so_far = sum(1 for d in details if d.get('accuracy_correct')) / len(details)
        print(f"  [{i+1}/{len(test_data)}] {case['id']} {status} 预测:{pred_comp} 期望:{exp_comp} 累计:{acc_so_far:.1%}", flush=True)

        if (i+1) % 10 == 0:
            save_checkpoint(details, exp_name, output_file)

    save_checkpoint(details, exp_name, output_file)
    return details


def build_rag_prompt(query, clauses):
    clause_ctx = ""
    for i, c in enumerate(clauses):
        clause_ctx += f"\n### 条款 {i+1}: {c.get('clause_id','')}\n"
        clause_ctx += f"标题: {c.get('title','')}\n"
        clause_ctx += f"内容: {c.get('content','')[:800]}\n"
    return f"""你是海事消防法规合规检查专家。请严格依据以下法规条款判断场景是否合规。

## 场景描述
{query}

## 相关法规条款
{clause_ctx}

## 判断流程

**第一步：提取要求** — 从每个条款中提取具体要求
**第二步：逐项核对** — 将场景做法与每条要求对比（✅满足/❌不满足/⚠️未提及）
**第三步：综合判定**
- 有任何❌ → 不合规
- 全部✅ → 合规
- ⚠️未提及：无矛盾信息时不影响判定

## 输出格式
### 第一步：条款要求提取
### 第二步：逐项核对
### 第三步：最终结论
- **合规/不合规**：[明确结论]
- 引用条款：[编号列表]"""


def main():
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['A1', 'A3'], required=True)
    args = parser.parse_args()

    if args.exp == 'A1':
        print("=" * 50)
        print("A1: 去掉 Prompt（无推理引导，只拼条款）")
        print("=" * 50)
        output = os.path.join(RESULTS_DIR, 'ablation_no_prompt.json')
        run_ablation(test_data, output, 'A1-NoPrompt', use_irir=True, use_graph=True, use_prompt=False)

    elif args.exp == 'A3':
        print("=" * 50)
        print("A3: 去掉知识图谱（纯向量检索 + 简单prompt，与E3一致）")
        print("=" * 50)
        output = os.path.join(RESULTS_DIR, 'ablation_no_graph.json')
        run_ablation(test_data, output, 'A3-NoGraph', use_irir=False, use_graph=False, use_prompt=True)


if __name__ == '__main__':
    main()
