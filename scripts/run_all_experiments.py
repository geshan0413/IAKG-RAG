#!/usr/bin/env python3
"""
统一实验入口 — 支持7组实验，断点续跑，每10条保存
用法: python run_all_experiments.py --exp E1 [--resume]
"""
import json, os, sys, time, re, argparse
from pathlib import Path
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

# ── 配置 ──
MIMO_KEY = "YOUR_API_KEY"
MIMO_URL = "https://api.xiaomimimo.com/v1"
TEST_FILE = '/root/autodl-tmp/.autodl/小论文所需所有文件/test_set.json'
RESULTS_DIR = '/root/autodl-tmp/.autodl/小论文所需所有文件/results'

EXPERIMENTS = {
    'E1': {'name': 'Pure LLM', 'output': 'exp1_pure_llm_300.json'},
    'E2': {'name': 'BM25+LLM', 'output': 'exp2_bm25_300.json'},
    'E3': {'name': 'Naive RAG', 'output': 'exp3_naive_rag_300.json'},
    'E4': {'name': 'Hybrid RAG', 'output': 'exp4_hybrid_rag_300.json'},
    'E5': {'name': 'IAKG-RAG', 'output': 'exp5_iakg_rag_300.json'},
    'A2': 'w/o IRIR', 'output': 'ablation_no_irir.json'},
}

# ── 工具函数 ──
def normalize_answer(answer):
    if not answer: return 'unknown'
    answer = answer.strip()
    # 找最终结论区域
    for m in re.finditer(r'最终结论', answer):
        section = answer[m.start():m.start()+800]
        bold = re.search(r'\*\*(不合规|合规)\*\*', section)
        if bold: return bold.group(1)
        cm = re.search(r'结论[：:]\s*(不合规|合规)', section)
        if cm: return cm.group(1)
    # 全文粗体
    bold_all = list(re.finditer(r'\*\*(不合规|合规)\*\*', answer))
    if bold_all: return bold_all[-1].group(1)
    # 【结论】格式
    cm = re.search(r'【结论】\s*\n?\s*(不?合规)', answer)
    if cm: return cm.group(1)
    # 关键词
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


# ── E1: Pure LLM ──
def run_e1(test_data, output_file, resume=False):
    client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E1] Pure LLM: 从第{start+1}条开始，共{len(test_data)}条")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model='mimo-v2-flash',
                messages=[
                    {'role':'system','content':'你是海事消防法规合规检查专家。'},
                    {'role':'user','content':f"请判断以下场景是否合规，并引用具体条款。\n\n{q}"}
                ], temperature=0.1, max_tokens=4000, timeout=180)
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': None, 'retrieved_contexts': None,
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
            save_checkpoint(details, 'Pure LLM', output_file)
    
    save_checkpoint(details, 'Pure LLM', output_file)
    return details


# ── E2: BM25+LLM ──
def run_e2(test_data, output_file, resume=False):
    import jieba
    from rank_bm25 import BM25Okapi
    
    client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)
    clauses_json = '/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json'
    with open(clauses_json) as f:
        clauses = json.load(f)
    
    print(f"构建BM25索引({len(clauses)}条)...")
    tokenized = [list(jieba.cut(f"{c['clause_id']} {c['title']} {c['content']}")) for c in clauses]
    bm25 = BM25Okapi(tokenized)
    
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E2] BM25+LLM: 从第{start+1}条开始")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        # BM25检索
        tokens = list(jieba.cut(q))
        scores = bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:10]
        retrieved = [{'clause_id': clauses[j]['clause_id'], 'title': clauses[j]['title'], 'content': clauses[j]['content'], 'score': float(scores[j])} for j in top_idx]
        
        prompt = build_rag_prompt(q, retrieved)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model='mimo-v2-flash',
                messages=[{'role':'system','content':'你是海事消防法规合规检查专家。'},{'role':'user','content':prompt}],
                temperature=0.1, max_tokens=4000, timeout=180)
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved, 'retrieved_contexts': [r['content'] for r in retrieved],
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
            save_checkpoint(details, 'BM25+LLM', output_file)
    
    save_checkpoint(details, 'BM25+LLM', output_file)
    return details


# ── E3: Naive RAG ──
def run_e3(test_data, output_file, resume=False):
    from embed_clauses import VectorRetriever
    client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)
    retriever = VectorRetriever()
    
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E3] Naive RAG: 从第{start+1}条开始")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        retrieved = retriever.search(q, top_k=10)
        prompt = build_rag_prompt(q, retrieved)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model='mimo-v2-flash',
                messages=[{'role':'system','content':'你是海事消防法规合规检查专家。'},{'role':'user','content':prompt}],
                temperature=0.1, max_tokens=4000, timeout=180)
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved, 'retrieved_contexts': [r.get('content','') for r in retrieved],
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
            save_checkpoint(details, 'Naive RAG', output_file)
    
    save_checkpoint(details, 'Naive RAG', output_file)
    return details


# ── E4: Hybrid RAG ──
def run_e4(test_data, output_file, resume=False):
    import jieba
    from rank_bm25 import BM25Okapi
    from embed_clauses import VectorRetriever
    import numpy as np
    
    client = OpenAI(api_key=MIMO_KEY, base_url=MIMO_URL)
    clauses_json = '/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json'
    with open(clauses_json) as f:
        clauses = json.load(f)
    
    print("构建BM25索引...")
    tokenized = [list(jieba.cut(f"{c['clause_id']} {c['title']} {c['content']}")) for c in clauses]
    bm25 = BM25Okapi(tokenized)
    vector_ret = VectorRetriever()
    
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E4] Hybrid RAG: 从第{start+1}条开始")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        # BM25
        tokens = list(jieba.cut(q))
        bm25_scores = bm25.get_scores(tokens)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:50]
        bm25_max = float(bm25_scores[bm25_top[0]]) if bm25_top and bm25_scores[bm25_top[0]]>0 else 1.0
        bm25_dict = {}
        for j in bm25_top:
            cid = clauses[j]['clause_id']
            bm25_dict[cid] = {'clause_id':cid, 'title':clauses[j]['title'], 'content':clauses[j]['content'], 'bm25_norm': float(bm25_scores[j])/bm25_max}
        
        # Vector
        vec_results = vector_ret.search(q, top_k=50)
        vec_max = max([r['score'] for r in vec_results]) if vec_results else 1.0
        vec_dict = {}
        for r in vec_results:
            cid = r['clause_id']
            vec_dict[cid] = {'clause_id':cid, 'title':r['title'], 'content':r['content'], 'vec_norm': r['score']/vec_max if vec_max>0 else 0}
        
        # Fusion
        all_ids = set(bm25_dict.keys()) | set(vec_dict.keys())
        fused = []
        for cid in all_ids:
            b = bm25_dict.get(cid,{}).get('bm25_norm',0)
            v = vec_dict.get(cid,{}).get('vec_norm',0)
            info = bm25_dict.get(cid) or vec_dict.get(cid)
            fused.append({'clause_id':cid, 'title':info['title'], 'content':info['content'], 'score': 0.5*b+0.5*v})
        fused.sort(key=lambda x: x['score'], reverse=True)
        retrieved = fused[:10]
        
        prompt = build_rag_prompt(q, retrieved)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model='mimo-v2-flash',
                messages=[{'role':'system','content':'你是海事消防法规合规检查专家。'},{'role':'user','content':prompt}],
                temperature=0.1, max_tokens=4000, timeout=180)
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"ERROR: {e}"
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved, 'retrieved_contexts': [r.get('content','') for r in retrieved],
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
            save_checkpoint(details, 'Hybrid RAG', output_file)
    
    save_checkpoint(details, 'Hybrid RAG', output_file)
    return details


# ── E5: IAKG-RAG (续跑) ──
def run_e5(test_data, output_file, resume=True):
    from compliance_rag_v7 import ComplianceRAG_V7
    rag = ComplianceRAG_V7()
    
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E5] IAKG-RAG: 从第{start+1}条开始")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        t0 = time.time()
        try:
            raw = rag.query(q, top_k=15)
            answer = raw['answer']
            retrieved = raw.get('retrieved_clauses', [])
        except Exception as e:
            answer = f"ERROR: {e}"
            retrieved = []
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved, 'retrieved_contexts': [r.get('content','') if isinstance(r,dict) else str(r) for r in retrieved],
            'expected_compliance': exp_comp, 'expected_refs': list(exp_refs),
            'reference': f"{exp_comp}，依据：{case['answer'].get('clause_content','')[:500]}",
            'reference_contexts': gt_contexts,
            'accuracy_correct': pred_comp == exp_comp,
            'llm_time_ms': round((time.time()-t0)*1000),
            'intent': raw.get('intent','') if isinstance(raw,dict) else ''
        })
        
        status = "✅" if pred_comp == exp_comp else "❌"
        acc_so_far = sum(1 for d in details if d.get('accuracy_correct')) / len(details)
        print(f"  [{i+1}/{len(test_data)}] {case['id']} {status} 预测:{pred_comp} 期望:{exp_comp} 累计:{acc_so_far:.1%}", flush=True)
        
        if (i+1) % 10 == 0:
            save_checkpoint(details, 'IAKG-RAG', output_file)
    
    save_checkpoint(details, 'IAKG-RAG', output_file)
    return details


# ── E6: w/o IRIR ──
def run_e6(test_data, output_file, resume=False):
    """IAKG-RAG without IRIR - override retrieve to skip intent recognition"""
    import compliance_rag_v7 as crag_mod
    
    # Monkey-patch: disable IRIR by returning default positive intent
    def neutral_intent(query, client):
        return {'intent': 'positive', 'missing_entities': [], 'context_entities': [], 'fallback_level': 99}
    
    # Save original and patch
    orig_intent = crag_mod.intent_recognition
    crag_mod.intent_recognition = neutral_intent
    
    from compliance_rag_v7 import ComplianceRAG_V7
    rag = ComplianceRAG_V7()
    
    details = load_checkpoint(output_file) if resume else []
    start = len(details)
    print(f"[E6] w/o IRIR: 从第{start+1}条开始")
    
    for i in range(start, len(test_data)):
        case = test_data[i]
        q = case['question']
        exp_comp = case['answer']['compliance']
        exp_refs = set(r.strip() for r in case['answer'].get('clause_reference','').split(';') if r.strip())
        
        t0 = time.time()
        try:
            raw = rag.query(q, top_k=15)
            answer = raw['answer']
            retrieved = raw.get('retrieved_clauses', [])
        except Exception as e:
            answer = f"ERROR: {e}"
            retrieved = []
        
        pred_comp = normalize_answer(answer)
        pred_refs = extract_clause_refs(answer)
        gt_contexts = [c.strip() for c in case['answer'].get('clause_content','').split('\n') if len(c.strip())>20]
        
        details.append({
            'id': case['id'], 'question': q, 'full_answer': answer,
            'predicted_compliance': pred_comp, 'predicted_refs': list(pred_refs),
            'retrieved_clauses': retrieved, 'retrieved_contexts': [r.get('content','') if isinstance(r,dict) else str(r) for r in retrieved],
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
            save_checkpoint(details, 'w/o IRIR', output_file)
    
    save_checkpoint(details, 'w/o IRIR', output_file)
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=list(EXPERIMENTS.keys()), required=True)
    parser.add_argument('--resume', action='store_true', help='从断点继续')
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")
    
    output_file = os.path.join(RESULTS_DIR, EXPERIMENTS[args.exp]['output'])
    print(f"实验: {args.exp} - {EXPERIMENTS[args.exp]['name']}")
    print(f"输出: {output_file}")
    print(f"{'='*60}")
    
    runners = {
        'E1': run_e1, 'E2': run_e2, 'E3': run_e3,
        'E4': run_e4, 'E5': run_e5, 'E6': run_e6
    }
    
    t_start = time.time()
    details = runners[args.exp](test_data, output_file, resume=args.resume)
    t_total = time.time() - t_start
    
    correct = sum(1 for d in details if d.get('accuracy_correct'))
    print(f"\n{'='*60}")
    print(f"✅ {EXPERIMENTS[args.exp]['name']} 完成: {correct}/{len(details)} = {correct/len(details):.1%}")
    print(f"耗时: {t_total/60:.1f} 分钟")


if __name__ == '__main__':
    main()
