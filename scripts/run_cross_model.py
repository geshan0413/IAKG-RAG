#!/usr/bin/env python3
"""
KG-RAG v6 跨模型对比实验
基于 compliance_rag_v6.py 的完整 IRIR 框架，仅替换 LLM
"""
import json, sys, os, time
from pathlib import Path
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'

# v6 代码目录
V6_DIR = Path('/root/autodl-tmp/.autodl/小论文所需所有文件/代码')
sys.path.insert(0, str(V6_DIR))

# 导入 v6 模块（先导入，后面猴子补丁替换 LLM 配置）
import compliance_rag_v6 as v6
from eval_baselines import normalize_answer, extract_clause_refs, clause_topic_match
from openai import OpenAI

TEST_FILE = '/root/autodl-tmp/.autodl/测试集.json'
OUTPUT_DIR = Path('/root/autodl-tmp/.autodl/大论文')

# AutoDL API 配置
AUTODL_KEY = "YOUR_DEEPSEEK_API_KEY"
AUTODL_URL = "https://www.autodl.art/api/v1"

MODELS = {
    'GLM-5': {
        'api_key': AUTODL_KEY,
        'base_url': AUTODL_URL,
        'model': 'GLM-5',
        'thinking': True,  # 需要关闭思维链
    },
    'Kimi-K2.5': {
        'api_key': AUTODL_KEY,
        'base_url': AUTODL_URL,
        'model': 'Kimi-K2.5',
        'thinking': True,
    },
}


def make_rag(model_name, config):
    """创建使用指定 LLM 的 v6 RAG 实例"""

    class CrossModelRAG(v6.ComplianceRAG_V6):
        def __init__(self):
            # 手动初始化检索器（不调用父类 __init__，因为它会创建 MiMo client）
            from graph_retriever import GraphRetriever
            from embed_clauses import VectorRetriever

            print(f"初始化 IAKG-RAG v6（LLM: {model_name}）...")
            self.graph_ret = GraphRetriever()
            self.vector_ret = VectorRetriever()
            self.llm_client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
            self.model_name = model_name
            self.thinking = config.get('thinking', False)
            print(f"✅ 系统就绪（LLM: {model_name}）\n")

        def llm_judge(self, prompt: str) -> str:
            kwargs = {
                'model': self.model_name,
                'messages': [
                    {"role": "system", "content": "你是海事消防法规合规检查专家，严格依据法规条款进行判断。"},
                    {"role": "user", "content": prompt},
                ],
                'temperature': 0.1,
                'max_tokens': 2000,
                'timeout': 120,
            }
            # 关闭思维链
            if self.thinking:
                kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}

            response = self.llm_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        def query(self, question: str, top_k: int = 15) -> dict:
            """重写 query，修复 print 中硬编码的 'MiMo'"""
            print(f"\n{'='*60}")
            print(f"📝 查询: {question[:100]}...")
            print(f"{'='*60}\n")

            retrieval_results = self.retrieve(question, top_k=top_k)
            prompt = self.build_prompt(question, retrieval_results)

            print(f"\n🤖 调用 {self.model_name} 进行合规判断...")
            answer = self.llm_judge(prompt)
            print(f"\n📋 {self.model_name} 回答:\n{answer}")

            return {
                'entities': retrieval_results['entities'],
                'intent': retrieval_results['intent'],
                'reformulated_queries': retrieval_results['reformulated_queries'],
                'retrieved_clauses': retrieval_results['results'],
                'prompt': prompt,
                'answer': answer,
            }

    # 替换 v6 模块级变量，让 intent_recognition_llm 也用新模型
    orig_model = v6.MODEL_NAME
    orig_client_init = v6.OpenAI

    v6.MODEL_NAME = model_name

    rag = CrossModelRAG()

    # intent_recognition_llm 用的是模块级 client（通过 v6.OpenAI），
    # 但它在函数内用的是传入的 client 参数，由 retrieve() 调用时传 self.llm_client
    # 所以只需要确保 self.llm_client 是对的，以及 MODEL_NAME 被替换即可

    return rag


def run_test(model_name, config):
    """用指定模型跑完整测试集"""
    with open(TEST_FILE) as f:
        test_data = json.load(f)

    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name}")
    print(f"# 测试集: {len(test_data)} 条")
    print(f"{'#'*60}")

    rag = make_rag(model_name, config)

    # 断点续跑：检查是否有中间结果
    interim_file = OUTPUT_DIR / f'kg_rag_{model_name.lower().replace("-", "_")}_interim.json'
    results = []
    start_offset = 0
    if interim_file.exists():
        interim = json.load(open(interim_file))
        results = interim.get('results', [])
        start_offset = len(results)
        print(f"🔄 从断点续跑: 已完成 {start_offset} 条, 准确率 {interim.get('accuracy', 0):.1f}%")

    start_time = time.time()

    for i, case in enumerate(test_data):
        if i < start_offset:
            continue
        qid = case['id']
        question = case['question']
        expected_compliance = case['answer'].get('compliance', '')
        expected_refs_str = case['answer'].get('clause_reference', '')
        expected_refs = set()
        if expected_refs_str:
            for ref in expected_refs_str.split(';'):
                ref = ref.strip()
                if ref:
                    expected_refs.add(ref)

        print(f"\n[{i+1}/{len(test_data)}] {qid}")
        case_start = time.time()

        try:
            raw = rag.query(question)
            answer = raw['answer']
            predicted = normalize_answer(answer)
            predicted_refs = extract_clause_refs(answer)

            compliance_correct = (predicted == expected_compliance)

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

            result = {
                'id': qid,
                'question': question[:200],
                'expected_compliance': expected_compliance,
                'predicted_compliance': predicted,
                'compliance_correct': compliance_correct,
                'expected_refs': list(expected_refs),
                'predicted_refs': list(predicted_refs),
                'exact_matches': list(exact_matches),
                'topic_matches': list(topic_matches),
                'retrieved_clause_ids': [c['clause_id'] for c in raw.get('retrieved_clauses', [])],
                'full_answer': raw.get('answer', ''),
                'elapsed': time.time() - case_start,
            }
            status = "✅" if compliance_correct else "❌"
            print(f"  {status} 预测={predicted} 期望={expected_compliance} ({result['elapsed']:.1f}s)")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            result = {
                'id': qid,
                'error': str(e),
                'expected_compliance': expected_compliance,
                'elapsed': time.time() - case_start,
            }

        results.append(result)

        # 每10条保存中间结果 + 准确率检查
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            correct_so_far = sum(1 for r in results if r.get('compliance_correct'))
            acc_so_far = correct_so_far / (i + 1) * 100
            print(f"\n--- 中间进度: {i+1}/{len(test_data)}, "
                  f"准确率: {correct_so_far}/{i+1}={acc_so_far:.1f}%, "
                  f"耗时: {elapsed/60:.1f}min ---")

            # 中间结果保存
            interim_file = OUTPUT_DIR / f'kg_rag_{model_name.lower().replace("-", "_")}_interim.json'
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump({'count': len(results), 'accuracy': acc_so_far, 'results': results},
                          f, ensure_ascii=False, indent=2)

            # 前10条准确率低于90%就停下
            if i + 1 == 10 and acc_so_far < 90:
                print(f"\n⚠️ 前10条准确率 {acc_so_far:.1f}% < 90%，停止 {model_name}，检查问题")
                rag.close()
                return {
                    'model': model_name,
                    'accuracy': acc_so_far / 100,
                    'correct': correct_so_far,
                    'total': i + 1,
                    'stopped': True,
                    'reason': f'前10条准确率 {acc_so_far:.1f}% < 90%',
                    'details': results,
                }

    rag.close()

    # 统计
    total_time = time.time() - start_time
    total = len(results)
    correct = sum(1 for r in results if r.get('compliance_correct'))
    errors = [r for r in results if not r.get('compliance_correct') and 'error' not in r]
    api_errors = [r for r in results if 'error' in r]

    total_expected = sum(len(r.get('expected_refs', [])) for r in results)
    exact_hits = sum(len(r.get('exact_matches', [])) for r in results)
    topic_hits = sum(len(r.get('topic_matches', [])) + len(r.get('exact_matches', [])) for r in results)

    # 不合规/合规分别统计
    nc_results = [r for r in results if r.get('expected_compliance') == '不合规']
    c_results = [r for r in results if r.get('expected_compliance') == '合规']
    nc_correct = sum(1 for r in nc_results if r.get('compliance_correct'))
    c_correct = sum(1 for r in c_results if r.get('compliance_correct'))

    summary = {
        'timestamp': datetime.now().isoformat(),
        'system': 'IAKG-RAG v6',
        'model': model_name,
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'nc_total': len(nc_results),
        'nc_correct': nc_correct,
        'nc_accuracy': nc_correct / len(nc_results) if nc_results else 0,
        'c_total': len(c_results),
        'c_correct': c_correct,
        'c_accuracy': c_correct / len(c_results) if c_results else 0,
        'judgment_errors': len(errors),
        'api_errors': len(api_errors),
        'clause_exact_rate': exact_hits / total_expected if total_expected > 0 else 0,
        'clause_topic_rate': topic_hits / total_expected if total_expected > 0 else 0,
        'total_time_min': total_time / 60,
    }

    output_file = OUTPUT_DIR / f'kg_rag_{model_name.lower().replace("-", "_")}_100.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 IAKG-RAG v6 + {model_name} 结果")
    print(f"{'='*60}")
    print(f"合规准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"不合规准确率: {nc_correct}/{len(nc_results)} = {nc_correct/len(nc_results)*100:.1f}%" if nc_results else "")
    print(f"合规准确率: {c_correct}/{len(c_results)} = {c_correct/len(c_results)*100:.1f}%" if c_results else "")
    if total_expected > 0:
        print(f"条款精确命中率: {exact_hits}/{total_expected} = {exact_hits/total_expected*100:.1f}%")
        print(f"条款主题命中率: {topic_hits}/{total_expected} = {topic_hits/total_expected*100:.1f}%")
    print(f"API 错误: {len(api_errors)}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"结果已保存: {output_file}")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载测试集确认
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")
    print(f"待测模型: {', '.join(MODELS.keys())}")
    print(f"框架: IAKG-RAG v6 (完整 IRIR)")
    print()

    all_summaries = {}

    for model_name, config in MODELS.items():
        summary = run_test(model_name, config)
        all_summaries[model_name] = summary

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"跨模型对比汇总（IAKG-RAG v6 + 100条测试集）")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'准确率':>8} {'不合规':>8} {'合规':>8} {'条款命中':>8}")
    print("-" * 50)

    # 加入 MiMo 已有结果
    mimo_file = OUTPUT_DIR / 'kg_rag_mimo_v2_flash_100.json'
    if mimo_file.exists():
        mimo_data = json.load(open(mimo_file))
        ms = mimo_data['summary']
        print(f"{'MiMo-v2-flash':<15} {ms['accuracy']:>8.4f} {ms.get('nc_accuracy', 0):>8.4f} "
              f"{ms.get('c_accuracy', 0):>8.4f} {ms.get('clause_exact_rate', 0):>8.4f}")

    for name, s in all_summaries.items():
        print(f"{name:<15} {s['accuracy']:>8.4f} {s.get('nc_accuracy', 0):>8.4f} "
              f"{s.get('c_accuracy', 0):>8.4f} {s.get('clause_exact_rate', 0):>8.4f}")

    # 保存汇总
    summary_file = OUTPUT_DIR / 'cross_model_comparison.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'framework': 'IAKG-RAG v6',
            'test_size': len(test_data),
            'summaries': all_summaries,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_file}")


if __name__ == '__main__':
    main()
