#!/usr/bin/env python3
"""
消融实验 C：去除 IRIR 机制
对比完整版 KG-RAG vs 去掉负向意图处理后的表现
"""
import json, sys, os, time, re
from pathlib import Path
from datetime import datetime
from openai import OpenAI

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).parent / '小论文所需所有文件' / '代码'))

from graph_retriever import extract_entities_enhanced as extract_entities, GraphRetriever, sanitize_fulltext_query
from embed_clauses import VectorRetriever
from eval_baselines import normalize_answer, extract_clause_refs, clause_topic_match

# ── 配置 ──
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"
TEST_FILE = '/root/autodl-tmp/.autodl/测试集.json'
OUTPUT_DIR = Path('/root/autodl-tmp/.autodl')


class ComplianceRAG_NoIRIR:
    """去除 IRIR 机制的合规 RAG（去掉安全网 + 困难条款查找）"""

    def __init__(self):
        print("初始化 [消融C] 去IRIR系统...")
        self.graph_ret = GraphRetriever()
        self.vector_ret = VectorRetriever()
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print("✅ 系统就绪\n")

    def llm_judge(self, prompt: str) -> str:
        response = self.llm_client.chat.completions.create(
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

    def retrieve(self, query: str, top_k: int = 15) -> dict:
        """检索阶段：只保留图谱+向量双路检索，去掉安全网和困难条款查找"""

        # 1. 实体抽取
        entities = extract_entities(query)
        print(f"📋 抽取实体: {json.dumps(entities, ensure_ascii=False)}")

        # 2. 图谱检索
        graph_results = self.graph_ret.search(entities, top_k=top_k * 5, scenario_text=query)
        print(f"🔗 图谱检索: {len(graph_results)} 条")

        # 3. 向量检索
        vector_results = self.vector_ret.search(query, top_k=top_k + 5, threshold=0.15)
        print(f"📐 向量检索: {len(vector_results)} 条")

        # 4. 融合去重
        merged = {}
        for r in graph_results:
            cid = r['id']
            merged[cid] = {
                'clause_id': cid,
                'title': r.get('title', ''),
                'content': r.get('preview', r.get('content', '')),
                'graph_score': 1.0,
                'vector_score': 0.0,
                'match_types': [r.get('match_type', 'graph')],
            }

        for r in vector_results:
            cid = r['clause_id']
            if cid in merged:
                merged[cid]['vector_score'] = r['score']
                merged[cid]['match_types'].append('vector')
                if len(r.get('content', '')) > len(merged[cid].get('content', '')):
                    merged[cid]['content'] = r['content']
            else:
                merged[cid] = {
                    'clause_id': cid,
                    'title': r.get('title', ''),
                    'content': r.get('content', ''),
                    'graph_score': 0.0,
                    'vector_score': r['score'],
                    'match_types': ['vector'],
                }

        # 5. 综合评分排序
        for cid, item in merged.items():
            graph_weight = 0.5
            vector_weight = 0.5
            item['final_score'] = (
                graph_weight * item['graph_score'] +
                vector_weight * item['vector_score']
            )
            if len(item['match_types']) > 1:
                item['final_score'] *= 1.3

        sorted_results = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)
        top_results = sorted_results[:top_k]

        # ❌ 不做安全网（Safety Net）
        # ❌ 不做困难条款直接查找（Hard Clause Lookup）

        print(f"📊 融合结果: {len(top_results)} 条")
        for i, r in enumerate(top_results):
            tags = '+'.join(r['match_types'])
            print(f"  {i+1}. [{tags}] {r['clause_id']}: {r['title'][:40]} (score={r['final_score']:.2f})")

        return {
            'entities': entities,
            'results': top_results,
        }

    def build_prompt(self, query: str, retrieval_results: dict) -> str:
        """复用原始 Prompt（与完整版相同）"""
        entities = retrieval_results['entities']
        results = retrieval_results['results']

        clause_context = ""
        for i, r in enumerate(results):
            clause_context += f"\n### 条款 {i+1}: {r['clause_id']}\n"
            clause_context += f"标题: {r['title']}\n"
            clause_context += f"内容: {r['content'][:800]}\n"

        entity_summary = []
        if entities['ship_types']:
            entity_summary.append(f"船型: {', '.join(entities['ship_types'])}")
        if entities['years']:
            entity_summary.append(f"建造年份: {entities['years']}")
        if entities['equipments']:
            entity_summary.append(f"涉及设备: {', '.join(entities['equipments'])}")
        if entities['spaces']:
            entity_summary.append(f"涉及空间: {', '.join(entities['spaces'])}")
        if entities['numbers']:
            nums = [f"{n['value']}{n['unit']}" for n in entities['numbers']]
            entity_summary.append(f"关键数值: {', '.join(nums)}")

        prompt = f"""你是一个海事消防法规合规检查专家。请严格依据以下法规条款的原文内容，判断给定场景是否合规。

## 场景描述
{query}

## 抽取的关键实体
{chr(10).join(entity_summary)}

## 相关法规条款（请逐条仔细阅读）
{clause_context}

## ⚠️ 核心判断原则（严格遵守）

你的任务是将场景中的**实际做法**与法规条款中的**具体要求**逐项核对。不要预设立场，也不要因为描述模糊就放过问题。

### 判断流程（必须严格执行）

**第一步：筛选相关条款（极其重要）**
在提取要求之前，严格判断每个条款是否与场景的**检查对象**直接相关：
- 场景有明确的检查对象（如"应急消防泵起动系统""走廊消防箱""挡火闸"等）
- **只有当条款直接规范该检查对象时，才纳入判断**
- 跳过所有涉及**不同设备/系统/处所**的条款，即使它们在技术上存在于同一条法规中
- ❌ 错误示范：场景检查消防泵起动 → 引用惰性气体系统条款判定不合规
- ❌ 错误示范：场景检查走廊消防箱 → 引用货物区域水雾枪条款
- ✅ 正确做法：只保留与检查对象直接相关的条款

**第二步：提取要求并检查豁免（豁免优先）**
从每个相关条款提取要求时，**必须同时检查豁免/例外条款**：
- 一般性条款（"不得使用闪点<60°C的燃油"）和专门性豁免条款（"应急发电机可用≥43°C燃油"）同时存在时，**豁免优先**
- 判断顺序：**先检查豁免是否适用** → 只有当豁免**明确不适用**时，才适用一般条款
- **豁免适用的判断标准**：
  - 场景对象属于豁免条款覆盖的范围（如"应急发电机"属于豁免对象）→ 豁免适用
  - 豁免条件中提到的附加条件（如"不得储存在机舱内"）在场景中**未提及** → 不能反推为不满足，应认为豁免适用
  - **举证责任**：判定豁免不适用时，必须引用场景中的具体事实与豁免条件矛盾，不能仅因"未提及"就否定豁免
- 例：闪点45°C用于应急发电机 → 有豁免条款允许应急发电机用≥43°C → 豁免适用 → 合规
- 例：重油日用柜在机舱内 → 有豁免条款允许日用柜在机舱 → 豁免适用 → 不能仅因禁止性条款就判不合规

**第三步：逐项核对**
将场景中的实际做法与每条要求对比：
- ✅ 满足：场景中的做法明确符合该要求（包括符合条款允许的等效方案，如A-60隔热替代钢质外套）
- ❌ 不满足：场景中的做法**明确违反**该要求（必须引用原文指出具体哪项条件不达标）
- ⚠️ 未提及：场景描述中**没有涉及**此项要求

**第四步：综合判定**

判定规则：
1. 如果有任何一项**相关条款的**要求被判定为"❌ 不满足"→ **不合规**
2. 如果所有要求都是"✅ 满足"→ **合规**
3. 如果存在"⚠️ 未提及"的要求，**按以下规则处理**：
   - **核心原则：无明确违反即合规。** 未提及 ≠ 不满足。
   - PSC检查场景描述的是**检查官的检查结果**。如果检查官检查了某设备并只提出特定问题，未提及的方面说明检查官认为满足
   - **绝对不能**因为"场景未证明X满足"就判定不合规——这是举证责任倒置
   - **只有**当场景中提供了**与要求直接矛盾**的事实时，才可判定不合规
   - 例：场景说"检查官检查了消防泵，发现起动时间超标" → 只判起动时间不合规，其他未提及方面不影响
   - 若该要求是条件性的，或场景本身未涉及该领域 → 该项不影响判定

### 🚫 绝对禁止的推理模式
- ❌ "场景未证明豁免条件满足 → 豁免不成立" — 应反过来：场景属于豁免范围 → 豁免成立，除非场景事实与豁免条件矛盾
- ❌ "场景未提及X → 无法确认X合规 → 不合规" — 未提及不等于不满足
- ❌ 用与检查对象无关的条款作为不合规理由 — 必须严格筛选相关条款
- ❌ "场景描述中未提供任何信息证明这些豁免条件已满足。因此，不能认定豁免成立" — 这是错误的举证逻辑

### 关键注意点
- 条款中的"或""等效""允许""替代"表示多种方案均可接受，满足任一即可
- 场景描述的基本事实（如船型、年份、设备类型）视为准确的，不要质疑场景设定本身
- 不要因为场景没提到某细节就假设"可能存在隐患"——只依据场景给出的事实判断
- 条款未明确规定的事项不能作为不合规的理由
- **当多个条款指向同一问题时，以最具体的条款为准**（特别法优于一般法）
- **场景边界原则**：场景描述了什么就判断什么，不要引入场景未涉及的设备/系统来判定不合规
- **检查官逻辑**：PSC场景描述检查官的发现。如果检查官没提出某项违规，说明该方面通过了检查

### 🔍 "缺少/仅有"类场景的特殊处理规则
当场景描述"仅配备了一台X""未发现备用X""缺少X"时，**必须主动检查该设备是否要求双套/备用/冗余配置**：
- 这类场景的核心问题就是"数量不足"，因此即使检索到的条款中没有明确提到"双套"，也要仔细阅读条款是否隐含冗余要求
- **重点关注**：条款中出现"应装有双套""备用""冗余""至少2台""独立驱动"等词时，与场景中的"仅有1台"直接矛盾 → ❌ 不满足
- 例：场景说"仅配备一台抽样风机" → 条款要求"应装有双套抽样风机" → ❌ 不满足
- **反向推理**：当场景明确描述了某种缺失（如"未发现水雾枪"），应该检查条款是否要求该设备。即使检索结果中没有直接包含该条款，如果场景提到了缺失某种设备，你应该追问：该处所/该船型是否需要这种设备？

### 🔢 数值阈值判断规则（极其重要）

当场景中出现数值时，**必须与条款中的阈值进行精确数学比较**：

**规则1：满足阈值即合规**
- 如果条款要求"不少于X"、"至少X"、"≥X"，且场景值 ≥ X → ✅ 满足
- 如果条款要求"不大于X"、"不超过X"、"≤X"，且场景值 ≤ X → ✅ 满足
- 例：条款要求≥40%，场景值=40% → ✅ 满足（"不少于"包含等于）
- 例：条款要求≥1200L，场景值=1000L → ❌ 不满足

**规则2：计算后比较**
- 如果场景给出了比例关系（如"70%即超过三分之二"），先确认数学关系再判断
- 三分之二 ≈ 66.7%，70% > 66.7% → ✅ 满足

**规则3：满足任一替代方案即合规**
- 如果条款给出多个替代方案（"应为A或B或C"），满足任一即可
- 例：FSS Code/10.2.2.1.1要求应急泵排量≥40%且≥某个最低值，两个条件都需满足

**规则4：豁免条件优先适用**
- 如果条款有豁免（"但X总吨以下可免除""应急发电机可用≥43°C"），先检查场景是否属于豁免范围
- 场景属于豁免范围 → 豁免适用 → 合规（除非场景事实与豁免附加条件直接矛盾）
- 例：1600总吨以下货船可免除驾驶室遥控启动 → 场景800总吨 → 豁免适用 → 合规
- 例：应急发电机可用≥43°C闪点燃油 → 场景是应急发电机闪点45°C → 豁免适用 → 合规

## 输出格式

### 第一步：相关条款筛选
简要说明哪些条款与场景相关，哪些不相关（跳过）

### 第二步：条款要求提取与豁免检查
列出从每个相关条款中提取的具体要求，同时标注是否存在豁免条款

### 第三步：逐项核对
对每条要求：
- 条款要求：[引用原文]
- 场景做法：[从场景中提取对应事实]
- 判断：✅ 满足 / ❌ 不满足 / ⚠️ 未提及
- 分析：[简要说明判断依据]

### 第四步：最终结论
- **合规** / **不合规**
- **依据**：简要总结判断逻辑"""

        return prompt

    def query(self, question: str, top_k: int = 15) -> dict:
        print(f"\n{'='*60}")
        print(f"📝 查询: {question[:100]}...")
        print(f"{'='*60}\n")

        retrieval_results = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, retrieval_results)
        print("\n🤖 调用 MiMo 进行合规判断...")
        answer = self.llm_judge(prompt)
        print(f"\n📋 MiMo 回答:\n{answer}")

        return {
            'entities': retrieval_results['entities'],
            'retrieved_clauses': retrieval_results['results'],
            'prompt': prompt,
            'answer': answer,
        }

    def close(self):
        self.graph_ret.close()


if __name__ == '__main__':
    with open(TEST_FILE) as f:
        test_data = json.load(f)

    print(f"测试集: {len(test_data)} 条")
    print(f"消融实验 C：去除 IRIR（安全网 + 困难条款查找）\n")

    rag = ComplianceRAG_NoIRIR()
    results = []
    start_time = time.time()

    for i, case in enumerate(test_data):
        case_start = time.time()
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
                'retrieved_clauses': raw.get('retrieved_clauses', []),
                'full_answer': answer,
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
            }

        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            correct_so_far = sum(1 for r in results if r.get('compliance_correct'))
            print(f"\n--- 中间进度: {i+1}/{len(test_data)}, 准确率: {correct_so_far}/{i+1}={correct_so_far/(i+1)*100:.1f}%, 耗时: {elapsed/60:.1f}min ---")

    # 汇总
    total_time = time.time() - start_time
    total = len(results)
    correct = sum(1 for r in results if r.get('compliance_correct'))
    errors = [r for r in results if not r.get('compliance_correct') and 'error' not in r]

    total_expected = sum(len(r.get('expected_refs', [])) for r in results)
    exact_hits = sum(len(r.get('exact_matches', [])) for r in results)
    topic_hits = sum(len(r.get('topic_matches', [])) + len(r.get('exact_matches', [])) for r in results)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'ablation': 'C_no_irir',
        'description': '去除IRIR（安全网+困难条款查找）',
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'error_count': len(errors),
        'clause_exact_rate': exact_hits / total_expected if total_expected > 0 else 0,
        'clause_topic_rate': topic_hits / total_expected if total_expected > 0 else 0,
        'total_time_min': total_time / 60,
        'errors': [{'id': e['id'], 'expected': e['expected_compliance'], 'predicted': e['predicted_compliance']} for e in errors],
    }

    output_file = OUTPUT_DIR / 'ablation_no_irir.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 消融C（去IRIR）结果")
    print(f"{'='*60}")
    print(f"合规准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"条款精确命中率: {exact_hits}/{total_expected} = {exact_hits/total_expected*100:.1f}%")
    print(f"条款主题命中率: {topic_hits}/{total_expected} = {topic_hits/total_expected*100:.1f}%")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"结果已保存: {output_file}")

    if errors:
        print(f"\n❌ 错误案例 ({len(errors)}个):")
        for e in errors:
            print(f"  {e['id']}: 预测={e['predicted_compliance']} 期望={e['expected_compliance']}")

    rag.close()
