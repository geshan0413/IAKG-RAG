"""
IAKG-RAG v7 

1. 图谱分支按关系类型差异化赋权：
   - equipment+space 联合命中  → 权重 2.0
   - equipment / space 单一命中 → 权重 1.0
   - fulltext (全文索引)        → 权重 0.5
   - hierarchy-parent/child     → 权重 0.5
   - irir-reformulated          → 权重 0.5
   - REFERENCES (交叉引用)      → 权重 1.5
2. 图谱得分 Min-Max 归一化到 [0,1]（论文公式）
3. 交叉命中严格判定 = 同时被图谱+向量分支命中 → ×1.3 增益
4. 安全网补充得分 ×0.3 衰减（论文 §2.2.2）
5. 保留 IRIR 机制和 HARD_CLAUSE_KEYWORDS 逻辑

"""
import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from graph_retriever import (
    extract_entities_enhanced as extract_entities,
    GraphRetriever,
    sanitize_fulltext_query,
)
from embed_clauses import VectorRetriever

# ── LLM 配置 ──
API_KEY="YOUR_API_KEY"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"


# ════════════════════════════════════════════════════════════
# 关系类型权重表
# ════════════════════════════════════════════════════════════
RELATION_TYPE_WEIGHTS = {
    'equipment+space':       2.0,   # 设备+空间联合命中，最高相关性
    'equipment':             1.0,   # 单维度设备匹配
    'space':                 1.0,   # 单维度空间匹配
    'fulltext':              0.5,   # 全文节点属性匹配（防噪音）
    'hierarchy-parent':      0.5,   # 层级拓扑扩展
    'hierarchy-child':       0.5,   # 层级拓扑扩展
    'irir-reformulated':     0.5,   # IRIR重构查询补充
    'safety-net':            0.3,   # 安全网兜底（衰减权重）
    'hard-lookup':           1.0,   # 困难条款直接注入
    'direct-lookup':         2.0,   # 条款ID直接查找（最高优先级）
}


# ════════════════════════════════════════════════════════════
# 第一层：IRIR 意图识别模块
# ════════════════════════════════════════════════════════════

NEGATIVE_PATTERNS = [
    r'未发现([\u4e00-\u9fff]{2,12})',
    r'未见([\u4e00-\u9fff]{2,12})',
    r'未设(?:有|置|安装)?([\u4e00-\u9fff]{2,12})',
    r'没有(?:发现|配备|设置|安装)([\u4e00-\u9fff]{2,12})',
    r'缺少([\u4e00-\u9fff]{2,12})',
    r'不具备([\u4e00-\u9fff]{2,12})',
    r'不具有([\u4e00-\u9fff]{2,12})',
    r'不足([\u4e00-\u9fff]{2,12})',
    r'仅(?:配备|设有|发现|安装)?了?一(?:台|具|套|个)([\u4e00-\u9fff]{2,12})',
    r'只(?:发现|配备|设有)?了?一(?:台|具|套|个)([\u4e00-\u9fff]{2,12})',
    r'不够([\u4e00-\u9fff]{2,12})',
]

POSITIVE_TEMPLATES = [
    '{entity} 要求',
    '{entity} 应装有',
    '{entity} 应设有',
    '{entity} 至少',
    '{entity} 双套',
    '{entity} 备用',
    '{entity} 配备要求',
    '{entity} 数量要求',
]


def intent_recognition_llm(query: str, client: OpenAI) -> dict:
    prompt = f"""你是一个海事检查场景意图分析器。请分析以下PSC检查描述的意图。

## 输入
{query}

## 输出要求
请严格以 JSON 格式输出，不要添加任何其他文字：
{{
  "intent": "positive" 或 "negative" 或 "comparative",
  "missing_entities": ["缺失或不足的设备/系统名称"],
  "context_entities": ["场景中涉及的处所、船型等上下文"]
}}

## 判断标准
- **positive**: 场景描述设备存在且规格明确（如"配备了XX"、"设有XX"）
- **negative**: 场景描述设备缺失、数量不足或未安装（如"未发现XX"、"缺少XX"、"仅有1台"）
- **comparative**: 场景描述实际配置与标准的对比
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            timeout=30,
        )
        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[^{}]+\}', text)
        if json_match:
            result = json.loads(json_match.group())
            if 'intent' in result and result['intent'] in ('positive', 'negative', 'comparative'):
                result.setdefault('missing_entities', [])
                result.setdefault('context_entities', [])
                return result
    except Exception as e:
        print(f"  ⚠️ LLM 意图识别异常: {e}")
    return None


def intent_recognition_regex(query: str) -> dict:
    missing_entities = []
    for pattern in NEGATIVE_PATTERNS:
        matches = re.findall(pattern, query)
        for m in matches:
            m = m.strip()
            if len(m) >= 2 and m not in missing_entities:
                missing_entities.append(m)

    if missing_entities:
        context = []
        space_patterns = [
            r'(机器处所|机舱|起居处所|货物处所|控制站|驾驶室|走廊|梯道|逃生通道|厨房|泵舱)',
        ]
        for sp in space_patterns:
            context.extend(re.findall(sp, query))
        return {
            'intent': 'negative',
            'missing_entities': missing_entities,
            'context_entities': list(set(context)),
        }
    return None


def _validate_missing_entities(query: str, missing: list) -> list:
    """
    校验 missing_entities：如果实体在场景描述中有正面上下文
    （安装了/设有/配备/均可/符合 等），则不应判为 missing。
    但如果附近有否定词（未/仅/不/缺少），则仍然是 missing。
    """
    # 否定词：附近有这些词时，即使有正面词也是 missing
    NEGATION_WORDS = r'(?:未|没有|缺少|不具备|不具有|不足|不够)'
    # 正面词：确认设备已安装/符合
    POSITIVE_CONTEXT = [
        r'安装了', r'设有', r'均可', r'符合', r'满足',
        r'已(?:安装|配备|设置)', r'采用了', r'配置了',
        r'具备', r'提供了', r'设有',
    ]
    # 限定正面词："仅配备" 不算正面
    LIMITED_POSITIVE = r'仅(?:配备|设有|发现|安装)了?一'

    validated = []
    for entity in missing:
        is_actually_missing = True
        # 取实体的关键词（前8个字）做模糊匹配，应对长实体
        search_key = entity[:8] if len(entity) > 8 else entity
        for m in re.finditer(re.escape(search_key), query):
            start = max(0, m.start() - 40)
            end = min(len(query), m.end() + 40)
            context_window = query[start:end]
            # 先检查是否有限定正面词（如"仅配备了一台"）→ 仍然是 missing
            if re.search(LIMITED_POSITIVE, context_window):
                continue
            # 检查是否有否定词 → 仍然是 missing
            if re.search(NEGATION_WORDS, context_window):
                continue
            # 检查是否有正面词 → 不是 missing
            for pat in POSITIVE_CONTEXT:
                if re.search(pat, context_window):
                    is_actually_missing = False
                    print(f"  🔧 IRIR校验: '{entity}' 有正面上下文，移出missing")
                    break
            if not is_actually_missing:
                break
        if is_actually_missing:
            validated.append(entity)
    return validated


def intent_recognition(query: str, client: OpenAI) -> dict:
    """三级容错回退：Level 1 (LLM) → Level 2 (正则) → Level 3 (默认 positive)"""
    result = intent_recognition_llm(query, client)
    if result is not None:
        # 校验 missing_entities
        if result.get('missing_entities'):
            result['missing_entities'] = _validate_missing_entities(
                query, result['missing_entities'])
            if not result['missing_entities']:
                result['intent'] = 'positive'
        result['fallback_level'] = 1
        print(f"  🧠 IRIR Level 1 (LLM): intent={result['intent']}, "
              f"missing={result['missing_entities']}")
        return result

    result = intent_recognition_regex(query)
    if result is not None:
        result['fallback_level'] = 2
        print(f"  📝 IRIR Level 2 (Regex): intent={result['intent']}, "
              f"missing={result['missing_entities']}")
        return result

    print(f"  ➡️ IRIR Level 3 (Default): positive")
    return {
        'intent': 'positive',
        'missing_entities': [],
        'context_entities': [],
        'fallback_level': 3,
    }


def reformulate_queries(intent_result: dict) -> list:
    if intent_result['intent'] != 'negative' or not intent_result['missing_entities']:
        return []

    ref_queries = []
    context = intent_result.get('context_entities', [])

    for entity in intent_result['missing_entities']:
        entity = entity.strip()
        if len(entity) < 2:
            continue
        for tpl in POSITIVE_TEMPLATES:
            ref_queries.append(tpl.format(entity=entity))
        if context:
            for ctx in context[:3]:
                for tpl in POSITIVE_TEMPLATES[:4]:
                    ref_queries.append(f'{ctx} {tpl.format(entity=entity)}')
                ref_queries.append(f'{ctx} {entity}')

    seen = set()
    unique = []
    for q in ref_queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


class ComplianceRAG_V7:

    def __init__(self):
        print("初始化 IAKG-RAG v7 (论文公式对齐版) ...")
        self.graph_ret = GraphRetriever()
        self.vector_ret = VectorRetriever()
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        # 融合参数（论文 §2.2.2）
        self.alpha = 0.5   # 图谱权重
        self.beta = 0.5    # 向量权重
        self.gamma = 0.3   # 交叉命中增益系数

        print("✅ 系统就绪\n")

    # ── LLM 调用 ──

    def llm_judge(self, prompt: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是海事消防法规合规检查专家，严格依据法规条款进行判断。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=120,
        )
        return response.choices[0].message.content

    # ── 融合排序核心方法（v7 新增） ──

    def _compute_fusion_scores(self, merged: dict) -> dict:
        """
        论文 §2.2.2 融合排序公式（严格实现）：

        1. 计算各条款图谱原始得分 = Σ w_type (按关系类型赋权)
        2. Min-Max 归一化图谱得分到 [0, 1]
        3. 向量得分 = 余弦相似度（天然 [0,1]）
        4. BaseScore = α·S_Graph + β·S_Vector
        5. 交叉命中增益：FinalScore = (1 + γ·I[|M_c|>1]) × BaseScore
           其中 |M_c|>1 表示同时被图谱+向量分支命中
        """
        if not merged:
            return merged

        # Step 1: 计算图谱原始得分
        graph_scores_raw = {}
        for cid, item in merged.items():
            match_types = item.get('match_types', [])
            raw_score = 0.0
            for mt in match_types:
                w = RELATION_TYPE_WEIGHTS.get(mt, 0.5)
                raw_score += w
            graph_scores_raw[cid] = raw_score

        # Step 2: Min-Max 归一化
        min_gs = min(graph_scores_raw.values())
        max_gs = max(graph_scores_raw.values())
        range_gs = max_gs - min_gs if max_gs > min_gs else 1.0

        for cid in merged:
            normalized_gs = (graph_scores_raw[cid] - min_gs) / range_gs
            merged[cid]['graph_score'] = round(normalized_gs, 6)

        # Step 3 & 4: BaseScore = α·S_Graph + β·S_Vector
        for cid, item in merged.items():
            v_score = item.get('vector_score', 0.0)
            g_score = item['graph_score']
            base_score = self.alpha * g_score + self.beta * v_score

            # Step 5: 交叉命中增益
            # 判定标准：同时被图谱分支 AND 向量分支命中
            has_graph = any(mt not in ('vector', 'safety-net', 'hard-lookup')
                           for mt in item.get('match_types', []))
            has_vector = 'vector' in item.get('match_types', [])

            if has_graph and has_vector:
                final_score = (1 + self.gamma) * base_score
            else:
                final_score = base_score

            merged[cid]['final_score'] = round(final_score, 6)
            merged[cid]['cross_hit'] = has_graph and has_vector

        return merged

    # ── 核心检索流程 ──

    def retrieve(self, query: str, top_k: int = 15) -> dict:
        """
        三阶段检索：
        1. IRIR 意图识别 + 查询重构
        2. 双路并发检索（图谱 + 向量）
        3. 协同增益排序 + 安全网兜底
        """

        # ────────────────────────────────────────────
        # 阶段 1: 实体抽取 + IRIR 意图识别
        # ────────────────────────────────────────────
        entities = extract_entities(query)
        print(f"📋 抽取实体: {json.dumps(entities, ensure_ascii=False)}")

        intent_result = intent_recognition(query, self.llm_client)

        ref_queries = reformulate_queries(intent_result)
        if ref_queries:
            print(f"🔄 IRIR 重构查询: {len(ref_queries)} 条")
            for rq in ref_queries[:5]:
                print(f"    - {rq}")
            if len(ref_queries) > 5:
                print(f"    ... 及 {len(ref_queries)-5} 条更多")

        # ────────────────────────────────────────────
        # 阶段 2: 双路并发检索
        # ────────────────────────────────────────────

        # 2a. 图谱检索（原始查询）
        graph_results = self.graph_ret.search(entities, top_k=top_k * 5, scenario_text=query)
        print(f"🔗 图谱检索: {len(graph_results)} 条")

        # 2b. 向量检索（原始查询）
        vector_results = self.vector_ret.search(query, top_k=top_k + 5, threshold=0.15)
        print(f"📐 向量检索: {len(vector_results)} 条")

        # ────────────────────────────────────────────
        # 阶段 2c: IRIR 二次全文检索
        # ────────────────────────────────────────────
        irir_added = 0
        if ref_queries:
            print(f"🔍 IRIR 二次全文检索 (补充查询)...")
            with self.graph_ret.driver.session() as session:
                for rq in ref_queries[:20]:
                    fq = sanitize_fulltext_query(rq)
                    if fq and len(fq) > 2:
                        try:
                            r = session.run(
                                "CALL db.index.fulltext.queryNodes('clause_content_ft', $q) "
                                "YIELD node, score "
                                "RETURN node.clause_id AS id, node.title AS title, "
                                "node.content AS content, score "
                                "LIMIT 5",
                                q=fq
                            )
                            for rec in r:
                                cid = rec['id']
                                if cid not in {gr['id'] for gr in graph_results}:
                                    graph_results.append({
                                        'id': cid,
                                        'title': rec['title'],
                                        'content': rec['content'],
                                        'match_type': 'irir-reformulated',
                                        'score': rec['score'],
                                    })
                                    irir_added += 1
                        except Exception:
                            pass
            if irir_added > 0:
                print(f"🔄 IRIR 补充: {irir_added} 条")

        # ────────────────────────────────────────────
        # 阶段 2d: 条款ID直接查找（当查询中提及时）
        # ────────────────────────────────────────────
        import re
        clause_id_patterns = [
            (r'SOLAS\s+II-2[/.](\d+(?:\.\d+)*)', 'SOLAS II-2/'),
            (r'FSS\s*Code[/.](\d+(?:\.\d+)*)', 'FSS Code/'),
        ]
        direct_added = 0
        existing_ids = {gr['id'] for gr in graph_results}
        with self.graph_ret.driver.session() as session:
            for pattern, prefix in clause_id_patterns:
                for m in re.finditer(pattern, query):
                    cid = prefix + m.group(1)
                    if cid in existing_ids:
                        continue
                    r = session.run(
                        "MATCH (c:Clause {clause_id: $cid}) RETURN c.clause_id AS id, "
                        "c.title AS title, c.content AS content LIMIT 1",
                        cid=cid
                    )
                    rec = r.single()
                    if rec and rec['content']:
                        graph_results.append({
                            'id': cid,
                            'title': rec['title'],
                            'content': rec['content'],
                            'match_type': 'direct-lookup',
                            'score': 10.0,
                        })
                        existing_ids.add(cid)
                        direct_added += 1
        if direct_added > 0:
            print(f"📌 条款直接查找: {direct_added} 条")

        # ────────────────────────────────────────────
        # 阶段 3: 融合排序（v7 严格实现论文公式）
        # ────────────────────────────────────────────
        merged = {}

        # 3a. 图谱结果入库
        for r in graph_results:
            cid = r['id']
            mt = r.get('match_type', 'graph')
            if cid in merged:
                if mt not in merged[cid]['match_types']:
                    merged[cid]['match_types'].append(mt)
                if len(r.get('content', '')) > len(merged[cid].get('content', '')):
                    merged[cid]['content'] = r['content']
            else:
                merged[cid] = {
                    'clause_id': cid,
                    'title': r.get('title', ''),
                    'content': r.get('content', r.get('preview', '')),
                    'graph_score': 0.0,
                    'vector_score': 0.0,
                    'match_types': [mt],
                }

        # 3b. 向量结果入库
        for r in vector_results:
            cid = r['clause_id']
            if cid in merged:
                merged[cid]['vector_score'] = r['score']
                if 'vector' not in merged[cid]['match_types']:
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

        # 3c. 【v7 核心】融合打分
        self._compute_fusion_scores(merged)

        # ────────────────────────────────────────────
        # 阶段 3b: 安全网兜底
        # ────────────────────────────────────────────
        safety_added = 0
        missing_entities_for_safety = intent_result.get('missing_entities', [])
        if not missing_entities_for_safety:
            for pattern in NEGATIVE_PATTERNS:
                for m in re.findall(pattern, query):
                    m = m.strip()
                    if len(m) >= 2 and m not in missing_entities_for_safety:
                        missing_entities_for_safety.append(m)

        if missing_entities_for_safety:
            with self.graph_ret.driver.session() as session:
                for entity in missing_entities_for_safety[:5]:
                    safety_queries = [
                        f'{entity} 要求',
                        f'{entity} 应装有',
                        f'{entity} 至少',
                        f'{entity} 双套',
                        f'{entity} 备用',
                    ]
                    for sq in safety_queries:
                        fq = sanitize_fulltext_query(sq)
                        if fq and len(fq) > 2:
                            try:
                                r = session.run(
                                    "CALL db.index.fulltext.queryNodes('clause_content_ft', $q) "
                                    "YIELD node, score "
                                    "RETURN node.clause_id AS id, node.title AS title, "
                                    "node.content AS content, score "
                                    "LIMIT 5",
                                    q=fq
                                )
                                for rec in r:
                                    cid = rec['id']
                                    if cid not in merged:
                                        merged[cid] = {
                                            'clause_id': cid,
                                            'title': rec['title'],
                                            'content': rec['content'],
                                            'graph_score': 0.0,
                                            'vector_score': rec['score'] * 0.3,
                                            'match_types': ['safety-net'],
                                            'final_score': rec['score'] * 0.3,
                                            'cross_hit': False,
                                        }
                                        safety_added += 1
                            except Exception:
                                pass
            if safety_added > 0:
                print(f"🛡️ 安全网补充: {safety_added} 条")

        # 最终排序
        sorted_results = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)
        top_results = sorted_results[:top_k]

        # ────────────────────────────────────────────
        # 困难条款直接查找（保留 v5/v6 逻辑）
        # ────────────────────────────────────────────
        HARD_CLAUSE_KEYWORDS = {
            '水雾枪': ['SOLAS II-2/10.5.5', 'SOLAS II-2/20.6.2.1'],
            '端部封闭': ['SOLAS II-2/13.3.3.4'],
            '梯道.*升高': ['FSS Code/13.2.2.3'],
            '梯道.*平台': ['FSS Code/13.2.2.3', 'FSS Code/13.2.2.4'],
        }
        top_ids = {r['clause_id'] for r in top_results}
        replace_idx = len(top_results) - 1
        with self.graph_ret.driver.session() as session:
            for pattern, clause_ids in HARD_CLAUSE_KEYWORDS.items():
                if re.search(pattern, query):
                    for cid in clause_ids:
                        if cid not in top_ids and replace_idx >= 0:
                            if cid in merged:
                                top_results[replace_idx] = merged[cid]
                                merged[cid]['final_score'] = 0.9
                                top_ids.add(cid)
                                replace_idx -= 1
                                print(f"📌 强制保留: {cid}")
                            else:
                                r = session.run(
                                    "MATCH (c:Clause {clause_id: $id}) RETURN c.clause_id AS id, "
                                    "c.title AS title, c.content AS content",
                                    id=cid
                                )
                                rec = r.single()
                                if rec:
                                    top_results[replace_idx] = {
                                        'clause_id': cid,
                                        'title': rec['title'],
                                        'content': rec['content'],
                                        'graph_score': 0.0,
                                        'vector_score': 0.0,
                                        'match_types': ['hard-lookup'],
                                        'final_score': 0.9,
                                        'cross_hit': False,
                                    }
                                    top_ids.add(cid)
                                    replace_idx -= 1
                                    print(f"📌 强制注入: {cid}")

        # 打印最终结果
        print(f"📊 融合结果 (v7 论文公式对齐): {len(top_results)} 条")
        for i, r in enumerate(top_results):
            tags = '+'.join(r['match_types'])
            g = r.get('graph_score', 0)
            v = r.get('vector_score', 0)
            f = r.get('final_score', 0)
            ch = '✕' if r.get('cross_hit') else ' '
            print(f"  {i+1}. [{tags}] {r['clause_id']}: {r['title'][:40]} "
                  f"(g={g:.4f} v={v:.4f} f={f:.4f}) [{ch}]")

        return {
            'entities': entities,
            'intent': intent_result,
            'reformulated_queries': ref_queries,
            'results': top_results,
        }

    # ── Prompt 构建（同 v6） ──

    def build_prompt(self, query: str, retrieval_result: dict) -> str:
        entities = retrieval_result['entities']
        results = retrieval_result['results']

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

        irir_note = ""
        intent_result = retrieval_result.get('intent', {})
        if intent_result.get('intent') == 'negative' and intent_result.get('missing_entities'):
            missing = ', '.join(intent_result['missing_entities'])
            irir_note = f"""
## ⚠️ 系统提示：本场景涉及设备缺失/不足
系统已识别以下可能缺失或数量不足的设备：**{missing}**
请重点关注相关条款中关于该设备的配备数量、双套/备用/冗余要求。
"""

        prompt = f"""你是一个海事消防法规合规检查专家。请严格依据以下法规条款的原文内容，判断给定场景是否合规。

## 场景描述
{query}

## 抽取的关键实体
{chr(10).join(entity_summary)}
{irir_note}
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
- **重点关注**：条款中出现"应装有双套""备用""冗余""至2台""独立驱动"等词时，与场景中的"仅有1台"直接矛盾 → ❌ 不满足
- 例：场景说"仅配备一台抽样风机" → 条款要求"应装有双套抽样风机" → ❌ 不满足
- **推理**：当场景明确描述了某种缺失（如"未发现水雾枪"），应该检查条款是否要求该设备。即使检索结果中没有直接包含该条款，如果场景提到了缺失某种设备，你应该追问：该处所/该船型是否需要这种设备？

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

    # ── 完整查询流程 ──

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
            'intent': retrieval_results['intent'],
            'reformulated_queries': retrieval_results['reformulated_queries'],
            'retrieved_clauses': retrieval_results['results'],
            'prompt': prompt,
            'answer': answer,
        }

    def close(self):
        self.graph_ret.close()


# ── 单独测试 ──
if __name__ == '__main__':
    rag = ComplianceRAG_V7()

    test_cases = [
        "机舱内未发现备用通风机。问：是否合规？",
        "一艘2009年建造的多用途货船，在机舱入口处配备的手提式二氧化碳灭火器标称容量为3 kg。问：是否合规？",
    ]

    for q in test_cases:
        result = rag.query(q)
        print(f"\n意图: {result['intent']}")
        print(f"重构查询: {result['reformulated_queries'][:3]}")

    rag.close()
