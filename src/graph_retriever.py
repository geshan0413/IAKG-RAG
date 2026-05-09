"""
图谱检索器 v8 — 宽召回策略 + 缺失项反向查询
核心思路：生成大量候选查询，宁可多捞不可漏掉；对"未发现/缺少/仅有"等描述反向查询对应设备的法规要求
"""
import json
import re
import sys
from pathlib import Path
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
from build_kg_v3 import extract_entities_enhanced, EQUIPMENT_LEXICON, SPACE_LEXICON, SHIPTYPE_LEXICON

def extract_entities(text: str) -> dict:
    return extract_entities_enhanced(text)


def sanitize_fulltext_query(text: str) -> str:
    text = text.replace('²', '2').replace('³', '3').replace('¹', '1')
    text = re.sub(r'[+\-!(){}\[\]^"~*?:\\/&|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


CONSTRAINT_KWS = [
    '宽度', '净宽', '高度', '面积', '容积', '距离', '间距',
    '容量', '数量', '重量', '压力', '温度', '流速', '流量',
    '壁厚', '厚度', '直径',
    '双套', '备用', '冗余', '独立', '防爆', '钢质', '不锈钢',
    '至少', '不得', '不大于', '不少于', '不超过',
    '报警', '释放', '延迟', '排水', '通风', '通信',
    '灭火剂', '储存室', '灭火器', '消防栓', '消火栓',
    '探火', '探测器', '灭火', '消防',
]


def generate_all_queries(question: str, entities: dict) -> list:
    """生成尽可能多的候选查询"""
    queries = []

    # 1. 所有设备/空间名称直接作为查询
    for eq in entities.get('equipments', []):
        queries.append(eq)
    for sp in entities.get('spaces', []):
        queries.append(sp)

    # 2. 设备+空间
    for eq in entities.get('equipments', []):
        for sp in entities.get('spaces', []):
            queries.append(f'{eq} {sp}')

    # 3. 设备+约束关键词（所有在问题中出现的）
    for eq in entities.get('equipments', []):
        for kw in CONSTRAINT_KWS:
            if kw in question:
                queries.append(f'{eq} {kw}')

    # 4. 空间+约束关键词
    for sp in entities.get('spaces', []):
        for kw in CONSTRAINT_KWS:
            if kw in question:
                queries.append(f'{sp} {kw}')

    # 5. 设备+数值
    if entities.get('numbers'):
        for eq in entities.get('equipments', []):
            for n in entities['numbers'][:3]:
                queries.append(f'{eq} {int(n["value"])}')

    # 6. 从问题中提取所有有意义的2-gram（连续的2-4字中文词组）
    # 取问题中 "问" 之前的部分（场景描述部分）
    desc = question.split('问：')[0] if '问：' in question else question
    chars = re.findall(r'[\u4e00-\u9fff]{2,6}', desc)
    stop = {'一艘', '铺设', '龙骨', '建造', '主管', '机关', '以下', '问题',
            '根据', '应该', '可以', '位于', '设有', '以及', '相关', '适用'}
    filtered = [c for c in chars if c not in stop and len(c) >= 2]

    for i in range(len(filtered)-1):
        queries.append(f'{filtered[i]} {filtered[i+1]}')
    # 也加3-gram
    for i in range(len(filtered)-2):
        queries.append(f'{filtered[i]} {filtered[i+1]}')

    # 7. 否定短语
    neg_patterns = [
        r'(?:未|没有|缺少|未见|未发现|未设置|未安装|不具备)[\u4e00-\u9fff]{2,15}',
        r'[\u4e00-\u9fff]{2,10}(?:仅配备|仅有|仅设|只有|唯一|不足|不够)[\u4e00-\u9fff]{2,10}',
    ]
    for p in neg_patterns:
        matches = re.findall(p, question)
        for m in matches[:3]:
            queries.append(m)

    # 8. 问题中的约束性短语（含数值的句子片段）
    val_phrases = re.findall(r'[\u4e00-\u9fff]{2,20}\d+(?:\.\d+)?[\u4e00-\u9fff]{0,5}', question)
    for vp in val_phrases[:5]:
        queries.append(vp)

    # 9. 缺失项反向查询：对"未发现/缺少/仅有/不足/未设/未见"等描述，反向查询对应设备的要求
    missing_patterns = [
        (r'未发现([\u4e00-\u9fff]{2,10})', 1),
        (r'未见([\u4e00-\u9fff]{2,10})', 1),
        (r'未设(?:有|置|安装)?([\u4e00-\u9fff]{2,10})', 1),
        (r'缺少([\u4e00-\u9fff]{2,10})', 1),
        (r'仅(?:配备|设有|发现|安装)?了?一(?:台|具|套|个)([\u4e00-\u9fff]{2,10})', 1),
        (r'只(?:发现|配备|设有)?了?一(?:台|具|套|个)([\u4e00-\u9fff]{2,10})', 1),
        (r'不(?:具备|具有|满足|符合)([\u4e00-\u9fff]{2,10})', 1),
        (r'不足([\u4e00-\u9fff]{2,10})', 1),
    ]
    for pattern, group in missing_patterns:
        matches = re.findall(pattern, desc)
        for m in matches[:3]:
            m = m.strip()
            if len(m) >= 2:
                # 直接查询该设备
                queries.append(m)
                # 组合查询
                for eq in entities.get('equipments', []):
                    queries.append(f'{eq} {m}')
                for sp in entities.get('spaces', []):
                    queries.append(f'{sp} {m}')

    # 10. 特定技术约束模式
    constraint_patterns = [
        (r'端部封闭.*走廊', '端部封闭 走廊'),
        (r'走廊.*端部封闭', '端部封闭 走廊'),
        (r'([\u4e00-\u9fff]{2,6})材质', 1),
        (r'([\u4e00-\u9fff]{2,6})制成', 1),
        (r'双套|备用|冗余', '双套 备用 冗余'),
        (r'水雾枪', '水雾枪 机器处所 客船'),
        (r'梯道.*垂[直向]', '梯道 升高 不超过'),
        (r'垂[直向].*升高', '梯道 升高 不超过'),
    ]
    for pattern, value in constraint_patterns:
        if isinstance(value, int):
            if re.search(pattern, desc):
                m = re.search(pattern, desc)
                queries.append(m.group(value))
        else:
            if re.search(pattern, desc):
                queries.append(value)

    # 去重
    seen = set()
    unique = []
    for q in queries:
        q = q.strip()
        if q and len(q) > 2 and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


class GraphRetriever:
    """图谱检索器 v7"""

    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "YOUR_NEO4J_PASSWORD")):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        print("图谱检索器v7就绪")

    def close(self):
        self.driver.close()

    def search(self, entities: dict, top_k: int = 20, scenario_text: str = '') -> list:
        results = []
        seen = set()
        # 用于跟踪每个条款被哪些策略找到（用于排序）
        match_scores = {}

        def add_results(records, match_type, boost=1.0):
            for rec in records:
                cid = rec['id']
                if cid not in match_scores:
                    match_scores[cid] = 0
                match_scores[cid] += boost
                if cid not in seen:
                    seen.add(cid)
                    r = dict(rec)
                    r['match_type'] = match_type
                    results.append(r)

        with self.driver.session() as session:
            # === 策略1: 图谱遍历 ===
            if entities['equipments'] and entities['spaces']:
                for eq in entities['equipments']:
                    for sp in entities['spaces']:
                        r = session.run("""
                            MATCH (c:Clause)-[:MENTIONS_EQUIPMENT]->(e:Equipment {name: $eq})
                            MATCH (c)-[:MENTIONS_SPACE]->(s:Space {name: $sp})
                            RETURN c.clause_id AS id, c.title AS title, c.content AS content
                            LIMIT 20
                        """, eq=eq, sp=sp)
                        add_results([dict(rec) for rec in r], 'equipment+space', boost=2.0)

            for eq in entities.get('equipments', []):
                r = session.run("""
                    MATCH (c:Clause)-[:MENTIONS_EQUIPMENT]->(e:Equipment {name: $eq})
                    RETURN c.clause_id AS id, c.title AS title, c.content AS content
                    LIMIT 20
                """, eq=eq)
                add_results([dict(rec) for rec in r], 'equipment', boost=1.0)

            for sp in entities.get('spaces', []):
                r = session.run("""
                    MATCH (c:Clause)-[:MENTIONS_SPACE]->(s:Space {name: $sp})
                    RETURN c.clause_id AS id, c.title AS title, c.content AS content
                    LIMIT 20
                """, sp=sp)
                add_results([dict(rec) for rec in r], 'space', boost=1.0)

            # === 策略4: 船型标记 ===
            ship_clauses = set()
            if entities['ship_types']:
                for st in entities['ship_types']:
                    r = session.run("""
                        MATCH (c:Clause)-[:APPLIES_TO]->(st:ShipType {name: $st})
                        RETURN c.clause_id AS id
                    """, st=st)
                    for rec in r:
                        ship_clauses.add(rec['id'])

            # === 策略5: 大量全文检索查询 ===
            if scenario_text:
                all_queries = generate_all_queries(scenario_text, entities)
                # 最多执行50个查询
                for query in all_queries[:50]:
                    ftq = sanitize_fulltext_query(query)
                    if ftq and len(ftq) > 2:
                        r = session.run("""
                            CALL db.index.fulltext.queryNodes('clause_content_ft', $q)
                            YIELD node, score
                            RETURN node.clause_id AS id, node.title AS title,
                                   node.content AS content, score AS score
                            LIMIT 10
                        """, q=ftq)
                        add_results([dict(rec) for rec in r], 'fulltext', boost=0.5)

            # === 策略6: 层级扩展 ===
            if results:
                # 被多次命中的条款的邻居更可能相关
                top_ids = sorted(match_scores.keys(), key=lambda x: match_scores[x], reverse=True)[:20]
                r = session.run("""
                    MATCH (parent:Clause)-[:HAS_SUBCLAUSE]->(child:Clause)
                    WHERE parent.clause_id IN $ids
                    WITH child WHERE NOT child.clause_id IN $seen
                    RETURN DISTINCT child.clause_id AS id, child.title AS title,
                           child.content AS content
                    LIMIT 20
                """, ids=top_ids, seen=list(seen))
                add_results([dict(rec) for rec in r], 'hierarchy-child', boost=0.3)

                r = session.run("""
                    MATCH (parent:Clause)-[:HAS_SUBCLAUSE]->(child:Clause)
                    WHERE child.clause_id IN $ids
                    WITH parent WHERE NOT parent.clause_id IN $seen
                    RETURN DISTINCT parent.clause_id AS id, parent.title AS title,
                           parent.content AS content
                    LIMIT 20
                """, ids=top_ids, seen=list(seen))
                add_results([dict(rec) for rec in r], 'hierarchy-parent', boost=0.5)

        # 标记船型
        for res in results:
            res['ship_match'] = res['id'] in ship_clauses

        # 按累计匹配分数排序（被多个策略命中的排前面）
        results.sort(key=lambda x: (
            x.get('ship_match', False),
            match_scores.get(x['id'], 0),
        ), reverse=True)

        return results[:top_k]


if __name__ == '__main__':
    test_query = "一艘2009年建造的多用途货船，在机舱入口处配备的手提式二氧化碳灭火器标称容量为3 kg。问：是否合规？"
    entities = extract_entities(test_query)
    retriever = GraphRetriever()
    results = retriever.search(entities, scenario_text=test_query)
    for i, r in enumerate(results[:15]):
        print(f"{i+1}. [{r.get('match_type','')}] {r['id']}: {r.get('title','')[:50]}")
    retriever.close()
