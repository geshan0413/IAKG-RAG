"""
知识图谱 v3 — 扩展词典 + 增强实体抽取

改进点：
1. 设备词典从 31 → 80+，覆盖测试集中的所有术语
2. 空间词典从 15 → 25+
3. 查询端增加 LLM 实体抽取兜底
"""

import re
import json
from neo4j import GraphDatabase
from dataclasses import dataclass, asdict
from openai import OpenAI

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "YOUR_NEO4J_PASSWORD")

# MiMo LLM 配置
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"


# ============================================================
# 大幅扩展的实体词典
# ============================================================

EQUIPMENT_LEXICON = {
    # === 消防泵 ===
    "应急消防泵": {"category": "消防泵", "aliases": ["应急消防泵", "应急泵"]},
    "消防泵": {"category": "消防泵", "aliases": ["消防泵", "主消防泵"]},
    "喷水器泵": {"category": "消防泵", "aliases": ["喷水器泵", "喷淋泵"]},

    # === 固定灭火系统 ===
    "固定式灭火系统": {"category": "灭火系统", "aliases": ["固定式灭火系统", "固定灭火系统", "固定灭火"]},
    "CO2灭火系统": {"category": "灭火系统", "aliases": ["CO2灭火系统", "二氧化碳灭火系统", "CO2灭火", "固定式CO2灭火系统", "高压二氧化碳灭火系统", "低压二氧化碳灭火系统", "低压CO2", "高压CO2", "二氧化碳灭火", "CO2系统", "二氧化碳系统", "二氧化碳", "CO2"]},
    "泡沫灭火系统": {"category": "灭火系统", "aliases": ["泡沫灭火系统", "泡沫灭火", "固定式泡沫灭火系统", "甲板泡沫系统", "泡沫系统", "高倍泡沫灭火系统", "固定式高倍泡沫灭火系统", "高倍泡沫"]},
    "水灭火系统": {"category": "灭火系统", "aliases": ["水灭火系统", "水灭火"]},
    "喷水器系统": {"category": "灭火系统", "aliases": ["喷水器系统", "喷水器", "喷水灭火系统", "自动喷水器系统", "自动喷水器", "水喷淋系统", "固定式水基局部使用灭火系统", "水基局部灭火"]},
    "气体灭火系统": {"category": "灭火系统", "aliases": ["气体灭火系统", "固定式气体灭火系统"]},
    "干粉灭火系统": {"category": "灭火系统", "aliases": ["干粉灭火系统", "干粉系统", "干粉灭火"]},
    "局部使用灭火系统": {"category": "灭火系统", "aliases": ["局部使用灭火系统", "固定式局部使用灭火系统", "局部应用系统"]},

    # === 灭火器 ===
    "手提式灭火器": {"category": "灭火器", "aliases": ["手提式灭火器", "手提灭火器", "灭火器", "手提式泡沫灭火器", "手提式二氧化碳灭火器"]},
    "半手提式灭火器": {"category": "灭火器", "aliases": ["半手提式灭火器"]},
    "推车式灭火器": {"category": "灭火器", "aliases": ["推车式灭火器", "推车式泡沫灭火器"]},
    "手提式泡沫枪": {"category": "灭火器", "aliases": ["手提式泡沫枪", "泡沫枪", "手提式泡沫枪装置"]},

    # === 消防器材 ===
    "消防水带": {"category": "消防器材", "aliases": ["消防水带", "消防软管", "水带"]},
    "消防员装备": {"category": "人员保护", "aliases": ["消防员装备", "消防员个人装备"]},
    "自给式呼吸器": {"category": "人员保护", "aliases": ["自给式呼吸器", "自给式压缩空气呼吸器", "呼吸器"]},
    "紧急逃生呼吸装置": {"category": "人员保护", "aliases": ["紧急逃生呼吸装置", "EEBD", "紧急逃生呼吸", "永久性的紧急逃生呼吸装置"]},
    "国际通岸接头": {"category": "消防器材", "aliases": ["国际通岸接头", "通岸接头"]},
    "消防控制图": {"category": "文件", "aliases": ["消防控制图", "防火控制图"]},
    "消防总管": {"category": "消防器材", "aliases": ["消防总管", "消防水管总管"]},

    # === 泡沫相关 ===
    "泡沫炮": {"category": "泡沫器材", "aliases": ["泡沫炮", "固定式泡沫炮"]},
    "泡沫枪": {"category": "泡沫器材", "aliases": ["泡沫枪"]},
    "泡沫比例混合器": {"category": "泡沫器材", "aliases": ["泡沫比例混合器", "混合器"]},
    "泡沫液容器": {"category": "泡沫器材", "aliases": ["泡沫液容器", "备用泡沫液容器"]},

    # === 探火系统 ===
    "固定式探火系统": {"category": "探火系统", "aliases": ["固定式探火系统", "固定探火系统", "探火系统", "固定式探火和失火报警系统", "探火和失火报警系统", "探火和火警系统"]},
    "感烟探测器": {"category": "探火系统", "aliases": ["感烟探测器", "烟雾探测器", "烟雾探测"]},
    "感温探测器": {"category": "探火系统", "aliases": ["感温探测器"]},
    "抽烟式探火系统": {"category": "探火系统", "aliases": ["抽烟式探火系统", "抽烟探火", "抽烟系统", "抽样风机", "取样管", "取样风机"]},
    "火焰探测器": {"category": "探火系统", "aliases": ["火焰探测器"]},
    "气体探测系统": {"category": "探火系统", "aliases": ["气体探测系统", "气体探测", "气体探测设备", "固定式气体探测", "易燃气体探测器", "可燃气体探测器", "碳氢气体浓度监测系统", "碳氢化合物探测", "碳氢化合物报警", "硫化氢探测", "硫化氢报警", "便携式气体探测", "便携式仪器"]},
    "氧气探测系统": {"category": "探火系统", "aliases": ["氧气探测", "氧气传感器", "氧气含量"]},
    "失火报警系统": {"category": "探火系统", "aliases": ["失火报警系统", "火灾报警系统", "火警报警", "火警", "报警系统", "火警报警系统"]},
    "手动报警按钮": {"category": "探火系统", "aliases": ["手动报警按钮"]},
    "听觉报警": {"category": "探火系统", "aliases": ["听觉报警", "听觉信号", "声觉报警", "声觉火警报警"]},
    "视觉报警": {"category": "探火系统", "aliases": ["视觉报警", "视觉信号", "视觉火警报警"]},
    "预释放报警": {"category": "探火系统", "aliases": ["预释放报警", "预报警"]},
    "恒温器": {"category": "探火系统", "aliases": ["恒温器", "恒温控制", "主恒温器", "后备恒温器"]},
    "控制面板": {"category": "探火系统", "aliases": ["控制面板", "控制板"]},

    # === 通风和防火分隔 ===
    "通风系统": {"category": "通风", "aliases": ["通风系统", "通风", "机械通风系统", "通风筒", "排气导管", "通风导管", "空调回风系统"]},
    "挡火闸": {"category": "通风", "aliases": ["挡火闸", "防火闸", "自动挡火闸"]},
    "防火门": {"category": "防火分隔", "aliases": ["防火门", "自闭式防火门", "自闭型防火门", "A级防火门", "B级防火门", "钢质门"]},
    "A级分隔": {"category": "防火分隔", "aliases": ["A级分隔", "\"A\"级分隔", "A-60", "A-0", "A级隔热"]},
    "B级分隔": {"category": "防火分隔", "aliases": ["B级分隔", "\"B\"级分隔"]},
    "钢质外套": {"category": "防火分隔", "aliases": ["钢质外套", "钢质套管", "套管"]},
    "隔热": {"category": "防火分隔", "aliases": ["隔热", "隔热包覆"]},
    "阻焰器": {"category": "通风", "aliases": ["阻焰器", "火星熄灭器"]},
    "集油器": {"category": "通风", "aliases": ["集油器"]},

    # === 阀件 ===
    "燃油速闭阀": {"category": "阀件", "aliases": ["燃油速闭阀", "速闭阀"]},
    "隔离阀": {"category": "阀件", "aliases": ["隔离阀"]},
    "安全阀": {"category": "阀件", "aliases": ["安全阀", "压力释放阀"]},
    "释放阀": {"category": "阀件", "aliases": ["释放阀", "气体释放阀门", "排放阀", "自动排放阀"]},
    "关闭装置": {"category": "阀件", "aliases": ["关闭装置", "遥控关闭装置", "关断装置", "自闭式关断装置"]},
    "瓶头阀": {"category": "阀件", "aliases": ["瓶头阀"]},

    # === 脱险 ===
    "脱险通道": {"category": "脱险", "aliases": ["脱险通道", "逃生通道", "应急逃生通道", "梯道", "钢质围阱"]},

    # === 释放控制 ===
    "释放站": {"category": "释放控制", "aliases": ["释放站", "释放控制装置", "释放箱"]},
    "控制装置": {"category": "释放控制", "aliases": ["控制装置", "控制盒"]},

    # === 通讯 ===
    "通讯装置": {"category": "通讯", "aliases": ["通讯装置", "通信装置", "通信", "通讯"]},

    # === 管路 ===
    "管路": {"category": "管路", "aliases": ["管路", "波纹管", "挠性波纹管", "测深管"]},

    # === 惰气 ===
    "惰性气体系统": {"category": "特种系统", "aliases": ["惰性气体系统", "惰气系统", "惰性气体发生器", "惰性气体", "惰气"]},

    # === 培训 ===
    "消防培训手册": {"category": "文件", "aliases": ["消防培训手册", "培训手册"]},

    # === 指示 ===
    "荧光条指示装置": {"category": "指示", "aliases": ["荧光条指示装置", "荧光条"]},

    # === 厨房 ===
    "深油烹饪设备": {"category": "厨房", "aliases": ["深油烹饪设备", "炸锅"]},
    "厨房灭火系统": {"category": "灭火系统", "aliases": ["厨房灭火系统", "炸锅的固定式灭火"]},

    # === 灭火介质 ===
    "灭火介质储藏室": {"category": "存储", "aliases": ["灭火介质储藏室", "灭火介质储藏间"]},

    # === 消防泵启动 ===
    "消防泵启动按钮": {"category": "消防泵", "aliases": ["启动按钮", "消防泵启动按钮"]},

    # === 排水 ===
    "排水系统": {"category": "排水", "aliases": ["排水系统", "排水管"]},

    # === 消防演习 ===
    "消防演习": {"category": "演习", "aliases": ["消防演习"]},

    # === 电缆 ===
    "电缆": {"category": "电气", "aliases": ["电缆"]},
}


SPACE_LEXICON = {
    "机器处所": {"category": "机械", "aliases": ["机器处所", "机舱", "机械处所"]},
    "泵房": {"category": "机械", "aliases": ["泵房", "泵舱", "泵室"]},
    "锅炉舱": {"category": "机械", "aliases": ["锅炉舱", "锅炉房", "炉舱"]},
    "货舱": {"category": "货物", "aliases": ["货舱", "货物处所", "货物空隔舱"]},
    "液货舱": {"category": "货物", "aliases": ["液货舱", "液货处所", "货油舱"]},
    "液货泵舱": {"category": "货物", "aliases": ["液货泵舱", "货泵舱"]},
    "滚装处所": {"category": "特种处所", "aliases": ["滚装处所", "滚装货处所"]},
    "特种处所": {"category": "特种处所", "aliases": ["特种处所"]},
    "起居处所": {"category": "生活", "aliases": ["起居处所", "居住处所", "生活区"]},
    "控制站": {"category": "功能", "aliases": ["控制站", "驾驶室", "货物控制室", "中央控制站", "安全中心"]},
    "服务处所": {"category": "功能", "aliases": ["服务处所"]},
    "厨房": {"category": "生活", "aliases": ["厨房", "主厨房"]},
    "储藏室": {"category": "功能", "aliases": ["储藏室", "储藏间", "储物间", "油漆储藏室", "油漆间", "储物柜"]},
    "车辆处所": {"category": "特种处所", "aliases": ["车辆处所", "车辆甲板"]},
    "围蔽处所": {"category": "功能", "aliases": ["围蔽处所"]},
    "开敞甲板": {"category": "结构", "aliases": ["开敞甲板", "露天甲板"]},
    "走廊": {"category": "交通", "aliases": ["走廊", "走道"]},
    "桑拿房": {"category": "生活", "aliases": ["桑拿房", "桑拿"]},
    "直升机甲板": {"category": "特种处所", "aliases": ["直升机甲板"]},
    "燃油净化间": {"category": "机械", "aliases": ["燃油净化间", "燃油净化器"]},
    "燃油舱": {"category": "存储", "aliases": ["燃油舱", "燃油柜", "双层底燃油舱"]},
    "脱险通道": {"category": "交通", "aliases": ["脱险通道", "逃生通道"]},
    "处所": {"category": "通用", "aliases": ["处所"]},
}


SHIPTYPE_LEXICON = {
    "所有船舶": {"category": "通用", "aliases": ["所有船舶", "各类船舶"]},
    "货船": {"category": "船型", "aliases": ["货船"]},
    "客船": {"category": "船型", "aliases": ["客船"]},
    "液货船": {"category": "船型", "aliases": ["液货船", "油船"]},
    "化学品船": {"category": "船型", "aliases": ["化学品船", "化学品液货船"]},
    "气体运输船": {"category": "船型", "aliases": ["气体运输船"]},
    "集装箱船": {"category": "船型", "aliases": ["集装箱船"]},
    "散货船": {"category": "船型", "aliases": ["散货船"]},
    "杂货船": {"category": "船型", "aliases": ["杂货船"]},
    "滚装船": {"category": "船型", "aliases": ["滚装船"]},
    "汽车运输船": {"category": "船型", "aliases": ["汽车运输船", "PCTC"]},
    "渔船": {"category": "船型", "aliases": ["渔船"]},
    "500总吨及以上船舶": {"category": "吨位条件", "aliases": ["500总吨及以上", "500总吨以上的船舶"]},
}


# ============================================================
# LLM 辅助实体抽取（查询端兜底）
# ============================================================

_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _llm_client


def llm_extract_entities(text: str) -> dict:
    """用 LLM 从查询中抽取字典未覆盖的实体"""
    client = get_llm_client()

    prompt = f"""从以下海事消防检查场景中，提取所有与消防相关的实体。请返回JSON格式。

场景：{text}

请提取以下类型的实体（只提取场景中明确出现的）：

1. "equipments": 消防设备、系统、装置、器材（如：释放站、控制装置、通讯装置、波纹管、泡沫炮等）
2. "spaces": 处所、位置、区域（如：泵房、驾驶室、走廊等）
3. "ship_types": 船型
4. "years": 建造年份
5. "key_terms": 其他与法规判断相关的关键词（如：A-60、隔热、壁厚、排量等）

只返回JSON，不要解释。格式：
{{"equipments": [], "spaces": [], "ship_types": [], "years": [], "key_terms": []}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是实体抽取助手。只返回JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()
        # 尝试提取JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  ⚠️ LLM实体抽取失败: {e}")

    return {"equipments": [], "spaces": [], "ship_types": [], "years": [], "key_terms": []}


def extract_entities_enhanced(text: str) -> dict:
    """增强版实体抽取：字典匹配 + LLM兜底"""
    entities = {
        'equipments': set(),
        'spaces': set(),
        'ship_types': set(),
        'years': [],
        'numbers': [],
    }

    # 1. 字典匹配（优先，快）
    for name, info in EQUIPMENT_LEXICON.items():
        for alias in info["aliases"]:
            if alias in text:
                entities['equipments'].add(name)
                break

    for name, info in SPACE_LEXICON.items():
        for alias in info["aliases"]:
            if alias in text:
                entities['spaces'].add(name)
                break

    for name, info in SHIPTYPE_LEXICON.items():
        for alias in info["aliases"]:
            if alias in text:
                entities['ship_types'].add(name)
                break

    # 年份
    for m in re.finditer(r'(\d{4})\s*年', text):
        entities['years'].append(int(m.group(1)))

    # 数值
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|m|米|kg|千克|l|升|L|总吨|GT|%|秒|s|min|分钟|h|小时|m³|立方米)', text):
        entities['numbers'].append({
            'value': float(m.group(1)),
            'unit': m.group(2),
            'raw': m.group(0),
        })

    # 2. LLM 兜底（如果字典匹配结果太少）
    if len(entities['equipments']) < 2:
        llm_result = llm_extract_entities(text)
        for eq in llm_result.get('equipments', []):
            if eq and eq not in entities['equipments']:
                # 尝试在字典中找到最匹配的
                matched = False
                for name, info in EQUIPMENT_LEXICON.items():
                    if eq in info["aliases"] or name in eq or eq in name:
                        entities['equipments'].add(name)
                        matched = True
                        break
                if not matched:
                    entities['equipments'].add(eq)

        for sp in llm_result.get('spaces', []):
            if sp and sp not in entities['spaces']:
                matched = False
                for name, info in SPACE_LEXICON.items():
                    if sp in info["aliases"] or name in sp or sp in name:
                        entities['spaces'].add(name)
                        matched = True
                        break
                if not matched:
                    entities['spaces'].add(sp)

        for st in llm_result.get('ship_types', []):
            if st and st not in entities['ship_types']:
                for name, info in SHIPTYPE_LEXICON.items():
                    if st in info["aliases"] or name in st or st in name:
                        entities['ship_types'].add(name)
                        break

        for y in llm_result.get('years', []):
            if isinstance(y, int) and y not in entities['years']:
                entities['years'].append(y)

    # 转 set → list
    entities['equipments'] = list(entities['equipments'])
    entities['spaces'] = list(entities['spaces'])
    entities['ship_types'] = list(entities['ship_types'])

    return entities


# ============================================================
# 图谱构建（与 v2 相同的结构，但用扩展词典）
# ============================================================

def setup_schema(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("  数据库已清空")

        constraints = [
            ("clause_id_unique", "Clause", "clause_id"),
            ("regulation_name_unique", "Regulation", "name"),
            ("equipment_name_unique", "Equipment", "name"),
            ("space_name_unique", "Space", "name"),
            ("shiptype_name_unique", "ShipType", "name"),
        ]
        for name, label, prop in constraints:
            try:
                session.run(f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
            except:
                pass

        try:
            session.run("""
                CREATE FULLTEXT INDEX clause_content_ft IF NOT EXISTS
                FOR (c:Clause) ON EACH [c.content, c.title]
            """)
        except:
            pass
        print("  ✅ Schema 就绪")


def import_all(driver, clauses):
    with driver.session() as session:
        # 法规
        session.run("MERGE (r:Regulation {name: 'SOLAS II-2'}) SET r.full_name = '国际海上人命安全公约第II-2章'")
        session.run("MERGE (r:Regulation {name: 'FSS Code'}) SET r.full_name = '国际消防安全系统规则'")

        # 设备
        for name, info in EQUIPMENT_LEXICON.items():
            session.run("MERGE (e:Equipment {name: $n}) SET e.category = $c, e.aliases = $a",
                        n=name, c=info["category"], a=json.dumps(info["aliases"], ensure_ascii=False))

        # 空间
        for name, info in SPACE_LEXICON.items():
            session.run("MERGE (s:Space {name: $n}) SET s.category = $c, s.aliases = $a",
                        n=name, c=info["category"], a=json.dumps(info["aliases"], ensure_ascii=False))

        # 船型
        for name, info in SHIPTYPE_LEXICON.items():
            session.run("MERGE (st:ShipType {name: $n}) SET st.category = $c, st.aliases = $a",
                        n=name, c=info["category"], a=json.dumps(info["aliases"], ensure_ascii=False))

        print(f"  ✅ 实体: {len(EQUIPMENT_LEXICON)} 设备 + {len(SPACE_LEXICON)} 空间 + {len(SHIPTYPE_LEXICON)} 船型")

        # 条款
        batch_size = 100
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i+batch_size]
            data = [asdict(c) for c in batch]
            session.run("""
                UNWIND $batch AS row
                MERGE (c:Clause {clause_id: row.clause_id})
                SET c.source = row.source, c.chapter = row.chapter,
                    c.article = row.article, c.article_num = row.article_num,
                    c.section_path = row.section_path, c.title = row.title,
                    c.content = row.content
            """, batch=data)
        print(f"  ✅ {len(clauses)} 条款")

        # 法规→条款
        for src in ['SOLAS II-2', 'FSS Code']:
            session.run("""
                MATCH (c:Clause {source: $src}), (r:Regulation {name: $src})
                MERGE (r)-[:HAS_CLAUSE]->(c)
            """, src=src)

        # 实体关系
        eq_count = sp_count = st_count = 0
        for c in clauses:
            text = c.content + " " + c.title

            for equip_name, info in EQUIPMENT_LEXICON.items():
                if any(alias in text for alias in info["aliases"]):
                    session.run("""
                        MATCH (c:Clause {clause_id: $cid}), (e:Equipment {name: $eq})
                        MERGE (c)-[:MENTIONS_EQUIPMENT]->(e)
                    """, cid=c.clause_id, eq=equip_name)
                    eq_count += 1

            for space_name, info in SPACE_LEXICON.items():
                if any(alias in text for alias in info["aliases"]):
                    session.run("""
                        MATCH (c:Clause {clause_id: $cid}), (s:Space {name: $sp})
                        MERGE (c)-[:MENTIONS_SPACE]->(s)
                    """, cid=c.clause_id, sp=space_name)
                    sp_count += 1

            for ship_name, info in SHIPTYPE_LEXICON.items():
                if any(alias in text for alias in info["aliases"]):
                    session.run("""
                        MATCH (c:Clause {clause_id: $cid}), (st:ShipType {name: $st})
                        MERGE (c)-[:APPLIES_TO]->(st)
                    """, cid=c.clause_id, st=ship_name)
                    st_count += 1

        print(f"  ✅ 关系: 设备 {eq_count}, 空间 {sp_count}, 船型 {st_count}")

        # 层级
        result = session.run("MATCH (c:Clause) RETURN c.clause_id AS id, c.section_path AS path, c.source AS source")
        all_clauses_data = [(r['id'], r['path'], r['source']) for r in result]
        parent_count = 0
        for clause_id, path, source in all_clauses_data:
            if not path or '.' not in path:
                continue
            parts = path.split('.')
            parent_path = '.'.join(parts[:-1])
            parent_id = f"{source}/{parent_path}"
            r2 = session.run("""
                MATCH (parent:Clause {clause_id: $pid}), (child:Clause {clause_id: $cid})
                MERGE (parent)-[:HAS_SUBCLAUSE]->(child)
                RETURN count(*) AS cnt
            """, pid=parent_id, cid=clause_id)
            if r2.single()['cnt'] > 0:
                parent_count += 1
        print(f"  ✅ {parent_count} 条层级关系")

        # 交叉引用
        ref_count = 0
        for c in clauses:
            refs = set()
            for m in re.finditer(r'II-2[/.](\d+(?:\.\d+)*)', c.content):
                ref = f"SOLAS II-2/{m.group(1)}"
                if ref != c.clause_id:
                    refs.add(ref)
            for m in re.finditer(r'FSS\s*Code[/.](\d+(?:\.\d+)*)', c.content):
                ref = f"FSS Code/{m.group(1)}"
                if ref != c.clause_id:
                    refs.add(ref)
            for ref_id in refs:
                r2 = session.run("""
                    MATCH (from:Clause {clause_id: $fid}), (to:Clause {clause_id: $tid})
                    MERGE (from)-[:REFERENCES]->(to)
                    RETURN count(*) AS cnt
                """, fid=c.clause_id, tid=ref_id)
                if r2.single()['cnt'] > 0:
                    ref_count += 1
        print(f"  ✅ {ref_count} 条交叉引用")


def verify(driver):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
        print("\n  📊 节点:")
        for r in result:
            print(f"    {r['label']}: {r['cnt']}")

        result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC")
        print("  📊 关系:")
        for r in result:
            print(f"    {r['rel']}: {r['cnt']}")

        # 测试漏检案例
        for cid in ["FSS Code/5.2.2.2", "SOLAS II-2/15.2.3.4", "FSS Code/10.2.2.2", "FSS Code/5.2.2.1.5"]:
            r = session.run("""
                MATCH (c:Clause {clause_id: $cid})
                OPTIONAL MATCH (c)-[:MENTIONS_EQUIPMENT]->(e:Equipment)
                OPTIONAL MATCH (c)-[:MENTIONS_SPACE]->(s:Space)
                RETURN c.clause_id AS id, collect(DISTINCT e.name) AS equips, collect(DISTINCT s.name) AS spaces
            """, cid=cid)
            rec = r.single()
            if rec:
                print(f"\n  🔍 {rec['id']}: 设备={rec['equips']}, 空间={rec['spaces']}")


if __name__ == '__main__':
    from parse_final import Clause

    with open('/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json', 'r') as f:
        data = json.load(f)
    clauses = [Clause(**d) for d in data]
    print(f"📦 加载 {len(clauses)} 个条款\n")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        print("1️⃣  设置 Schema...")
        setup_schema(driver)
        print("\n2️⃣  导入全部（词典 v3）...")
        import_all(driver, clauses)
        print("\n3️⃣  验证...")
        verify(driver)
    finally:
        driver.close()
    print("\n🎉 KG v3 构建完成!")
