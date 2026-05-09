"""
基线对比系统：
1. 纯 LLM（无检索）
2. BM25 + LLM（关键词检索）
3. 朴素 RAG（向量检索）
"""
import json
import os
import sys
import re
import numpy as np
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

# LLM 配置（硅基流动）
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"
INDEX_DIR = '/root/.openclaw/workspace/projects/kg-rag-solas/index'
CLAUSES_JSON = '/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json'


class LLMClient:
    """共享的 LLM 调用逻辑"""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def judge(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
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


class PureLLM:
    """基线1：纯 LLM，无检索"""

    def __init__(self):
        self.llm = LLMClient()
        print("✅ 纯 LLM 基线就绪")

    def build_prompt(self, query: str) -> str:
        prompt = f"""你是一个海事消防法规合规检查专家。请依据你的专业知识，判断给定场景是否合规。

## 场景描述
{query}

## ⚠️ 核心判断原则

你熟悉 SOLAS 第II-2章和 FSS Code 的所有条款，请根据你的知识进行判断。

### 判断流程（必须严格执行）

**第一步：识别相关法规条款**
列出与场景相关的所有 SOLAS II-2 和 FSS Code 条款，包含：
- 条款来源（编号）
- 要求内容（尽可能引用原文）
- 要求类型：强制性要求 / 条件性要求

**第二步：逐项核对**
将场景中的实际做法与每条要求对比：
- ✅ 满足：场景中的做法明确符合该要求
- ❌ 不满足：场景中的做法明确违反该要求
- ⚠️ 未提及：场景描述中没有涉及此项要求

**第三步：综合判定**

判定规则：
1. 如果有任何一项要求被判定为"❌ 不满足"→ **不合规**
2. 如果所有要求都是"✅ 满足"→ **合规**
3. 如果存在"⚠️ 未提及"的要求：
   - 只有当场景中提供了**与该要求矛盾**的信息时，才判定为不合规
   - 单纯未提及某项要求，不作为不合规的理由
   - 特别注意：如果场景描述了多项合规特征，未提及的方面默认视为满足
   - 若该要求是条件性的，或场景本身未涉及该领域 → 该项不影响判定

### 关键注意点
- 条款中的"或""等效""允许""替代"表示多种方案均可接受，满足任一即可
- 场景描述的基本事实视为准确，不要质疑场景设定本身
- 不要因为场景没提到某细节就假设"可能存在隐患"

## 输出格式

### 第一步：条款要求识别
列出与场景相关的条款要求（编号 + 原文引用或知识回忆）

### 第二步：逐项核对
对每条要求：
- 条款要求：[引用原文]
- 场景做法：[从场景中提取对应事实]
- 判断：✅ 满足 / ❌ 不满足 / ⚠️ 未提及
- 分析：[简要说明判断依据]

### 第三步：最终结论
- **合规** / **不合规**
- **依据**：简要总结判断逻辑"""
        return prompt

    def query(self, question: str) -> dict:
        prompt = self.build_prompt(question)
        print(f"\n📝 纯 LLM 查询: {question[:100]}...")
        answer = self.llm.judge(prompt)
        print(f"📋 回答: {answer[:200]}...")
        return {'prompt': prompt, 'answer': answer, 'retrieved_clauses': []}


class BM25Retriever:
    """BM25 检索器（基于 rank_bm25 + jieba）"""

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        # 加载条款
        with open(CLAUSES_JSON, 'r', encoding='utf-8') as f:
            self.clauses = json.load(f)

        # 构建 BM25 索引
        self.tokenized_corpus = []
        for c in self.clauses:
            text = f"{c['clause_id']} {c['title']} {c['content']}"
            tokens = list(jieba.cut(text))
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引就绪: {len(self.clauses)} 条")

    def search(self, query: str, top_k: int = 10) -> list:
        import jieba
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)

        # 取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            c = self.clauses[idx]
            results.append({
                'clause_id': c['clause_id'],
                'title': c['title'],
                'content': c['content'],
                'score': float(scores[idx]),
            })
        return results


class BM25LLM:
    """基线2：BM25 + LLM"""

    def __init__(self):
        self.llm = LLMClient()
        self.retriever = BM25Retriever()
        print("✅ BM25+LLM 基线就绪")

    def build_prompt(self, query: str, clauses: list) -> str:
        clause_context = ""
        for i, r in enumerate(clauses):
            clause_context += f"\n### 条款 {i+1}: {r['clause_id']}\n"
            clause_context += f"标题: {r['title']}\n"
            clause_context += f"内容: {r['content'][:800]}\n"

        prompt = f"""你是一个海事消防法规合规检查专家。请严格依据以下法规条款的原文内容，判断给定场景是否合规。

## 场景描述
{query}

## 相关法规条款（请逐条仔细阅读）
{clause_context}

## ⚠️ 核心判断原则（严格遵守）

你的任务是将场景中的**实际做法**与法规条款中的**具体要求**逐项核对。不要预设立场，也不要因为描述模糊就放过问题。

### 判断流程（必须严格执行）

**第一步：提取要求**
从每个条款中提取所有具体要求，逐条列出。每条要求包含：
- 条款来源（编号）
- 要求内容（引用原文）
- 要求类型：强制性要求 / 条件性要求

**第二步：逐项核对**
将场景中的实际做法与每条要求对比：
- ✅ 满足：场景中的做法明确符合该要求
- ❌ 不满足：场景中的做法明确违反该要求（必须引用原文指出具体哪项条件不达标）
- ⚠️ 未提及：场景描述中没有涉及此项要求

**第三步：综合判定**

判定规则：
1. 如果有任何一项要求被判定为"❌ 不满足"→ **不合规**
2. 如果所有要求都是"✅ 满足"→ **合规**
3. 如果存在"⚠️ 未提及"的要求：
   - 只有当场景中提供了**与该要求矛盾**的信息时，才判定为不合规
   - 单纯未提及某项要求，不作为不合规的理由
   - 特别注意：如果场景描述了多项合规特征，未提及的方面默认视为满足
   - 若该要求是条件性的，或场景本身未涉及该领域 → 该项不影响判定

### 关键注意点
- 条款中的"或""等效""允许""替代"表示多种方案均可接受，满足任一即可
- 场景描述的基本事实视为准确，不要质疑场景设定本身
- 不要因为场景没提到某细节就假设"可能存在隐患"
- 条款未明确规定的事项不能作为不合规的理由

## 输出格式

### 第一步：条款要求提取
列出从每个条款中提取的具体要求（编号 + 原文引用）

### 第二步：逐项核对
对每条要求：
- 条款要求：[引用原文]
- 场景做法：[从场景中提取对应事实]
- 判断：✅ 满足 / ❌ 不满足 / ⚠️ 未提及
- 分析：[简要说明判断依据]

### 第三步：最终结论
- **合规** / **不合规**
- **依据**：简要总结判断逻辑"""
        return prompt

    def query(self, question: str, top_k: int = 10) -> dict:
        clauses = self.retriever.search(question, top_k=top_k)
        print(f"\n📝 BM25 检索: {len(clauses)} 条")
        for i, c in enumerate(clauses[:5]):
            print(f"  {i+1}. {c['clause_id']}: {c['title'][:40]} (score={c['score']:.1f})")

        prompt = self.build_prompt(question, clauses)
        answer = self.llm.judge(prompt)
        print(f"📋 回答: {answer[:200]}...")
        return {'prompt': prompt, 'answer': answer, 'retrieved_clauses': clauses}


class NaiveRAG:
    """基线3：朴素 RAG（纯向量检索 + LLM）"""

    def __init__(self):
        from embed_clauses import VectorRetriever
        self.llm = LLMClient()
        self.retriever = VectorRetriever()
        print("✅ 朴素 RAG 基线就绪")

    def build_prompt(self, query: str, clauses: list) -> str:
        clause_context = ""
        for i, r in enumerate(clauses):
            clause_context += f"\n### 条款 {i+1}: {r['clause_id']}\n"
            clause_context += f"标题: {r['title']}\n"
            clause_context += f"内容: {r['content'][:800]}\n"

        prompt = f"""你是一个海事消防法规合规检查专家。请严格依据以下法规条款的原文内容，判断给定场景是否合规。

## 场景描述
{query}

## 相关法规条款（请逐条仔细阅读）
{clause_context}

## ⚠️ 核心判断原则（严格遵守）

你的任务是将场景中的**实际做法**与法规条款中的**具体要求**逐项核对。不要预设立场，也不要因为描述模糊就放过问题。

### 判断流程（必须严格执行）

**第一步：提取要求**
从每个条款中提取所有具体要求，逐条列出。每条要求包含：
- 条款来源（编号）
- 要求内容（引用原文）
- 要求类型：强制性要求 / 条件性要求

**第二步：逐项核对**
将场景中的实际做法与每条要求对比：
- ✅ 满足：场景中的做法明确符合该要求
- ❌ 不满足：场景中的做法明确违反该要求（必须引用原文指出具体哪项条件不达标）
- ⚠️ 未提及：场景描述中没有涉及此项要求

**第三步：综合判定**

判定规则：
1. 如果有任何一项要求被判定为"❌ 不满足"→ **不合规**
2. 如果所有要求都是"✅ 满足"→ **合规**
3. 如果存在"⚠️ 未提及"的要求：
   - 只有当场景中提供了**与该要求矛盾**的信息时，才判定为不合规
   - 单纯未提及某项要求，不作为不合规的理由
   - 特别注意：如果场景描述了多项合规特征，未提及的方面默认视为满足
   - 若该要求是条件性的，或场景本身未涉及该领域 → 该项不影响判定

### 关键注意点
- 条款中的"或""等效""允许""替代"表示多种方案均可接受，满足任一即可
- 场景描述的基本事实视为准确，不要质疑场景设定本身
- 不要因为场景没提到某细节就假设"可能存在隐患"
- 条款未明确规定的事项不能作为不合规的理由

## 输出格式

### 第一步：条款要求提取
列出从每个条款中提取的具体要求（编号 + 原文引用）

### 第二步：逐项核对
对每条要求：
- 条款要求：[引用原文]
- 场景做法：[从场景中提取对应事实]
- 判断：✅ 满足 / ❌ 不满足 / ⚠️ 未提及
- 分析：[简要说明判断依据]

### 第三步：最终结论
- **合规** / **不合规**
- **依据**：简要总结判断逻辑"""
        return prompt

    def query(self, question: str, top_k: int = 10) -> dict:
        vector_results = self.retriever.search(question, top_k=top_k, threshold=0.15)
        print(f"\n📝 向量检索: {len(vector_results)} 条")
        for i, c in enumerate(vector_results[:5]):
            print(f"  {i+1}. {c['clause_id']}: {c['title'][:40]} (score={c['score']:.2f})")

        # 统一格式
        clauses = [{'clause_id': r['clause_id'], 'title': r['title'],
                     'content': r['content'], 'score': r['score']} for r in vector_results]

        prompt = self.build_prompt(question, clauses)
        answer = self.llm.judge(prompt)
        print(f"📋 回答: {answer[:200]}...")
        return {'prompt': prompt, 'answer': answer, 'retrieved_clauses': clauses}


if __name__ == '__main__':
    test_q = "一艘2020年建造的散货船，机舱内配备了固定式CO2灭火系统。检查发现灭火剂的数量仅能释放相当于最大被保护处所总容积30%的自由气体。问：是否合规？"

    print("=" * 60)
    print("基线1：纯 LLM")
    print("=" * 60)
    s1 = PureLLM()
    r1 = s1.query(test_q)

    print("\n" + "=" * 60)
    print("基线2：BM25 + LLM")
    print("=" * 60)
    s2 = BM25LLM()
    r2 = s2.query(test_q, top_k=10)

    print("\n" + "=" * 60)
    print("基线3：朴素 RAG")
    print("=" * 60)
    s3 = NaiveRAG()
    r3 = s3.query(test_q, top_k=10)

    print("\n" + "=" * 60)
    print("📋 三个基线回答对比")
    print("=" * 60)
    for name, r in [("纯LLM", r1), ("BM25+LLM", r2), ("朴素RAG", r3)]:
        print(f"\n--- {name} ---")
        print(r['answer'][:300])
