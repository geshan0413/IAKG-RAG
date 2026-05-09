"""
条款向量化 + FAISS 索引构建
"""
import json
import os
import numpy as np
import faiss

# 配置
MODEL_PATH = '/root/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116'
CLAUSES_JSON = '/root/.openclaw/workspace/projects/kg-rag-solas/parsed_clauses.json'
INDEX_DIR = '/root/.openclaw/workspace/projects/kg-rag-solas/index'


def build_index():
    """构建条款向量索引"""
    from sentence_transformers import SentenceTransformer
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 加载模型
    print("加载向量模型...")
    model = SentenceTransformer(MODEL_PATH)

    # 加载条款
    with open(CLAUSES_JSON, 'r', encoding='utf-8') as f:
        clauses = json.load(f)
    print(f"加载 {len(clauses)} 个条款")

    # 构建待编码文本：clause_id + title + content
    texts = []
    ids = []
    for c in clauses:
        # 拼接 clause_id、标题、内容，增强语义
        text = f"[{c['clause_id']}] {c['title']}: {c['content']}"
        texts.append(text)
        ids.append(c['clause_id'])

    # 批量编码
    print("向量化中...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"向量矩阵: {embeddings.shape}")

    # 构建 FAISS 索引（内积 = 余弦相似度，因为已归一化）
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS 索引: {index.ntotal} 条")

    # 保存
    faiss.write_index(index, os.path.join(INDEX_DIR, 'clauses.faiss'))
    with open(os.path.join(INDEX_DIR, 'clause_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(ids, f, ensure_ascii=False)
    np.save(os.path.join(INDEX_DIR, 'embeddings.npy'), embeddings)

    # 保存条款元数据（用于检索后返回内容）
    meta = {c['clause_id']: {
        'title': c['title'],
        'content': c['content'],
        'source': c['source'],
        'article': c['article'],
    } for c in clauses}
    with open(os.path.join(INDEX_DIR, 'clause_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 索引已保存到 {INDEX_DIR}")
    print(f"  - clauses.faiss ({os.path.getsize(os.path.join(INDEX_DIR, 'clauses.faiss')) / 1024:.0f} KB)")
    print(f"  - clause_ids.json")
    print(f"  - clause_meta.json")


class VectorRetriever:
    """向量检索器（云端embedding API版）"""

    def __init__(self, index_dir=INDEX_DIR):
        self.index = faiss.read_index(os.path.join(index_dir, 'clauses.faiss'))
        with open(os.path.join(index_dir, 'clause_ids.json'), 'r') as f:
            self.clause_ids = json.load(f)
        with open(os.path.join(index_dir, 'clause_meta.json'), 'r') as f:
            self.clause_meta = json.load(f)

        # 使用本地BGE-large-zh-v1.5（1024维，离线可用）
        os.environ['HF_HUB_OFFLINE'] = '1'
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(
            '/root/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116'
        )
        self.use_local = True
        print(f"向量检索器就绪（本地BGE-large-zh）: {self.index.ntotal} 条索引")

    def _embed_query(self, query: str) -> np.ndarray:
        """获取query向量（本地模型）"""
        vec = self.embed_model.encode([query], normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def search(self, query: str, top_k: int = 10, threshold: float = 0.3):
        """语义检索"""
        q_emb = self._embed_query(query).reshape(1, -1)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue
            cid = self.clause_ids[idx]
            meta = self.clause_meta.get(cid, {})
            results.append({
                'clause_id': cid,
                'score': float(score),
                'title': meta.get('title', ''),
                'content': meta.get('content', ''),
                'source': meta.get('source', ''),
            })
        return results


if __name__ == '__main__':
    build_index()
