"""
向量构建模块基础版
实现基础文本嵌入功能，将文本内容转换为向量表示，包含向量缓存机制
"""
import hashlib
from typing import List, Dict, Optional, Any, Union
from collections import OrderedDict
from utils.logger import logger
from core.model_client import get_openai_client, OpenAIClientError


class VectorBuilderError(Exception):
    """向量构建异常"""
    pass


class LRUVectorCache:
    """LRU缓存实现，用于向量缓存管理"""

    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存

        Args:
            max_size: 最大缓存条目数
        """
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[List[float]]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存的向量，如果不存在返回None
        """
        if key in self.cache:
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List[float]) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 向量值
        """
        if key in self.cache:
            # 更新现有值并移到末尾
            self.cache.move_to_end(key)
        else:
            # 检查是否需要删除最久未使用的项
            if len(self.cache) >= self.max_size:
                # 删除最久未使用的项（第一个）
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def size(self) -> int:
        """获取当前缓存大小"""
        return len(self.cache)


class VectorBuilder:
    """向量构建器类"""
    
    def __init__(self, model: Optional[str] = None, enable_cache: bool = True, cache_size: int = 1000):
        """
        初始化向量构建器

        Args:
            model: 嵌入模型名称，默认使用客户端配置
            enable_cache: 是否启用缓存
            cache_size: 缓存大小限制
        """
        self.model = model
        self.enable_cache = enable_cache

        # 使用LRU缓存替代无限制字典缓存
        self._vector_cache = LRUVectorCache(max_size=cache_size)

        # 缓存统计
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        
        # 获取OpenAI客户端
        try:
            self.client = get_openai_client()
            logger.info(f"VectorBuilder initialized with model: {model or 'default'}, cache: {enable_cache}")
        except Exception as e:
            raise VectorBuilderError(f"Failed to initialize OpenAI client: {e}")
    
    def build_vector(self, text: str) -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: 要转换的文本
            
        Returns:
            向量列表
        """
        text = text.strip()
        self._cache_stats["total_requests"] += 1
        
        # 检查缓存
        if self.enable_cache:
            cached_vector = self.get_cached_vector(text)
            if cached_vector is not None:
                self._cache_stats["hits"] += 1
                logger.debug(f"Cache hit for text (length: {len(text)})")
                return cached_vector
            else:
                self._cache_stats["misses"] += 1
        
        logger.info(f"Building vector for text (length: {len(text)})")
        
        try:
            # 调用OpenAI API生成向量
            vector = self.client.create_embedding(text, model=self.model)
            
            # 缓存向量
            if self.enable_cache:
                text_hash = self._get_text_hash(text)
                self._vector_cache.put(text_hash, vector)
                logger.debug(f"Cached vector for text hash: {text_hash[:8]}...")
            
            logger.info(f"Successfully built vector (dimension: {len(vector)})")
            return vector
            
        except OpenAIClientError as e:
            error_msg = f"Failed to build vector: {e}"
            logger.error(error_msg)
            raise VectorBuilderError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while building vector: {e}"
            logger.error(error_msg)
            raise VectorBuilderError(error_msg)
    
    def build_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        批量处理多个文本，转换为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表的列表
            
        Raises:
            VectorBuilderError: 转换失败时抛出
        """
        if not texts:
            raise VectorBuilderError("Text list cannot be empty")
        
        # 过滤空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise VectorBuilderError("No valid texts found")
        
        logger.info(f"Building vectors for {len(valid_texts)} texts")
        
        # 检查缓存，分离已缓存和未缓存的文本
        cached_vectors = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(valid_texts):
            self._cache_stats["total_requests"] += 1
            
            if self.enable_cache:
                cached_vector = self.get_cached_vector(text)
                if cached_vector is not None:
                    cached_vectors[i] = cached_vector
                    self._cache_stats["hits"] += 1
                    continue
            
            uncached_texts.append(text)
            uncached_indices.append(i)
            if self.enable_cache:
                self._cache_stats["misses"] += 1
        
        logger.info(f"Cache hits: {len(cached_vectors)}, Cache misses: {len(uncached_texts)}")
        
        # 批量处理未缓存的文本
        new_vectors = []
        if uncached_texts:
            try:
                new_vectors = self.client.create_embeddings(uncached_texts, model=self.model)
                
                # 缓存新向量
                if self.enable_cache:
                    for text, vector in zip(uncached_texts, new_vectors):
                        text_hash = self._get_text_hash(text)
                        self._vector_cache.put(text_hash, vector)
                
                logger.info(f"Successfully built {len(new_vectors)} new vectors")
                
            except OpenAIClientError as e:
                error_msg = f"Failed to build vectors: {e}"
                logger.error(error_msg)
                raise VectorBuilderError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error while building vectors: {e}"
                logger.error(error_msg)
                raise VectorBuilderError(error_msg)
        
        # 合并缓存和新生成的向量
        result_vectors: List[List[float]] = [[] for _ in range(len(valid_texts))]

        # 填入缓存的向量
        for i, vector in cached_vectors.items():
            result_vectors[i] = vector

        # 填入新生成的向量
        for i, vector in zip(uncached_indices, new_vectors):
            result_vectors[i] = vector
        
        logger.info(f"Successfully built vectors for {len(valid_texts)} texts")
        return result_vectors
    
    def get_cached_vector(self, text: str) -> Optional[List[float]]:
        """
        从缓存获取向量
        
        Args:
            text: 文本内容
            
        Returns:
            缓存的向量，如果不存在返回None
        """
        if not self.enable_cache or not text:
            return None
        
        text_hash = self._get_text_hash(text.strip())
        return self._vector_cache.get(text_hash)
    
    def clear_cache(self) -> int:
        """
        清理缓存
        
        Returns:
            清理的缓存项数量
        """
        cache_size = self._vector_cache.size()
        self._vector_cache.clear()
        
        # 重置统计
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        
        logger.info(f"Cleared {cache_size} cached vectors")
        return cache_size
    
    def _get_text_hash(self, text: str) -> str:
        """
        获取文本的哈希值作为缓存键
        
        Args:
            text: 文本内容
            
        Returns:
            文本的SHA256哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


# 全局实例
vector_builder = None

def get_vector_builder(model: Optional[str] = None, enable_cache: bool = True, cache_size: int = 1000) -> VectorBuilder:
    """
    获取向量构建器实例

    Args:
        model: 嵌入模型名称
        enable_cache: 是否启用缓存
        cache_size: 缓存大小限制

    Returns:
        向量构建器实例
    """
    global vector_builder
    if vector_builder is None:
        vector_builder = VectorBuilder(model=model, enable_cache=enable_cache, cache_size=cache_size)
    return vector_builder
