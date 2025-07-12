"""
相似度计算器基础版模块
实现余弦相似度计算功能，计算片段间的基础相似度
集成向量构建模块进行文本向量化
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import logger
from core.vector_builder import get_vector_builder, VectorBuilderError
from core.preprocessor import TextSegment
import numpy as np


class SimilarityCalculatorError(Exception):
    """相似度计算器异常"""
    pass


class SimilarityResult:
    """相似度结果类"""
    
    def __init__(self, segment1_id: str, segment2_id: str, similarity: float,
                 vector1: Optional[List[float]] = None, vector2: Optional[List[float]] = None):
        """
        初始化相似度结果
        
        Args:
            segment1_id: 第一个片段ID
            segment2_id: 第二个片段ID
            similarity: 相似度值（0-1之间）
            vector1: 第一个片段的向量（可选）
            vector2: 第二个片段的向量（可选）
        """
        self.segment1_id = segment1_id
        self.segment2_id = segment2_id
        self.similarity = similarity
        self.vector1 = vector1
        self.vector2 = vector2
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "segment1_id": self.segment1_id,
            "segment2_id": self.segment2_id,
            "similarity": self.similarity
        }
        
        # 可选包含向量信息
        if self.vector1 is not None:
            result["vector1_dim"] = len(self.vector1)
        if self.vector2 is not None:
            result["vector2_dim"] = len(self.vector2)
        
        return result
    
    def __repr__(self) -> str:
        return f"SimilarityResult({self.segment1_id} <-> {self.segment2_id}: {self.similarity:.4f})"


class SimilarityCalculator:
    """相似度计算器类"""
    
    def __init__(self, vector_model: Optional[str] = None, enable_vector_cache: bool = True):
        """
        初始化相似度计算器
        
        Args:
            vector_model: 向量模型名称
            enable_vector_cache: 是否启用向量缓存
        """
        self.vector_model = vector_model
        self.enable_vector_cache = enable_vector_cache
        
        # 获取向量构建器
        try:
            self.vector_builder = get_vector_builder(
                model=vector_model, 
                enable_cache=enable_vector_cache
            )
            logger.info(f"SimilarityCalculator initialized with model: {vector_model or 'default'}")
        except Exception as e:
            raise SimilarityCalculatorError(f"Failed to initialize vector builder: {e}")
    
    def calculate_similarity(self, segment1: TextSegment, segment2: TextSegment,
                           include_vectors: bool = False) -> SimilarityResult:
        """
        计算两个文本片段之间的相似度
        
        Args:
            segment1: 第一个文本片段
            segment2: 第二个文本片段
            include_vectors: 是否在结果中包含向量
            
        Returns:
            相似度结果
        """
        if not segment1.content or not segment2.content:
            logger.warning(f"Empty content in segments: {segment1.segment_id}, {segment2.segment_id}")
            return SimilarityResult(segment1.segment_id, segment2.segment_id, 0.0)
        
        logger.debug(f"Calculating similarity between {segment1.segment_id} and {segment2.segment_id}")
        
        try:
            # 获取向量
            vector1 = self.vector_builder.build_vector(segment1.content)
            vector2 = self.vector_builder.build_vector(segment2.content)
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(vector1, vector2)
            
            # 创建结果
            result_vectors = (vector1, vector2) if include_vectors else (None, None)
            result = SimilarityResult(
                segment1.segment_id, 
                segment2.segment_id, 
                similarity,
                result_vectors[0],
                result_vectors[1]
            )
            
            logger.debug(f"Similarity calculated: {similarity:.4f}")
            return result
            
        except VectorBuilderError as e:
            error_msg = f"Vector building failed: {e}"
            logger.error(error_msg)
            raise SimilarityCalculatorError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error calculating similarity: {e}"
            logger.error(error_msg)
            raise SimilarityCalculatorError(error_msg)
    
    def calculate_pairwise_similarities(self, segments: List[TextSegment],
                                      include_vectors: bool = False) -> List[SimilarityResult]:
        """
        计算片段列表中所有片段对的相似度
        
        Args:
            segments: 文本片段列表
            include_vectors: 是否在结果中包含向量
            
        Returns:
            相似度结果列表
        """
        if len(segments) < 2:
            logger.warning("Need at least 2 segments for pairwise similarity calculation")
            return []
        
        logger.info(f"Calculating pairwise similarities for {len(segments)} segments")
        
        try:
            # 批量生成向量，提高效率
            texts = [segment.content for segment in segments]
            vectors = self.vector_builder.build_vectors(texts)

            # 使用numpy优化相似度计算
            vectors_array = np.array(vectors)

            # 计算所有向量的模长
            norms = np.linalg.norm(vectors_array, axis=1)

            # 计算相似度矩阵（只计算上三角部分）
            results = []
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    # 使用numpy计算余弦相似度
                    if norms[i] > 0 and norms[j] > 0:
                        similarity = np.dot(vectors_array[i], vectors_array[j]) / (norms[i] * norms[j])
                        # 确保相似度在[-1, 1]范围内
                        similarity = np.clip(similarity, -1.0, 1.0)
                    else:
                        similarity = 0.0

                    result = SimilarityResult(
                        segments[i].segment_id,
                        segments[j].segment_id,
                        float(similarity),
                        vectors[i],
                        vectors[j])
                    results.append(result)

            logger.info(f"Calculated {len(results)} pairwise similarities using optimized numpy operations")
            return results
            
        except Exception as e:
            error_msg = f"Failed to calculate pairwise similarities: {e}"
            logger.error(error_msg)
            raise SimilarityCalculatorError(error_msg)
    
    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值（0-1之间）
            
        """
        if len(vector1) != len(vector2):
            raise SimilarityCalculatorError(
                f"Vector dimensions mismatch: {len(vector1)} vs {len(vector2)}"
            )
        
        if len(vector1) == 0:
            raise SimilarityCalculatorError("Empty vectors provided")
        
        try:
            # 计算点积
            dot_product = np.dot(vector1, vector2)
            
            # 计算向量模长（高效计算）
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            
            # 避免除零
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # 计算余弦相似度
            similarity = dot_product / (magnitude1 * magnitude2)
            
            # 余弦相似度的值域是[-1, 1]，但实际应用中我们只关心正相关，所以取绝对值
            similarity = abs(similarity)
            
            return similarity
            
        except Exception as e:
            raise SimilarityCalculatorError(f"Failed to calculate cosine similarity: {e}")
    
    def get_similarity_statistics(self, results: List[SimilarityResult]) -> Dict[str, Any]:
        """
        获取相似度结果的统计信息
        
        Args:
            results: 相似度结果列表
            
        Returns:
            统计信息字典
        """
        if not results:
            return {
                "total_pairs": 0,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "high_similarity_count": 0,
                "medium_similarity_count": 0,
                "low_similarity_count": 0
            }
        
        similarities = [r.similarity for r in results]
        avg_similarity = sum(similarities) / len(similarities)
        
        # 分类统计（高：>0.8，中：0.5-0.8，低：<0.5）
        high_count = sum(1 for s in similarities if s > 0.8)
        medium_count = sum(1 for s in similarities if 0.5 <= s <= 0.8)
        low_count = sum(1 for s in similarities if s < 0.5)
        
        return {
            "total_pairs": len(results),
            "avg_similarity": round(avg_similarity, 4),
            "min_similarity": round(min(similarities), 4),
            "max_similarity": round(max(similarities), 4),
            "high_similarity_count": high_count,
            "medium_similarity_count": medium_count,
            "low_similarity_count": low_count
        }
    

# 全局实例
similarity_calculator = None

def get_similarity_calculator(vector_model: Optional[str] = None, 
                            enable_vector_cache: bool = True) -> SimilarityCalculator:
    """
    获取相似度计算器实例
    
    Args:
        vector_model: 向量模型名称
        enable_vector_cache: 是否启用向量缓存
        
    Returns:
        相似度计算器实例
    """
    global similarity_calculator
    if similarity_calculator is None:
        similarity_calculator = SimilarityCalculator(
            vector_model=vector_model,
            enable_vector_cache=enable_vector_cache
        )
    return similarity_calculator
