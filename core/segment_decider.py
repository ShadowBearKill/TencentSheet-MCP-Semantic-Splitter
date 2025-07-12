"""
分割决策器基础版模块
实现固定阈值分割决策功能，基于相似度阈值进行分割决策
与相似度计算器模块集成
"""
from typing import List, Dict, Any, Optional, Tuple, Set
from utils.logger import logger
from core.preprocessor import TextSegment
from core.similarity_calculator import get_similarity, SimilarityCalculatorError, SimilarityResult


class SegmentDeciderError(Exception):
    pass


class SegmentGroup:
    """片段组类"""
    
    def __init__(self, group_id: str, segments: List[TextSegment]):
        """
        初始化片段组
        
        Args:
            group_id: 组唯一标识
            segments: 包含的片段列表
        """
        self.group_id = group_id
        self.segments = segments
        self.segment_ids = [seg.segment_id for seg in segments]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "group_id": self.group_id,
            "segment_count": len(self.segments),
            "segment_ids": self.segment_ids,
            "segments": [seg.to_dict() for seg in self.segments]
        }
    
    def __repr__(self) -> str:
        return f"SegmentGroup(id={self.group_id}, segments={len(self.segments)})"


class SegmentDecider:
    """分割决策器类"""
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 vector_model: Optional[str] = None,
                 enable_vector_cache: bool = True):
        """
        初始化分割决策器
        
        Args:
            similarity_threshold: 相似度阈值，低于此值的片段将被分割
            vector_model: 向量模型名称
            enable_vector_cache: 是否启用向量缓存
        """
        if not 0 <= similarity_threshold <= 1:
            raise SegmentDeciderError(f"Similarity threshold must be between 0 and 1, got {similarity_threshold}")
        
        self.similarity_threshold = similarity_threshold
        self.vector_model = vector_model
        self.enable_vector_cache = enable_vector_cache
        
        # 获取相似度计算器
        try:
            self.similarity_calculator = get_similarity(
                vector_model=vector_model,
                enable_vector_cache=enable_vector_cache
            )
            logger.info(f"SegmentDecider initialized with threshold: {similarity_threshold}")
        except Exception as e:
            raise SegmentDeciderError(f"Failed to initialize similarity calculator: {e}")
    
    def decide_segments(self, segments: List[TextSegment]) -> List[SegmentGroup]:
        """
        基于相似度矩阵进行分割决策
        
        Args:
            segments: 文本片段列表
            
        Returns:
            分割后的片段组列表
        """
        if not segments:
            logger.warning("Empty segments list provided")
            return []
        
        if len(segments) == 1:
            logger.info("Only one segment, creating single group")
            return [SegmentGroup("group_0", segments)]
        
        logger.info(f"Making segment decisions for {len(segments)} segments with threshold {self.similarity_threshold}")
        
        try:
            # 计算相似度矩阵
            similarity_results = self.calculate_similarity_matrix(segments)
            
            # 基于阈值进行分割决策，分割成一个个连通分量。
            groups = self.apply_threshold_clustering(segments, similarity_results)
            
            logger.info(f"Segment decision completed: {len(groups)} groups created")
            return groups
            
        except Exception as e:
            error_msg = f"Failed to make segment decisions: {e}"
            logger.error(error_msg)
            raise SegmentDeciderError(error_msg)
    
    def apply_threshold(self, segments: List[TextSegment], 
                       custom_threshold: Optional[float] = None) -> List[SegmentGroup]:
        """
        应用固定阈值进行分割
        
        Args:
            segments: 文本片段列表
            custom_threshold: 自定义阈值，如果不提供则使用默认阈值
            
        Returns:
            分割后的片段组列表
        """
        original_threshold = self.similarity_threshold
        
        if custom_threshold is not None:
            if not 0 <= custom_threshold <= 1:
                raise SegmentDeciderError(f"Custom threshold must be between 0 and 1, got {custom_threshold}")
            self.similarity_threshold = custom_threshold
            logger.info(f"Using custom threshold: {custom_threshold}")
        
        try:
            result = self.decide_segments(segments)
            return result
        finally:
            # 恢复原始阈值
            self.similarity_threshold = original_threshold
    
    def calculate_similarity_matrix(self, segments: List[TextSegment]) -> List[SimilarityResult]:
        """
        计算片段间的相似度矩阵
        
        Args:
            segments: 文本片段列表
            
        Returns:
            相似度结果列表
        """
        logger.debug(f"Calculating similarity matrix for {len(segments)} segments")
        
        try:
            similarity_results = self.similarity_calculator.calculate_pairwise_similarities(segments)
            logger.debug(f"Calculated {len(similarity_results)} pairwise similarities")
            return similarity_results
            
        except SimilarityCalculatorError as e:
            raise SegmentDeciderError(f"Similarity calculation failed: {e}")
    
    def apply_threshold_clustering(self, segments: List[TextSegment], 
                                  similarity_results: List[SimilarityResult]) -> List[SegmentGroup]:
        """
        基于阈值进行聚类分组
        
        Args:
            segments: 文本片段列表
            similarity_results: 相似度结果列表
            
        Returns:
            片段组列表
        """
        # 创建片段ID到索引的映射
        segment_id_to_index = {seg.segment_id: i for i, seg in enumerate(segments)}
        
        # 构建邻接表：相似度高于阈值的片段对
        adjacency = {i: set() for i in range(len(segments))}
        
        for result in similarity_results:
            if result.similarity >= self.similarity_threshold:
                idx1 = segment_id_to_index[result.segment1_id]
                idx2 = segment_id_to_index[result.segment2_id]
                adjacency[idx1].add(idx2)
                adjacency[idx2].add(idx1)
        
        # 使用邻接表查找连通分量
        groups = self.find_connected_components(segments, adjacency)
        
        logger.debug(f"Threshold clustering created {len(groups)} groups")
        return groups
    
    def find_connected_components(self, segments: List[TextSegment], 
                                 adjacency: Dict[int, Set[int]]) -> List[SegmentGroup]:
        """
        查找连通分量（连通子图）
        
        Args:
            segments: 文本片段列表
            adjacency: 邻接表
            
        Returns:
            片段组列表
        """
        visited = set()
        groups = []
        
        for i in range(len(segments)):
            if i not in visited:
                # DFS查找连通分量
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        # 添加所有未访问的邻居
                        for neighbor in adjacency[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                # 创建片段组
                group_segments = [segments[idx] for idx in component]
                group_id = f"group_{len(groups)}"
                groups.append(SegmentGroup(group_id, group_segments))
        return groups
    
    def get_decision_statistics(self, groups: List[SegmentGroup]) -> Dict[str, Any]:
        """
        获取分割决策统计信息
        
        Args:
            groups: 片段组列表
            
        Returns:
            统计信息字典
        """
        if not groups:
            return {
                "total_groups": 0,
                "total_segments": 0,
                "avg_segments_per_group": 0.0,
                "min_group_size": 0,
                "max_group_size": 0,
                "single_segment_groups": 0
            }
        
        group_sizes = [len(group.segments) for group in groups]
        total_segments = sum(group_sizes)
        avg_size = total_segments / len(groups)
        single_segment_groups = sum(1 for size in group_sizes if size == 1)
        
        return {
            "total_groups": len(groups),
            "total_segments": total_segments,
            "avg_segments_per_group": round(avg_size, 2),
            "min_group_size": min(group_sizes),
            "max_group_size": max(group_sizes),
            "single_segment_groups": single_segment_groups
        }
    
    def analyze_threshold(self, segments: List[TextSegment], 
                               thresholds: List[float]) -> Dict[float, Dict[str, Any]]:
        """
        分析不同阈值对分割结果的影响
        
        Args:
            segments: 文本片段列表
            thresholds: 要测试的阈值列表
            
        Returns:
            阈值影响分析结果
        """
        if not segments:
            return {}
        
        logger.info(f"Analyzing threshold impact for {len(thresholds)} thresholds")
        
        results = {}
        original_threshold = self.similarity_threshold
        
        try:
            for threshold in thresholds:
                if not 0 <= threshold <= 1:
                    logger.warning(f"Skipping invalid threshold: {threshold}")
                    continue
                
                groups = self.apply_threshold(segments, threshold)
                stats = self.get_decision_statistics(groups)
                results[threshold] = stats
            
            logger.info(f"Threshold impact analysis completed for {len(results)} valid thresholds")
            return results
            
        finally:
            self.similarity_threshold = original_threshold
    

# 全局实例
segment_decider = None

def get_segment_decider(similarity_threshold: float = 0.7,
                       vector_model: Optional[str] = None,
                       enable_vector_cache: bool = True) -> SegmentDecider:
    global segment_decider
    if segment_decider is None:
        segment_decider = SegmentDecider(
            similarity_threshold=similarity_threshold,
            vector_model=vector_model,
            enable_vector_cache=enable_vector_cache
        )
    return segment_decider
