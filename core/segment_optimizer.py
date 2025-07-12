"""
片段优化器基础版模块
实现基础优化功能：碎片合并和超长片段拆分
优化分割结果的质量，与分割决策器模块集成
"""
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from utils.logger import logger
from core.preprocessor import TextSegment
from core.segment_decider import SegmentGroup, get_segment_decider


class SegmentOptimizerError(Exception):
    """片段优化器异常"""
    pass


class SegmentOptimizer:
    """片段优化器类"""
    
    def __init__(self, min_segment_length: int = 10, max_segment_length: int = 1000,
                 similarity_threshold: float = 0.7, vector_model: Optional[str] = None):
        """
        初始化片段优化器
        
        Args:
            min_segment_length: 最小片段长度，小于此长度的片段将被合并
            max_segment_length: 最大片段长度，大于此长度的片段将被拆分
            similarity_threshold: 相似度阈值，用于决定片段合并
            vector_model: 向量模型名称
        """
        if min_segment_length <= 0:
            raise SegmentOptimizerError(f"Min segment length must be positive, got {min_segment_length}")
        if max_segment_length <= min_segment_length:
            raise SegmentOptimizerError(f"Max segment length must be greater than min segment length, "
                                      f"got {max_segment_length} <= {min_segment_length}")
        
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.similarity_threshold = similarity_threshold
        self.vector_model = vector_model
        
        # 获取分割决策器
        try:
            self.segment_decider = get_segment_decider(
                similarity_threshold=similarity_threshold,
                vector_model=vector_model
            )
            logger.info(f"SegmentOptimizer initialized with min_length={min_segment_length}, "
                       f"max_length={max_segment_length}")
        except Exception as e:
            raise SegmentOptimizerError(f"Failed to initialize segment decider: {e}")
    
    def optimize_segments(self, groups: List[SegmentGroup]) -> List[SegmentGroup]:
        """
        优化片段组
        
        Args:
            groups: 片段组列表
            
        Returns:
            优化后的片段组列表
            
        Raises:
            SegmentOptimizerError: 优化失败时抛出
        """
        if not groups:
            logger.warning("Empty groups list provided")
            return []
        
        logger.info(f"Optimizing {len(groups)} segment groups")
        
        try:
            # 合并小片段
            merged_groups = self.merge_small_segments(groups)
            logger.info(f"After merging small segments: {len(merged_groups)} groups")
            
            # 拆分大片段
            optimized_groups = self.split_large_segments(merged_groups)
            logger.info(f"After splitting large segments: {len(optimized_groups)} groups")
            
            return optimized_groups
            
        except Exception as e:
            error_msg = f"Failed to optimize segments: {e}"
            logger.error(error_msg)
            raise SegmentOptimizerError(error_msg)
    
    def merge_small_segments(self, groups: List[SegmentGroup]) -> List[SegmentGroup]:
        """
        合并小片段
        
        Args:
            groups: 片段组列表
            
        Returns:
            合并后的片段组列表
        """
        if not groups:
            return []
        
        logger.info(f"Merging small segments in {len(groups)} groups")
        result_groups = []
        
        for group in groups:
            # 如果组内只有一个片段，检查是否需要与其他组合并
            if len(group.segments) == 1:
                segment = group.segments[0]
                
                # 如果片段内容长度小于最小长度，标记为需要合并
                if len(segment.content) < self.min_segment_length:
                    logger.debug(f"Small segment found: {segment.segment_id}, length={len(segment.content)}")
                    # 尝试找到最相似的组进行合并
                    merged = self._try_merge_with_most_similar(segment, result_groups)
                    if not merged:
                        # 如果无法合并，仍然保留为单独的组
                        result_groups.append(group)
                else:
                    # 长度足够，保留原组
                    result_groups.append(group)
            else:
                # 组内有多个片段，贪心算法进行合并
                optimized_segments = self._merge_small_segments_within_group(group.segments)
                new_group = SegmentGroup(group.group_id, optimized_segments)
                result_groups.append(new_group)
        return result_groups
    
    def _try_merge_with_most_similar(self, segment: TextSegment, 
                                   groups: List[SegmentGroup]) -> bool:
        """
        尝试将片段与最相似的组合并
        
        Args:
            segment: 要合并的片段
            groups: 现有片段组列表
            
        Returns:
            是否成功合并
        """
        if not groups:
            return False
        
        # 计算与每个组的相似度
        max_similarity = -1
        most_similar_group_idx = -1
        
        for i, group in enumerate(groups):
            # 计算与组内所有片段的平均相似度
            similarities = []
            for other_segment in group.segments:
                try:
                    result = self.segment_decider.similarity_calculator.calculate_similarity(
                        segment, other_segment
                    )
                    similarities.append(result.similarity)
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity: {e}")
                    continue
            
            # 计算平均相似度
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    most_similar_group_idx = i
        
        # 如果找到相似度高于阈值的组，进行合并
        if max_similarity >= self.similarity_threshold and most_similar_group_idx >= 0:
            logger.debug(f"Merging segment {segment.segment_id} with group {groups[most_similar_group_idx].group_id}, "
                        f"similarity={max_similarity:.4f}")
            groups[most_similar_group_idx].segments.append(segment)
            groups[most_similar_group_idx].segment_ids.append(segment.segment_id)
            return True
        
        return False
    
    def _merge_small_segments_within_group(self, segments: List[TextSegment]) -> List[TextSegment]:
        """
        合并组内的小片段
        
        Args:
            segments: 片段列表
            
        Returns:
            合并后的片段列表
        """
        if len(segments) <= 1:
            return segments
        
        # 按位置排序片段（先按行，再按列）
        sorted_segments = sorted(segments, key=lambda s: (s.start_pos[0], s.start_pos[1]))
        
        result_segments = []
        current_segment = sorted_segments[0]
        
        for i in range(1, len(sorted_segments)):
            next_segment = sorted_segments[i]
            
            # 如果当前片段很小，尝试与下一个合并
            if len(current_segment.content) < self.min_segment_length:
                # 合并片段
                merged_segment = self._merge_two_segments(current_segment, next_segment)
                current_segment = merged_segment
            else:
                # 当前片段足够大，添加到结果中
                result_segments.append(current_segment)
                current_segment = next_segment
        
        # 添加最后一个处理的片段
        result_segments.append(current_segment)
        
        return result_segments
    
    def _merge_two_segments(self, segment1: TextSegment, segment2: TextSegment) -> TextSegment:
        """
        合并两个片段
        
        Args:
            segment1: 第一个片段
            segment2: 第二个片段
            
        Returns:
            合并后的片段
        """
        # 合并内容
        merged_content = f"{segment1.content} {segment2.content}"
        
        # 合并单元格
        merged_cells = segment1.cells + segment2.cells
        
        # 计算新的位置范围
        start_row = min(segment1.start_pos[0], segment2.start_pos[0])
        start_col = min(segment1.start_pos[1], segment2.start_pos[1])
        end_row = max(segment1.end_pos[0], segment2.end_pos[0])
        end_col = max(segment1.end_pos[1], segment2.end_pos[1])
        
        # 创建新的片段ID
        merged_id = f"merged_{segment1.segment_id}_{segment2.segment_id}"
        
        # 确定片段类型（保留第一个片段的类型）
        segment_type = segment1.segment_type
        
        # 创建合并后的片段
        merged_segment = TextSegment(
            segment_id=merged_id,
            content=merged_content,
            segment_type=segment_type,
            cells=merged_cells,
            start_pos=(start_row, start_col),
            end_pos=(end_row, end_col)
        )
        
        logger.debug(f"Merged segments {segment1.segment_id} and {segment2.segment_id} into {merged_id}")
        return merged_segment
    
    def split_large_segments(self, groups: List[SegmentGroup]) -> List[SegmentGroup]:
        """
        拆分大片段
        
        Args:
            groups: 片段组列表
            
        Returns:
            拆分后的片段组列表
        """
        if not groups:
            return []
        
        logger.info(f"Splitting large segments in {len(groups)} groups")
        result_groups = []
        
        for group_idx, group in enumerate(groups):
            large_segments = []
            normal_segments = []
            
            # 分离大片段和正常片段
            for segment in group.segments:
                if len(segment.content) > self.max_segment_length:
                    large_segments.append(segment)
                else:
                    normal_segments.append(segment)
            
            # 如果没有大片段，保留原组
            if not large_segments:
                result_groups.append(group)
                continue
            
            # 处理大片段
            logger.debug(f"Found {len(large_segments)} large segments in group {group.group_id}")
            
            # 创建包含正常片段的组
            if normal_segments:
                normal_group = SegmentGroup(group.group_id, normal_segments)
                result_groups.append(normal_group)
            
            # 拆分每个大片段
            for i, large_segment in enumerate(large_segments):
                split_segments = self._split_segment(large_segment)
                
                # 如果成功拆分，为每个拆分片段创建新组
                if len(split_segments) > 1:
                    for j, split_segment in enumerate(split_segments):
                        split_group_id = f"{group.group_id}_split_{i}_{j}"
                        split_group = SegmentGroup(split_group_id, [split_segment])
                        result_groups.append(split_group)
                else:
                    # 无法拆分，保留原片段
                    large_group_id = f"{group.group_id}_large_{i}"
                    large_group = SegmentGroup(large_group_id, [large_segment])
                    result_groups.append(large_group)
        
        return result_groups
    
    def _split_segment(self, segment: TextSegment) -> List[TextSegment]:
        """
        拆分大片段
        
        Args:
            segment: 要拆分的片段
            
        Returns:
            拆分后的片段列表
        """
        content = segment.content
        
        logger.debug(f"Splitting large segment {segment.segment_id}, length={len(content)}")
        
        # 尝试在句子边界拆分
        split_segments = self._split_at_sentence_boundaries(segment)
        
        # 如果无法在句子边界拆分，尝试在段落边界拆分
        if len(split_segments) <= 1:
            split_segments = self._split_at_paragraph_boundaries(segment)
        
        # 如果仍然无法拆分，强制按长度拆分
        if len(split_segments) <= 1:
            split_segments = self._force_split_by_length(segment)
        
        logger.debug(f"Split segment {segment.segment_id} into {len(split_segments)} parts")
        return split_segments
    
    def _split_at_sentence_boundaries(self, segment: TextSegment) -> List[TextSegment]:
        """
        在句子边界拆分片段
        
        Args:
            segment: 要拆分的片段
            
        Returns:
            拆分后的片段列表
        """
        content = segment.content
        
        # 使用正则表达式匹配句子边界
        sentence_pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(sentence_pattern, content)
        
        return self._create_split_segments(segment, sentences, "sentence")
    
    def _split_at_paragraph_boundaries(self, segment: TextSegment) -> List[TextSegment]:
        """
        在段落边界拆分片段
        
        Args:
            segment: 要拆分的片段
            
        Returns:
            拆分后的片段列表
        """
        content = segment.content
        
        # 使用换行符拆分段落
        paragraphs = re.split(r'\n+', content)
        
        return self._create_split_segments(segment, paragraphs, "paragraph")
    
    def _force_split_by_length(self, segment: TextSegment) -> List[TextSegment]:
        """
        强制按长度拆分片段

        Args:
            segment: 要拆分的片段

        Returns:
            拆分后的片段列表
        """
        content = segment.content

        # 计算需要拆分的部分数
        parts_count = (len(content) + self.max_segment_length - 1) // self.max_segment_length

        # 拆分内容
        parts = []
        for i in range(parts_count):
            start = i * self.max_segment_length
            end = min((i + 1) * self.max_segment_length, len(content))
            # 确保不会超出最大长度
            part = content[start:end]
            if len(part) > self.max_segment_length:
                part = part[:self.max_segment_length]
            parts.append(part)

        return self._create_split_segments(segment, parts, "length")
    
    def _create_split_segments(self, original_segment: TextSegment, 
                             parts: List[str], split_type: str) -> List[TextSegment]:
        """
        根据拆分的部分创建新片段
        
        Args:
            original_segment: 原始片段
            parts: 拆分的内容部分
            split_type: 拆分类型
            
        Returns:
            拆分后的片段列表
        """
        # 如果只有一个部分，不需要拆分
        if len(parts) <= 1:
            return [original_segment]
        
        # 合并短部分
        merged_parts = []
        current_part = ""

        for part in parts:
            # 如果当前部分为空，跳过
            if not part:
                continue

            # 如果当前部分很短，尝试与累积部分合并
            if len(part) < self.min_segment_length:
                # 检查合并后是否超过最大长度
                combined_length = len(current_part) + (1 if current_part else 0) + len(part)
                if combined_length <= self.max_segment_length:
                    current_part += " " + part if current_part else part
                    continue
                else:
                    # 无法合并，先保存当前累积部分
                    if current_part:
                        merged_parts.append(current_part)
                    current_part = part
                    continue

            # 如果累积部分加上当前部分不超过最大长度，合并它们
            combined_length = len(current_part) + (1 if current_part else 0) + len(part)
            if combined_length <= self.max_segment_length:
                current_part += " " + part if current_part else part
            else:
                # 如果累积部分不为空，添加到结果
                if current_part:
                    merged_parts.append(current_part)

                # 开始新的累积部分
                current_part = part

        # 添加最后一个累积部分
        if current_part:
            merged_parts.append(current_part)
        
        # 如果合并后只有一个部分，不需要拆分
        if len(merged_parts) <= 1:
            return [original_segment]
        
        # 创建新片段
        result_segments = []
        for i, part in enumerate(merged_parts):
            # 创建新的片段ID
            new_id = f"{original_segment.segment_id}_split_{split_type}_{i}"
            
            # 创建新片段
            new_segment = TextSegment(
                segment_id=new_id,
                content=part,
                segment_type=original_segment.segment_type,
                cells=original_segment.cells,  # 保留原始单元格引用
                start_pos=original_segment.start_pos,
                end_pos=original_segment.end_pos
            )
            
            result_segments.append(new_segment)
        
        return result_segments


# 全局实例
segment_optimizer = None

def get_segment_optimizer(min_segment_length: int = 10, max_segment_length: int = 1000,
                         similarity_threshold: float = 0.7,
                         vector_model: Optional[str] = None) -> SegmentOptimizer:
    """
    获取片段优化器实例
    
    Args:
        min_segment_length: 最小片段长度
        max_segment_length: 最大片段长度
        similarity_threshold: 相似度阈值
        vector_model: 向量模型名称
        
    Returns:
        片段优化器实例
    """
    global segment_optimizer
    if segment_optimizer is None:
        segment_optimizer = SegmentOptimizer(
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            similarity_threshold=similarity_threshold,
            vector_model=vector_model
        )
    return segment_optimizer
