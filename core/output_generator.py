"""
输出生成器模块
生成结构化输出，包含片段ID、内容、位置信息的JSON格式结果
提供质量报告和元数据生成功能
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.logger import logger
from core.segment_optimizer import SegmentGroup
from core.preprocessor import TextSegment


class OutputError(Exception):
    pass


class Output:
    """输出生成器类"""
    
    def generate_quality_report(self, groups: List[SegmentGroup]) -> Dict[str, Any]:
        """
        生成质量报告
        
        Args:
            groups: 片段组列表
            
        Returns:
            质量报告字典
        """
        if not groups:
            return {"status": "no_data", "score": 0.0}
        
        # 计算各种质量指标
        segment_lengths = []
        for group in groups:
            for segment in group.segments:
                segment_lengths.append(len(segment.content))
        
        # 长度分布分析
        if segment_lengths:
            min_length = min(segment_lengths)
            max_length = max(segment_lengths)
            avg_length = sum(segment_lengths) / len(segment_lengths)
            
            # 计算长度一致性得分
            variance = sum((length - avg_length) ** 2 for length in segment_lengths) / len(segment_lengths)
            std_dev = variance ** 0.5
            consistency_score = max(0, 1 - (std_dev / avg_length)) if avg_length > 0 else 0
        else:
            min_length = max_length = avg_length = consistency_score = 0
        
        # 组大小分析
        group_sizes = [len(group.segments) for group in groups]
        avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
        
        # 计算总体质量得分
        quality_score = self.calculate_quality_score(
            consistency_score, avg_length, avg_group_size, len(groups)
        )
        
        return {
            "status": "analyzed",
            "overall_score": round(quality_score, 3),
            "metrics": {
                "segment_count": len(segment_lengths),
                "group_count": len(groups),
                "length_stats": {
                    "min": min_length,
                    "max": max_length,
                    "avg": round(avg_length, 2),
                    "consistency_score": round(consistency_score, 3)
                },
                "group_stats": {
                    "avg_size": round(avg_group_size, 2),
                    "size_distribution": self.get_size_distribution(group_sizes)
                }
            }
        }
    
    def calculate_quality_score(self, consistency_score: float, avg_length: float,
                               avg_group_size: float, group_count: int) -> float:
        """
        计算总体质量得分
        
        Args:
            consistency_score: 一致性得分
            avg_length: 平均长度
            avg_group_size: 平均组大小
            group_count: 组数量
            
        Returns:
            质量得分 (0-1)
        """
        # 长度合理性得分（理想长度范围：50-500字符）
        length_score = 1.0
        if avg_length < 50:
            length_score = avg_length / 50
        elif avg_length > 500:
            length_score = max(0.5, 1 - (avg_length - 500) / 1000)
        
        # 组大小合理性得分（理想组大小：1-5个片段）
        size_score = 1.0
        if avg_group_size > 5:
            size_score = max(0.5, 1 - (avg_group_size - 5) / 10)
        
        # 组数量合理性得分（避免过度分割或分割不足）
        count_score = 1.0
        if group_count < 2:
            count_score = 0.5  # 分割不足
        elif group_count > 20:
            count_score = max(0.7, 1 - (group_count - 20) / 50)  # 过度分割
        
        # 加权平均
        weights = [0.3, 0.25, 0.25, 0.2]  # 一致性、长度、大小、数量
        scores = [consistency_score, length_score, size_score, count_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def get_size_distribution(self, sizes: List[int]) -> Dict[str, int]:
        """
        获取大小分布统计
        
        Args:
            sizes: 大小列表
            
        Returns:
            分布统计字典
        """
        distribution = {"1": 0, "2-3": 0, "4-5": 0, "6+": 0}
        
        for size in sizes:
            if size == 1:
                distribution["1"] += 1
            elif 2 <= size <= 3:
                distribution["2-3"] += 1
            elif 4 <= size <= 5:
                distribution["4-5"] += 1
            else:
                distribution["6+"] += 1
        
        return distribution
    
    def save_output_to_file(self, output: Dict[str, Any], filepath: str) -> None:
        """
        保存输出到文件
        
        Args:
            output: 输出字典
            filepath: 文件路径
            
        Raises:
            OutputGeneratorError: 保存失败时抛出
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Output saved to file: {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to save output to file {filepath}: {e}"
            logger.error(error_msg)
            raise OutputError(error_msg)


# 全局实例
output_generator = None

def get_output_generator() -> Output:
    global output_generator
    if output_generator is None:
        output_generator = Output()
    return output_generator
