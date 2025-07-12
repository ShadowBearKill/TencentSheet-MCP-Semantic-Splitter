"""
预处理器基础版模块
实现基础切分功能：按行、列、单元格进行初步切分
处理来自API响应解析器的标准化数据，生成初步的文本片段用于后续语义分析
"""
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import logger
from api.response_parser import CellData


class PreprocessorError(Exception):
    """预处理器异常"""
    pass


class TextSegment:
    """文本片段类"""
    
    def __init__(self, segment_id: str, content: str, segment_type: str,
                 cells: List[CellData], start_pos: Tuple[int, int], 
                 end_pos: Tuple[int, int]):
        """
        初始化文本片段
        
        Args:
            segment_id: 片段唯一标识
            content: 片段文本内容
            segment_type: 片段类型（row/column/cell）
            cells: 包含的单元格数据列表
            start_pos: 起始位置 (row, col)
            end_pos: 结束位置 (row, col)
        """
        self.segment_id = segment_id
        self.content = content
        self.segment_type = segment_type
        self.cells = cells
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "segment_id": self.segment_id,
            "content": self.content,
            "segment_type": self.segment_type,
            "cell_count": len(self.cells),
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "cells": [cell.to_dict() for cell in self.cells]
        }
    
    def __repr__(self) -> str:
        return f"TextSegment(id={self.segment_id}, type={self.segment_type}, content_length={len(self.content)})"


class Preprocessor:
    """预处理器类"""
    
    def __init__(self, min_content_length: int = 1):
        """
        初始化预处理器
        
        Args:
            min_content_length: 最小内容长度，过滤空或过短的片段
        """
        self.min_content_length = min_content_length
        logger.info(f"Preprocessor initialized with min_content_length: {min_content_length}")
    
    def preprocess(self, cell_data_list: List[CellData], 
                  segment_types: int = 1) -> List[TextSegment]:
        """
        预处理单元格数据，生成文本片段
        
        Args:
            cell_data_list: 单元格数据列表
            segment_types: 要生成的片段类型列表，默认为 ['column']
            
        Returns:
            文本片段列表
            
        Raises:
            PreprocessorError: 预处理失败时抛出
        """
        if not cell_data_list:
            logger.warning("Empty cell data list provided")
            return []    
        logger.info(f"Preprocessing {len(cell_data_list)} cells with segment types: {segment_types}")
        
        try:
            segments = []
            
            # 按行切分
            if segment_types == 2:
                row_segments = self._segment_by_rows(cell_data_list)
                segments.extend(row_segments)
                logger.debug(f"Generated {len(row_segments)} row segments")
            
            # 按列切分
            elif segment_types == 3:
                column_segments = self._segment_by_columns(cell_data_list)
                segments.extend(column_segments)
                logger.debug(f"Generated {len(column_segments)} column segments")
            
            # 按单元格切分
            elif segment_types == 1:
                cell_segments = self._segment_by_cells(cell_data_list)
                segments.extend(cell_segments)
                logger.debug(f"Generated {len(cell_segments)} cell segments")
            
            else:
                raise PreprocessorError(f"Invalid segment types: {segment_types}")
            
            # 过滤空或过短的片段
            valid_segments = self._filter_segments(segments, segment_types)
            
            logger.info(f"Preprocessing completed: {len(valid_segments)} valid segments from {len(segments)} total")
            return valid_segments
            
        except Exception as e:
            error_msg = f"Failed to preprocess cell data: {e}"
            logger.error(error_msg)
            raise PreprocessorError(error_msg)
    
    def _segment_by_rows(self, cell_data_list: List[CellData]) -> List[TextSegment]:
        """
        按行进行切分
        
        Args:
            cell_data_list: 单元格数据列表
            
        Returns:
            行片段列表
        """
        # 按行分组
        rows_dict = {}
        for cell in cell_data_list:
            row = cell.row
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(cell)
        
        segments = []
        for row, cells in rows_dict.items():
            # 按列排序
            cells.sort(key=lambda c: c.col)
            
            # 提取文本内容
            content_parts = []
            for cell in cells:
                text = cell.value
                if text:
                    content_parts.append(text)
            
            if content_parts:
                content = " ".join(content_parts)
                
                # 计算位置范围
                min_col = min(cell.col for cell in cells)
                max_col = max(cell.col for cell in cells)
                
                segment_id = f"row_{row}"
                segment = TextSegment(
                    segment_id=segment_id,
                    content=content,
                    segment_type="row",
                    cells=cells,
                    start_pos=(row, min_col),
                    end_pos=(row, max_col)
                )
                segments.append(segment)
        
        return segments
    
    def _segment_by_columns(self, cell_data_list: List[CellData]) -> List[TextSegment]:
        """
        按列进行切分
        
        Args:
            cell_data_list: 单元格数据列表
            
        Returns:
            列片段列表
        """
        # 按列分组
        columns_dict = {}
        for cell in cell_data_list:
            col = cell.col
            if col not in columns_dict:
                columns_dict[col] = []
            columns_dict[col].append(cell)
        
        segments = []
        for col, cells in columns_dict.items():
            # 按行排序
            cells.sort(key=lambda c: c.row)
            
            # 提取文本内容
            content_parts = []
            for cell in cells:
                text = cell.value
                if text:
                    content_parts.append(text)
            
            if content_parts:
                content = " ".join(content_parts)
                
                # 计算位置范围
                min_row = min(cell.row for cell in cells)
                max_row = max(cell.row for cell in cells)
                
                segment_id = f"col_{col}"
                segment = TextSegment(
                    segment_id=segment_id,
                    content=content,
                    segment_type="column",
                    cells=cells,
                    start_pos=(min_row, col),
                    end_pos=(max_row, col)
                )
                segments.append(segment)
        
        return segments
    
    def _segment_by_cells(self, cell_data_list: List[CellData]) -> List[TextSegment]:
        """
        按单元格进行切分
        
        Args:
            cell_data_list: 单元格数据列表
            
        Returns:
            单元格片段列表
        """
        segments = []
        
        for cell in cell_data_list:
            text = cell.value
            if text:
                segment_id = f"cell_{cell.row}_{cell.col}"
                segment = TextSegment(
                    segment_id=segment_id,
                    content=text,
                    segment_type="cell",
                    cells=[cell],
                    start_pos=(cell.row, cell.col),
                    end_pos=(cell.row, cell.col)
                )
                segments.append(segment)
        
        return segments
    
    
    def _filter_segments(self, segments: List[TextSegment], segment_types: int) -> List[TextSegment]:
        """
        过滤空或过短的片段
        
        Args:
            segments: 原始片段列表
            
        Returns:
            过滤后的片段列表
        """
        valid_segments = []
        
        for segment in segments:
            if segment.content and (segment_types == 1 or len(segment.content.strip()) >= self.min_content_length):
                valid_segments.append(segment)
            else:
                logger.debug(f"Filtered out segment {segment.segment_id} (content too short)")
        
        return valid_segments


# 全局实例
preprocessor = None

def get_preprocessor(min_content_length: int = 1) -> Preprocessor:
    """
    获取预处理器实例
    
    Args:
        min_content_length: 最小内容长度
        
    Returns:
        预处理器实例
    """
    global preprocessor
    if preprocessor is None:
        preprocessor = Preprocessor(min_content_length=min_content_length)
    return preprocessor
