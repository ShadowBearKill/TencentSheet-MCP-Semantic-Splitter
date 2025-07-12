"""
表格范围计算工具模块
提供表格范围计算、验证和优化功能
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from utils.logger import logger


class RangeCalculatorError(Exception):
    """范围计算异常"""
    pass


class SheetInfo:
    """工作表信息"""
    
    def __init__(self, sheet_id: str, row_count: int, column_count: int, title: str = ""):
        self.sheet_id = sheet_id
        self.row_count = row_count
        self.column_count = column_count
        self.title = title
    
    def __repr__(self):
        return f"SheetRangeInfo(sheet_id='{self.sheet_id}', rows={self.row_count}, cols={self.column_count})"


class RangeCalculator:
    """表格范围计算器"""
    
    # API限制常量
    MAX_ROWS = 1000
    MAX_COLUMNS = 200
    MAX_CELLS = 10000
    
    def __init__(self):
        """初始化范围计算器"""
        pass
    
    def extract_sheet_range_info(self, sheet_info: Dict[str, Any]) -> List[SheetInfo]:
        """
        从表格信息中提取所有工作表的范围信息

        Args:
            sheet_info: 表格信息字典
            
        Returns:
            工作表信息列表
        """
        try:
            properties = sheet_info.get('properties', [])
            if not isinstance(properties, list):
                logger.warning("Properties is not a list, returning empty list")
                return []
            
            sheets = []
            for prop in properties:
                if not isinstance(prop, dict):
                    continue
                
                sheet_id = prop.get('sheetId')
                if not sheet_id:
                    logger.warning(f"Missing sheetId in property: {prop}")
                    continue
                
                # 获取行数和列数，默认值为0
                row_count = prop.get('rowCount', 0)
                column_count = prop.get('columnCount', 0)
                title = prop.get('title', '')
                
                # 如果没有数据，跳过该工作表
                if row_count <= 0 or column_count <= 0:
                    logger.warning(f"Sheet {sheet_id} has no data (rows={row_count}, cols={column_count})")
                    continue
                
                sheet = SheetInfo(sheet_id, row_count, column_count, title)
                sheets.append(sheet)
                logger.debug(f"Extracted range info: {sheet}")
            
            logger.info(f"Extracted {len(sheets)} sheet range infos")
            return sheets
            
        except Exception as e:
            logger.error(f"Failed to extract sheet range info: {e}")
            raise RangeCalculatorError(f"Failed to extract sheet range info: {e}")


    def split_large_range(self, sheet_range: SheetInfo) -> List[str]:
        """
        将超出限制的大范围分割为多个符合API限制的子范围

        Args:
            sheet_range: 工作表范围信息

        Returns:
            A1表示法的查询范围列表

        Raises:
            RangeCalculatorError: 分割失败时抛出
        """
        try:
            ranges = []

            # 确定分割策略
            if sheet_range.column_count <= self.MAX_COLUMNS:
                # 列数不超限，使用行分割
                ranges = self._split_by_rows(sheet_range)
            else:
                # 列数超限，使用块分割
                ranges = self._split_by_blocks(sheet_range)

            logger.info(f"Split large range for sheet {sheet_range.sheet_id} into {len(ranges)} sub-ranges")
            return ranges

        except Exception as e:
            logger.error(f"Failed to split large range: {e}")
            raise RangeCalculatorError(f"Failed to split large range: {e}")

    def _split_by_rows(self, sheet_range: SheetInfo) -> List[str]:
        """
        按行分割范围

        Args:
            sheet_range: 工作表范围信息

        Returns:
            A1表示法的查询范围列表
        """
        ranges = []
        end_col = self._number_to_column_letter(sheet_range.column_count)

        # 计算每个子范围的最大行数
        max_rows_per_chunk = min(self.MAX_ROWS, self.MAX_CELLS // sheet_range.column_count)

        current_row = 1
        # 遍历所有行，生成子范围
        while current_row <= sheet_range.row_count:
            end_row = min(current_row + max_rows_per_chunk - 1, sheet_range.row_count)
            range_str = f"A{current_row}:{end_col}{end_row}"
            ranges.append(range_str)
            current_row = end_row + 1

        logger.debug(f"Row split: {len(ranges)} ranges with max {max_rows_per_chunk} rows each")
        return ranges

    def _split_by_blocks(self, sheet_range: SheetInfo) -> List[str]:
        """
        按块分割范围

        Args:
            sheet_range: 工作表范围信息

        Returns:
            A1表示法的查询范围列表
        """
        ranges = []

        # 计算块大小，列取最大值，行根据列选取
        max_cols_per_chunk = self.MAX_COLUMNS
        max_rows_per_chunk = min(self.MAX_ROWS, self.MAX_CELLS // max_cols_per_chunk)

        # 先遍历行块，再遍历列块
        current_row = 1
        while current_row <= sheet_range.row_count:
            end_row = min(current_row + max_rows_per_chunk - 1, sheet_range.row_count)

            current_col = 1
            while current_col <= sheet_range.column_count:
                end_col_num = min(current_col + max_cols_per_chunk - 1, sheet_range.column_count)

                start_col = self._number_to_column_letter(current_col)
                end_col = self._number_to_column_letter(end_col_num)

                range_str = f"{start_col}{current_row}:{end_col}{end_row}"
                ranges.append(range_str)

                current_col = end_col_num + 1

            current_row = end_row + 1

        logger.debug(f"Block split: {len(ranges)} ranges with max {max_rows_per_chunk}x{max_cols_per_chunk} each")
        return ranges
    
    def _number_to_column_letter(self, num: int) -> str:
        """
        将列号转换为字母表示（1=A, 2=B, ..., 26=Z, 27=AA, ...）
        
        Args:
            num: 列号（从1开始）
            
        Returns:
            列字母表示
        """
        result = ""
        while num > 0:
            num -= 1
            result = chr(ord('A') + num % 26) + result
            num //= 26
        return result
    
    def _parse_a1_notation(self, range_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        解析A1表示法
        
        Args:
            range_str: A1表示法字符串，如 "A1:D10"
            
        Returns:
            ((start_row, start_col), (end_row, end_col)) 元组，行列从1开始
        """
        # 验证格式
        pattern = r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$'
        match = re.match(pattern, range_str.upper())
        if not match:
            raise ValueError(f"Invalid A1 notation: {range_str}")
        
        start_col_str, start_row_str, end_col_str, end_row_str = match.groups()
        
        # 转换列字母为数字
        start_col = self._column_letter_to_number(start_col_str)
        end_col = self._column_letter_to_number(end_col_str)
        
        # 转换行号
        start_row = int(start_row_str)
        end_row = int(end_row_str)
        
        # 验证范围有效性
        if start_row > end_row or start_col > end_col:
            raise ValueError("Start position must be before end position")
        
        return (start_row, start_col), (end_row, end_col)
    
    def _column_letter_to_number(self, letters: str) -> int:
        """
        将列字母转换为数字（A=1, B=2, ..., Z=26, AA=27, ...）
        
        Args:
            letters: 列字母，如 "A", "AB"
            
        Returns:
            列号（从1开始）
        """
        result = 0
        for char in letters:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result


# 全局实例
range_calculator = RangeCalculator()

def get_range_calculator() -> RangeCalculator:
    """
    获取范围计算器实例
    
    Returns:
        RangeCalculator实例
    """
    return range_calculator
