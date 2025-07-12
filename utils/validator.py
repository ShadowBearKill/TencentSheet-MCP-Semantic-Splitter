"""
数据验证模块
验证输入参数合法性，包括fileId、sheetId、range等
"""
import re
from typing import Optional, Tuple


class Validator:
    """数据验证类"""
    
    # 文件ID格式：字母数字组合，可能包含$符号
    FILE_ID_PATTERN = re.compile(r'^[A-Za-z0-9$]+$')
    
    # 工作表ID格式：字母数字组合
    SHEET_ID_PATTERN = re.compile(r'^[A-Za-z0-9]+$')
    
    # A1表示法单元格格式：字母+数字
    CELL_PATTERN = re.compile(r'^[A-Z]+\d+$')
    
    # A1表示法范围格式：单元格:单元格
    RANGE_PATTERN = re.compile(r'^[A-Z]+\d+:[A-Z]+\d+$')
    
    @staticmethod
    def validate_file_id(file_id: str) -> bool:
        """
        验证文件ID格式
        
        Args:
            file_id: 文件ID
        
        Returns:
            是否有效
        """
        if not file_id or not isinstance(file_id, str):
            return False
        
        return bool(Validator.FILE_ID_PATTERN.match(file_id))
    
    @staticmethod
    def validate_sheet_id(sheet_id: str) -> bool:
        """
        验证工作表ID格式
        
        Args:
            sheet_id: 工作表ID
        
        Returns:
            是否有效
        """
        if not sheet_id or not isinstance(sheet_id, str):
            return False
        
        return bool(Validator.SHEET_ID_PATTERN.match(sheet_id))
    
    @staticmethod
    def validate_cell_reference(cell_ref: str) -> bool:
        """
        验证单元格引用格式（A1表示法）
        
        Args:
            cell_ref: 单元格引用，如"A1", "B10", "AA100"
        
        Returns:
            是否有效
        """
        if not cell_ref or not isinstance(cell_ref, str):
            return False
        
        return bool(Validator.CELL_PATTERN.match(cell_ref.upper()))
    
    @staticmethod
    def validate_range(range_str: str) -> bool:
        """
        验证范围格式（A1表示法）
        
        Args:
            range_str: 范围字符串，如"A1:B10", "C5:D20"
        
        Returns:
            是否有效
        """
        if not range_str or not isinstance(range_str, str):
            return False
        
        # 检查基本格式
        if not Validator.RANGE_PATTERN.match(range_str.upper()):
            return False
        
        # 分解起始和结束单元格
        start_cell, end_cell = range_str.upper().split(':')
        
        # 验证两个单元格都有效
        if not (Validator.validate_cell_reference(start_cell) and 
                Validator.validate_cell_reference(end_cell)):
            return False
        
        # 验证范围逻辑（起始位置应该在结束位置之前或相等）
        return Validator._is_valid_range_order(start_cell, end_cell)
    
    @staticmethod
    def _is_valid_range_order(start_cell: str, end_cell: str) -> bool:
        """
        验证范围顺序是否正确
        
        Args:
            start_cell: 起始单元格
            end_cell: 结束单元格
        
        Returns:
            是否有效
        """
        start_col, start_row = Validator._parse_cell_reference(start_cell)
        end_col, end_row = Validator._parse_cell_reference(end_cell)
        
        # 列和行都应该是起始 <= 结束
        return start_col <= end_col and start_row <= end_row
    
    @staticmethod
    def _parse_cell_reference(cell_ref: str) -> Tuple[int, int]:
        """
        解析单元格引用为列号和行号
        
        Args:
            cell_ref: 单元格引用，如"A1", "B10"
        
        Returns:
            (列号, 行号) 元组
        """
        # 分离字母和数字
        col_letters = ""
        row_digits = ""
        
        for char in cell_ref:
            if char.isalpha():
                col_letters += char
            else:
                row_digits += char
        
        # 将列字母转换为数字（A=1, B=2, ..., Z=26, AA=27, ...）
        col_num = 0
        for char in col_letters:
            col_num = col_num * 26 + (ord(char) - ord('A') + 1)
        
        row_num = int(row_digits)
        
        return col_num, row_num
    
    @staticmethod
    def validate_range_size(range_str: str, max_rows: int = 1000, 
                          max_cols: int = 200, max_cells: int = 10000) -> bool:
        """
        验证范围大小是否符合API限制
        
        Args:
            range_str: 范围字符串
            max_rows: 最大行数
            max_cols: 最大列数
            max_cells: 最大单元格数
        
        Returns:
            是否符合限制
        """
        if not Validator.validate_range(range_str):
            return False
        
        start_cell, end_cell = range_str.upper().split(':')
        start_col, start_row = Validator._parse_cell_reference(start_cell)
        end_col, end_row = Validator._parse_cell_reference(end_cell)
        
        # 计算范围大小
        rows = end_row - start_row + 1
        cols = end_col - start_col + 1
        total_cells = rows * cols
        
        # 检查限制
        return (rows <= max_rows and 
                cols <= max_cols and 
                total_cells <= max_cells)
    
    @staticmethod
    def validate_api_params(file_id: str, sheet_id: str, range_str: str) -> Tuple[bool, Optional[str]]:
        """
        验证API调用参数
        
        Args:
            file_id: 文件ID
            sheet_id: 工作表ID
            range_str: 范围字符串
        
        Returns:
            (是否有效, 错误信息)
        """
        # 验证文件ID
        if not Validator.validate_file_id(file_id):
            return False, f"Invalid file ID: {file_id}"
        
        # 验证工作表ID
        if not Validator.validate_sheet_id(sheet_id):
            return False, f"Invalid sheet ID: {sheet_id}"
        
        # 验证范围
        if not Validator.validate_range(range_str):
            return False, f"Invalid range format: {range_str}"
        
        # 验证范围大小
        if not Validator.validate_range_size(range_str):
            return False, f"Range size exceeds API limits: {range_str}"
        
        return True, None


# 便捷函数
def validate_file_id(file_id: str) -> bool:
    """验证文件ID"""
    return Validator.validate_file_id(file_id)


def validate_sheet_id(sheet_id: str) -> bool:
    """验证工作表ID"""
    return Validator.validate_sheet_id(sheet_id)


def validate_range(range_str: str) -> bool:
    """验证范围格式"""
    return Validator.validate_range(range_str)


def validate_api_params(file_id: str, sheet_id: str, range_str: str) -> Tuple[bool, Optional[str]]:
    """验证API参数"""
    return Validator.validate_api_params(file_id, sheet_id, range_str)
