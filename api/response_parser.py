"""
API响应解析器模块
解析腾讯文档API返回的复杂数据结构，转换为内部统一的数据格式
"""
from typing import Dict, List, Optional, Any, Union
from utils.logger import logger


class ResponseParseError(Exception):
    pass


class CellData:
    """单元格数据类"""
    
    def __init__(self, row: int, col: int, value: str, data_type: str, 
                 cell_format: str):
        """
        初始化单元格数据
        
        Args:
            row: 行号（从0开始）
            col: 列号（从0开始）
            value: 单元格值“字符串”类型
            data_type: 数据类型
            cell_format: 单元格格式
        """
        self.row = row
        self.col = col
        self.value = value
        self.data_type = data_type
        self.cell_format = cell_format
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "data_type": self.data_type,
            "cell_format": self.cell_format
        }
    
    def __repr__(self) -> str:
        return f"CellData(row={self.row}, col={self.col}, value={self.value}, type={self.data_type})"



class ResponseParser:
    """API响应解析器类"""
    
    def parse_sheet_data(self, response_data: Dict[str, Any]) -> List[CellData]:
        """
        解析表格数据API响应
        
        Args:
            response_data: API响应数据
            
        Returns:
            单元格数据列表
        """
        try:
            # 检查响应格式
            if not isinstance(response_data, dict):
                raise ResponseParseError("Response data must be a dictionary")
            
            # 提取gridData
            griddata = response_data.get('data', {}).get('gridData', {})
            
            if not griddata:
                logger.warning("No gridData found in response")
                return []
            
            # 提取起始位置
            start_row = griddata.get('startRow', 0)
            start_col = griddata.get('startColumn', 0)
            
            # 提取行数据
            rows = griddata.get('rows', [])
            if not isinstance(rows, list):
                raise ResponseParseError("Rows must be a list")
            
            cell_data_list = []
            
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    logger.warning(f"Skipping invalid row at index {row_index}: {row}")
                    continue
                values = row.get('values', [])
                if not isinstance(values, list):
                    logger.warning(f"Invalid values in row {row_index}: {values}")
                    continue
                # 提取行中的单元格数据
                for col_index, cell in enumerate(values):
                    # 计算实际位置
                    actual_row = start_row + row_index
                    actual_col = start_col + col_index
                    
                    # 提取单元格数据
                    cell_data = self.extract_cell_data(actual_row, actual_col, cell)
                    if cell_data:
                        cell_data_list.append(cell_data)
            
            logger.info(f"Parsed {len(cell_data_list)} cell data records")
            return cell_data_list
            
        except Exception as e:
            error_msg = f"Failed to parse sheet data response: {e}"
            logger.error(error_msg)
            raise ResponseParseError(error_msg)
    
    def extract_cell_data(self, row: int, col: int, cell: Dict[str, Any]) -> Optional[CellData]:
        """
        提取单元格数据并处理不同数据类型

        Args:
            row: 行号
            col: 列号
            cell: 单元格数据字典

        Returns:
            CellData对象，如果无效返回None
        """
        try:
            if not isinstance(cell, dict):
                logger.warning(f"Invalid cell data at ({row}, {col}): {cell}")
                return None

            # 提取腾讯文档API格式的基本信息
            cell_value = cell.get('cellValue', {})
            data_type = cell.get('dataType', 'DATA_TYPE_UNSPECIFIED')
            cell_format = cell.get('cellFormat', "null")

            return CellData(row, col, str(cell_value), str(data_type), str(cell_format))

        except Exception as e:
            logger.warning(f"Failed to extract cell data at ({row}, {col}): {e}")
            return None

# 全局实例
response_parser = ResponseParser()

def get_response_parser() -> ResponseParser:
    """
    获取响应解析器实例
    """
    return response_parser
