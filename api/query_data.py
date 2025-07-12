"""
表格数据获取API模块
实现获取指定范围表格数据的功能
"""
import requests
from typing import Dict, List, Optional, Any
from utils.logger import logger
from utils.validator import validate_file_id, validate_sheet_id, validate_range
from utils.range_calculator import get_range_calculator, SheetInfo
from config.params import config


class SheetDataQueryError(Exception):
    pass


class SheetDataQuery:
    """表格数据查询类"""
    
    def __init__(self):
        self.base_url = config.sheet_base_url
        self.timeout = 30  
        self.range_calculator = get_range_calculator()
    

    def query_single(
        self,
        file_id: str,
        sheet_id: str,
        range_str: str,
        access_token: str,
        client_id: str,
        open_id: str
    ) -> Dict[str, Any]:
        """
        查询单个范围的表格数据（内部方法）

        Args:
            file_id: 在线表格唯一标识
            sheet_id: 工作表ID
            range_str: 查询范围（A1表示法）
            access_token: 访问令牌
            client_id: 应用ID
            open_id: 开放平台用户ID

        Returns:
            包含表格数据的字典
        """
        try:
            # 构建请求URL
            url = f"{self.base_url}{file_id}/{sheet_id}/{range_str}"

            # 构建请求头
            headers = config.get_headers(access_token, client_id, open_id)

            logger.debug(f"Querying single range: {range_str}")

            # 发送请求
            response = requests.get(
                url=url,
                headers=headers,
                timeout=self.timeout
            )

            # 检查HTTP状态码
            if response.status_code != 200:
                error_msg = f"HTTP request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise SheetDataQueryError(error_msg)

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"Failed to parse JSON response: {e}"
                logger.error(error_msg)
                raise SheetDataQueryError(error_msg)

            # 检查业务返回码
            if 'ret' in data and data['ret'] != 0:
                error_msg = f"API returned error code {data['ret']}: {data.get('msg', 'Unknown error')}"
                logger.error(error_msg)
                raise SheetDataQueryError(error_msg)
            return data

        except Exception as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            raise SheetDataQueryError(error_msg)

    def merge_griddata(self, griddata_list: List[Dict[str, Any]], ranges: List[str]) -> Dict[str, Any]:
        """
        合并多个gridData为统一的数据结构

        Args:
            grid_data_list: gridData列表
            ranges: 对应的范围列表

        Returns:
            合并后的gridData
        """
        try:
            if not griddata_list:
                return {}

            if len(griddata_list) == 1:
                return griddata_list[0]

            # 解析每个范围的起始位置
            range_positions = []
            for range_str in ranges:
                start_cell, _ = self.range_calculator.parse_a1(range_str)
                range_positions.append(start_cell)

            # 初始化合并结果
            merge_data = {
                "columnMetadata": [],
                "rowMetadata": [],
                "rows": [],
                "startColumn": 0,
                "startRow": 0
            }

            # 计算全局范围
            min_row = min(pos[0] for pos in range_positions)
            min_col = min(pos[1] for pos in range_positions)
            max_row = 0
            max_col = 0

            # 创建行数据映射
            row_data_map = {}

            for griddata, (start_row, start_col) in zip(griddata_list, range_positions):
                if not griddata or 'rows' not in griddata:
                    continue

                rows = griddata.get('rows', [])

                for row_idx, row in enumerate(rows):
                    # 计算全局行号
                    global_row = start_row + row_idx - 1

                    if global_row not in row_data_map: # 创建全局行
                        row_data_map[global_row] = {}

                    values = row.get('values', [])
                    for col_idx, value in enumerate(values):
                        # 计算全局列号
                        global_col = start_col + col_idx - 1
                        row_data_map[global_row][global_col] = value
                        max_col = max(max_col, global_col)

                max_row = max(max_row, max(row_data_map.keys()) if row_data_map else 0)

            # 构建合并后的rows数组
            for row_num in range(min_row - 1, max_row + 1):
                if row_num in row_data_map:
                    row_values = []
                    for col_num in range(min_col - 1, max_col + 1):
                        if col_num in row_data_map[row_num]:
                            row_values.append(row_data_map[row_num][col_num])
                        else:
                            # 空单元格
                            row_values.append({
                                "cellFormat": None,
                                "cellValue": {"text": ""},
                                "dataType": "DATA_TYPE_UNSPECIFIED"
                            })
                    merge_data["rows"].append({"values": row_values})
                else:
                    # 填充空行
                    row_values = []
                    for col_num in range(min_col - 1, max_col + 1):
                        # 空单元格
                        row_values.append({
                            "cellFormat": None,
                            "cellValue": {"text": ""},
                            "dataType": "DATA_TYPE_UNSPECIFIED"
                        })
                    merge_data["rows"].append({"values": row_values})

            merge_data["startRow"] = min_row - 1  # 转换为0基索引
            merge_data["startColumn"] = min_col - 1  # 转换为0基索引

            logger.info(f"Merged {len(griddata_list)} grid data into single result "
                       f"({len(merge_data['rows'])} rows, {max_col - min_col + 2} columns)")

            return merge_data

        except Exception as e:
            logger.error(f"Failed to merge grid data: {e}")
            raise SheetDataQueryError(f"Failed to merge grid data: {e}")

    def query_sheet_data(
        self,
        file_id: str,
        sheet_id: str,
        range_str: str,
        access_token: str,
        client_id: str,
        open_id: str
    ) -> Dict[str, Any]:
        """
        查询表格数据（支持大范围自动分块查询）

        Args:
            file_id: 在线表格唯一标识
            sheet_id: 工作表ID
            range_str: 查询范围（A1表示法）
            access_token: 访问令牌
            client_id: 应用ID
            open_id: 开放平台用户ID

        Returns:
            包含表格数据的字典
        """
        try:
            # 验证参数
            if not validate_range(range_str):
                raise SheetDataQueryError(f"Invalid range format: {range_str}")

            logger.info(f"Querying sheet data: file_id={file_id}, sheet_id={sheet_id}, range={range_str}")
            return self.query_single(file_id, sheet_id, range_str, access_token, client_id, open_id)


        except Exception as e:
            logger.error(f"Failed to query sheet data: {e}")
            raise SheetDataQueryError(f"Failed to query sheet data: {e}")

    def query_block(
        self,
        file_id: str,
        sheet_id: str,
        range_str: str,
        access_token: str,
        client_id: str,
        open_id: str
    ) -> Dict[str, Any]:
        """
        使用分块策略查询大范围数据

        Args:
            file_id: 文件ID
            sheet_id: 工作表ID
            range_str: 查询范围（A1表示法）
            access_token: 访问令牌
            client_id: 客户端ID
            open_id: 用户开放ID

        Returns:
            包含合并后表格数据的字典
        """
        try:
            # 解析范围以确定工作表大小
            start_cell, end_cell = self.range_calculator.parse_a1(range_str)
            end_row, end_col = end_cell

            # 临时SheetInfo
            temp_sheet_info = SheetInfo(sheet_id, end_row, end_col)

            # 获取分块范围列表，chunk为ranges_str的列表
            adjusted_ranges = self.range_calculator.split_range(temp_sheet_info)

            logger.info(f"Split range {range_str} into {len(adjusted_ranges)} chunks")

            # 执行分块查询
            griddata_list = []
            successful_ranges = []

            for i, block_range in enumerate(adjusted_ranges, 1):
                try:
                    logger.debug(f"Querying chunk {i}/{len(adjusted_ranges)}: {block_range}")
                    # 获得子块的数据
                    result = self.query_sheet_data(
                        file_id=file_id,
                        sheet_id=sheet_id,
                        range_str=block_range,
                        access_token=access_token,
                        client_id=client_id,
                        open_id=open_id
                    )
                    
                    griddata = result.get('data', {}).get('gridData')
                    if griddata:
                        griddata_list.append(griddata)
                        successful_ranges.append(block_range)
                        logger.debug(f"Successfully queried chunk {i}")
                    else:
                        logger.warning(f"No grid data in chunk {i}: {block_range}")

                except Exception as e:
                    logger.error(f"Failed to query chunk {i} ({block_range}): {e}")
                    # 继续处理其他块
                    continue

            if not griddata_list:
                raise SheetDataQueryError("All chunk queries failed")

            # 合并gridData
            logger.info(f"Merging {len(griddata_list)} successful chunks")
            merged_griddata = self.merge_griddata(griddata_list, successful_ranges)

            # 构建最终响应
            result = {
                "ret": 0,
                "msg": "Succeed",
                "data": {
                    "gridData": merged_griddata
                }
            }

            logger.info(f"Successfully completed chunked query for range {range_str}")
            return result

        except Exception as e:
            logger.error(f"Failed chunked query: {e}")
            raise SheetDataQueryError(f"Failed chunked query: {e}")

# 全局实例
sheet_data_query = SheetDataQuery()

def get_sheet_data_query() -> SheetDataQuery:
    """
    获取表格数据查询器实例
    """
    return sheet_data_query
