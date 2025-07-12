"""
Sheet MCP-Semantic-Splitter 主程序
协调各模块完整流程，从输入到输出完成整个处理流程
"""
import sys
import argparse
import json
from typing import Dict, Any, Optional, List
from utils.logger import logger
from utils.temp_file_manager import get_temp_file_manager
from utils.range_calculator import get_range_calculator, SheetInfo
from config.params import get_config

# 导入各个核心模块
from api.query_info import get_sheet_info_query
from api.query_data import get_sheet_data_query
from api.response_parser import get_response_parser
from core.preprocessor import get_preprocessor
from core.similarity_calculator import get_similarity_calculator
from core.segment_decider import get_segment_decider
from core.segment_optimizer import get_segment_optimizer
from core.output_generator import get_output_generator


class SemanticSplitterError(Exception):
    """语义分割器异常"""
    pass


class SemanticSplitter:
    """语义分割器主类"""
    
    def __init__(self):
        """初始化语义分割器"""
        self.config = get_config()
        self.temp_manager = get_temp_file_manager()
        self.range_calculator = get_range_calculator()

        # 初始化各个模块
        self.info_query = get_sheet_info_query()
        self.data_query = get_sheet_data_query()
        self.response_parser = get_response_parser()
        self.preprocessor = get_preprocessor()
        self.similarity_calculator = get_similarity_calculator()
        self.segment_decider = get_segment_decider()
        self.segment_optimizer = get_segment_optimizer()
        self.output_generator = get_output_generator()

        logger.info("SemanticSplitter initialized successfully")
    
    def process_sheet(self, sheet_access_token: str, sheet_client_id: str,
                     file_id: str, user_open_id: str, split_type: int = 1, 
                     output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        处理在线表格的完整流程

        Args:
            sheet_access_token: 表格访问令牌
            sheet_client_id: 客户端ID
            file_id: 表格文件标识符
            user_open_id: 用户开放ID（用户身份标识符）
            split_type: 分割类型（1: 单元格分割，2: 行分割，3: 段落分割）
            output_file: 输出文件路径（可选）

        Returns:
            处理结果字典
        """
        logger.info(f"Starting to process sheet: {file_id}")
        
        try:
            # 步骤1: 获取表格信息
            logger.info("Step 1: Querying sheet information...")
            sheet_info = self._query_sheet_info(
                sheet_access_token, sheet_client_id, file_id, user_open_id
            )
            # 处理所有工作表
            logger.info("Processing all sheets in the spreadsheet")
            sheet_results = self._process_all_sheets(
                sheet_access_token, sheet_client_id, file_id, user_open_id, split_type, sheet_info
            )

            # 步骤3: 合并结果并生成输出
            logger.info("Step 3: Merging results and generating output...")
            output = self._merge_sheet_results(sheet_results, sheet_info)

            # 步骤5: 保存输出
            if output_file:
                logger.info(f"Step 5: Saving output to file: {output_file}")
                self.output_generator.save_output_to_file(output, output_file)

            logger.info("Sheet processing completed successfully")
            return output
            
        except Exception as e:
            error_msg = f"Failed to process sheet {file_id}: {e}"
            logger.error(error_msg)
            raise SemanticSplitterError(error_msg)
    
    def _query_sheet_info(self, access_token: str, client_id: str,
                         file_id: str, open_id: str) -> Dict[str, Any]:
        """查询表格信息"""
        try:
            return self.info_query.query_sheet_info(
                file_id=file_id,
                access_token=access_token,
                client_id=client_id,
                open_id=open_id,
                concise=False
            )
        except Exception as e:
            raise SemanticSplitterError(f"Failed to query sheet info: {e}")
    
    def _query_sheet_data(self, access_token: str, client_id: str,
                         file_id: str, open_id: str, split_type: int, sheet_range: SheetInfo) -> Dict[str, Any]:
        """
        查询表格数据（支持大范围自动分块查询）

        Args:
            access_token: 访问令牌
            client_id: 客户端ID
            file_id: 文件ID
            open_id: 用户开放ID
            split_type: 分割类型（1: 单元格分割，2: 行分割，3: 段落分割）
            sheet_range: 工作表范围信息

        Returns:
            包含表格数据的字典
        """
        try:
            # 检查原始工作表大小是否超出API限制
            max_rows = self.range_calculator.MAX_ROWS
            max_columns = self.range_calculator.MAX_COLUMNS
            max_cells = self.range_calculator.MAX_CELLS

            needs_chunking = (
                sheet_range.row_count > max_rows or
                sheet_range.column_count > max_columns or
                sheet_range.row_count * sheet_range.column_count > max_cells
            )
            end_col = self.range_calculator._number_to_column_letter(sheet_range.column_count)
            full_range = f"A1:{end_col}{sheet_range.row_count}"
            if needs_chunking:
                # 工作表超出限制，使用分块查询处理完整数据
                logger.info(f"Sheet {sheet_range.sheet_id} size ({sheet_range.row_count}x{sheet_range.column_count}) "
                           f"exceeds API limits, using chunked query (split_type: {split_type})")
                return self.data_query._query_with_chunking(
                    file_id, sheet_range.sheet_id, full_range, access_token, client_id, open_id
                )
            else:
                # 工作表在限制内，计算最优范围进行单次查询
                logger.info(f"Sheet {sheet_range.sheet_id} within limits, using single query: {full_range} (split_type: {split_type})")
                # 返回原始请求的字典格式数据
                return self.data_query.query_sheet_data(
                    file_id=file_id,
                    sheet_id=sheet_range.sheet_id,
                    range_str=full_range,
                    access_token=access_token,
                    client_id=client_id,
                    open_id=open_id
                )
            
        except Exception as e:
            raise SemanticSplitterError(f"Failed to query sheet data: {e}")


    def _extract_sheet_range_info(self, sheet_info: Dict[str, Any]) -> List[SheetInfo]:
        """从表格信息中提取工作表信息"""
        try:
            return self.range_calculator.extract_sheet_range_info(sheet_info)
        except Exception as e:
            logger.error(f"Failed to extract sheet range info: {e}")
            return []

    def _process_single_sheet(self, access_token: str, client_id: str,
                             file_id: str, open_id: str, split_type: int, sheet_range: SheetInfo) -> Dict[str, Any]:
        """
        处理单个工作表

        Args:
            access_token: 访问令牌
            client_id: 客户端ID
            file_id: 文件ID
            open_id: 用户开放ID
            split_type: 分割类型（1: 单元格分割，2: 行分割，3: 段落分割）
            sheet_range: 工作表信息

        Returns:
            处理结果字典
        """
        try:
            # 获取完整表格数据
            sheet_data = self._query_sheet_data(access_token, client_id, file_id, open_id, split_type, sheet_range)

            # 解析响应数据
            parsed_data = self._parse_response_data(sheet_data)

            # 数据预处理
            segments = self._preprocess_data(parsed_data, split_type)

            # 分割
            groups = self._make_segmentation_decisions(segments)

            # 片段优化
            optimized_groups = self._optimize_segments(groups)

            return {
                'sheet_id': sheet_range.sheet_id,
                'segments': optimized_groups,
                'parsed_data': parsed_data
            }

        except Exception as e:
            logger.error(f"Failed to process sheet {sheet_range.sheet_id}: {e}")
            raise SemanticSplitterError(f"Failed to process sheet {sheet_range.sheet_id}: {e}")

    def _process_all_sheets(self, access_token: str, client_id: str,
                           file_id: str, open_id: str, split_type: int, sheet_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理所有工作表

        Args:
            access_token: 访问令牌
            client_id: 客户端ID
            file_id: 文件ID
            open_id: 用户开放ID
            split_type: 分割类型（1: 单元格分割，2: 行分割，3: 段落分割）
            sheet_info: 表格信息字典

        Returns:
            处理结果列表
        """
        try:
            # 提取所有工作表信息
            sheets = self._extract_sheet_range_info(sheet_info)

            if not sheets:
                raise SemanticSplitterError("No sheets found in the spreadsheet")

            logger.info(f"Processing {len(sheets)} sheets with split_type: {split_type}")

            results = []
            # 处理所有工作表
            for i, sheet in enumerate(sheets):
                logger.info(f"Processing sheet {i}/{len(sheets)}: {sheet.sheet_id} "
                           f"(size: {sheet.row_count}x{sheet.column_count})")
                try:
                    result = self._process_single_sheet(access_token, client_id, file_id, open_id, split_type, sheet)
                    results.append(result)
                    logger.info(f"Successfully processed sheet {sheet.sheet_id}")
                except Exception as e:
                    logger.error(f"Failed to process sheet {sheet.sheet_id}: {e}")
                    results.append({
                        'sheet_id': sheet.sheet_id,
                        'error': str(e),
                        'segments': [],
                        'parsed_data': {}
                    })

            return results

        except Exception as e:
            raise SemanticSplitterError(f"Failed to process all sheets: {e}")

    def _merge_sheet_results(self, sheet_results: List[Dict[str, Any]], sheet_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """合并多个工作表的处理结果"""
        try:
            sheets_data = []
            processed_sheets = []
            failed_sheets = []

            for result in sheet_results:
                sheet_id = result.get('sheet_id', 'unknown')

                if 'error' in result:
                    failed_sheets.append({
                        'sheet_id': sheet_id,
                        'error': result['error']
                    })
                else:
                    processed_sheets.append(sheet_id)
                    # 获取SegmentGroup列表
                    segment_groups = result.get('segments', [])

                    # 将SegmentGroup转换为输出格式
                    sheet_segments = []
                    for group in segment_groups:
                        output_segment = self._convert_segment_group_to_output(group)
                        sheet_segments.append(output_segment)
                    sheet_data = {
                        'sheet_id': sheet_id,
                        'segments': sheet_segments
                    }
                    sheets_data.append(sheet_data)

            # 返回按sheet分组的结构
            return {
                'segments': sheets_data
            }

        except Exception as e:
            raise SemanticSplitterError(f"Failed to merge sheet results: {e}")

    def _convert_segment_group_to_output(self, group) -> Dict[str, Any]:
        """
        将SegmentGroup转换为输出格式

        Args:
            group: SegmentGroup对象

        Returns:
            输出格式的字典
        """
        # 为每个组生成一个合并的片段
        group_content = []
        group_positions = []
        segment_ids = []

        for segment in group.segments:
            group_content.append(segment.content)
            # 使用start_pos和end_pos构建位置信息
            position_info = {
                "start_pos": segment.start_pos,
                "end_pos": segment.end_pos
            }
            group_positions.append(position_info)
            segment_ids.append(segment.segment_id)

        # 确定组的类型
        segment_type = self._determine_group_type(group)

        # 创建输出片段
        output_segment = {
            "segment_id": group.group_id,
            "content": "\n".join(group_content),
            "content_length": sum(len(content) for content in group_content),
            "sub_segments": len(group.segments),
            "sub_segment_ids": segment_ids,
            "segment_type": segment_type,
            "positions": group_positions,
        }
        return output_segment

    def _determine_group_type(self, group) -> str:
        """
        确定组的类型

        Args:
            group: 片段组

        Returns:
            组类型字符串
        """
        if not group.segments:
            return "empty"

        # 统计各种类型的片段
        type_counts = {}
        for segment in group.segments:
            segment_type = getattr(segment, 'segment_type', 'text')
            type_counts[segment_type] = type_counts.get(segment_type, 0) + 1

        # 返回最多的类型
        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]

        return "mixed"

    def _parse_response_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析响应数据"""
        try:
            # 解析表格数据响应，返回CellData列表
            cells = self.response_parser.parse_sheet_data_response(raw_data)
            return {"cells": cells}
        except Exception as e:
            raise SemanticSplitterError(f"Failed to parse response data: {e}")
    
    def _preprocess_data(self, parsed_data: Dict[str, Any], split_type: int):
        """数据预处理"""
        try:
            # 从解析的数据中提取单元格数据
            cells = parsed_data.get('cells', [])
            if not cells:
                logger.warning("No cells found in parsed data")
                return []
            
            # 预处理单元格数据
            return self.preprocessor.preprocess(cells, split_type)
        except Exception as e:
            raise SemanticSplitterError(f"Failed to preprocess data: {e}")
    

    def _make_segmentation_decisions(self, segments):
        """分割决策"""
        try:
            # 内部会计算相似度
            return self.segment_decider.decide_segments(segments)
        except Exception as e:
            raise SemanticSplitterError(f"Failed to make segmentation decisions: {e}")
    
    def _optimize_segments(self, groups):
        """片段优化"""
        try:
            return self.segment_optimizer.optimize_segments(groups)
        except Exception as e:
            raise SemanticSplitterError(f"Failed to optimize segments: {e}")
    

def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Sheet MCP-Semantic-Splitter - 智能表格语义分割工具"
    )
    
    parser.add_argument(
        "--access-token", 
        required=True,
        help="表格访问令牌"
    )
    
    parser.add_argument(
        "--client-id", 
        required=True,
        help="客户端ID"
    )
    
    parser.add_argument(
        "--open-id",
        required=True,
        help="表格开放ID（表格文件标识符）"
    )

    parser.add_argument(
        "--user-open-id",
        required=True,
        help="用户开放ID（用户身份标识符）"
    )
    
    parser.add_argument(
        "--sheet-id",
        help="工作表ID（可选）"
    )

    parser.add_argument(
        "--split-type",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="分割类型（1: 单元格分割，2: 行分割，3: 段落分割，默认为1）"
    )

    parser.add_argument(
        "--output", "-o",
        help="输出文件路径（可选）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细日志输出"
    )
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel("DEBUG")
    
    try:
        # 创建语义分割器实例
        splitter = SemanticSplitter()
        
        # 处理表格
        result = splitter.process_sheet(
            sheet_access_token=args.access_token,
            sheet_client_id=args.client_id,
            file_id=args.open_id,
            user_open_id=args.user_open_id,
            split_type=args.split_type,
            output_file=args.output
        )
        
        # 如果没有指定输出文件，打印结果到控制台
        if not args.output:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        logger.info("Processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
