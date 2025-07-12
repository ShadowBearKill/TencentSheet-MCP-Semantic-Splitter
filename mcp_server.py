"""
Sheet MCP-Semantic-Splitter MCP服务器
基于FastMCP框架的本地MCP服务器，提供表格语义分割功能
"""
import json
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from utils.logger import logger
from main import SemanticSplitter, SemanticSplitterError

# 创建FastMCP应用实例
mcp = FastMCP("Sheet-Semantic-Splitter")

# 全局语义分割器实例
_splitter: Optional[SemanticSplitter] = None

def get_splitter() -> SemanticSplitter:
    """获取语义分割器实例（单例模式）"""
    global _splitter
    if _splitter is None:
        _splitter = SemanticSplitter()
        logger.info("SemanticSplitter instance created for MCP server")
    return _splitter


@mcp.tool()
def semantic_split_sheet(
    sheet_access_token: str,
    sheet_client_id: str,
    file_id: str,
    user_open_id: str,
    split_type: int
) -> Dict[str, Any]:
    """
    对在线表格进行语义分割（自动处理所有工作表）

    Args:
        sheet_access_token: 表格访问令牌
        sheet_client_id: 客户端ID
        sheet_open_id: 表格开放ID（表格文件标识符）
        user_open_id: 用户开放ID（用户身份标识符）
        split_type: 分割类型（1: 单元格分割，2: 行分割，3: 段落分割）

    Returns:
        包含分割结果的字典，包括segments、summary、metadata和quality_report
        对于多工作表，segments中每个项目会包含sheet_id标识
    """
    try:
        logger.info(f"MCP Tool: semantic_split_sheet called for sheet {file_id}")
        
        # 获取分割器实例
        splitter = get_splitter()
        
        # 执行语义分割
        result = splitter.process_sheet(
            sheet_access_token=sheet_access_token,
            sheet_client_id=sheet_client_id,
            file_id=file_id,
            split_type=split_type,
            user_open_id=user_open_id
        )
        
        logger.info(f"MCP Tool: semantic_split_sheet completed successfully for sheet {file_id}")
        return result
        
    except SemanticSplitterError as e:
        error_msg = f"Semantic splitting failed: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in semantic_split_sheet: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)

def main():
    """启动MCP服务器"""
    try:
        logger.info("Starting Sheet MCP-Semantic-Splitter MCP Server...")
        
        get_splitter()
        
        logger.info("MCP Server initialized successfully")
        logger.info("Available tools: semantic_split_sheet, query_sheet_info, get_split_quality_report, get_sheet_list, get_server_status")
        
        # 运行MCP服务器
        mcp.run()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


if __name__ == "__main__":
    main()
