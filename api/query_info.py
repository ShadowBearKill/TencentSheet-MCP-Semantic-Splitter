"""
在线表格信息查询API模块
实现查询工作表基本信息的功能
"""
import requests
from typing import Dict, List, Optional, Any
from utils.logger import logger
from config.params import config


class SheetInfoQueryError(Exception):
    pass


class SheetInfoQuery:
    """在线表格信息查询类"""
    
    def __init__(self):
        self.base_url = config.sheet_base_url
        self.timeout = 30  # 请求超时时间
    
    def query(
        self, 
        file_id: str, 
        access_token: str, 
        client_id: str, 
        open_id: str,
        concise: bool = True
    ) -> Dict[str, Any]:
        """
        查询工作表信息
        Args:
            file_id: 在线表格唯一标识
            access_token: 访问令牌
            client_id: 应用ID
            open_id: 开放平台用户ID
            concise: 是否返回简洁信息
            
        Returns:
            包含工作表信息的字典
        """
        try:
            url = f"{self.base_url}{file_id}"
            if concise:
                url += "?concise=1"
            else:
                url += "?concise=0"
            
            # 请求头
            headers = config.get_headers(access_token, client_id, open_id)
            
            logger.info(f"Querying sheet info for file_id: {file_id}")
            logger.debug(f"Request URL: {url}")
            
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
                raise SheetInfoQueryError(error_msg)
            
            # 转成json
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"Failed to parse JSON response: {e}"
                logger.error(error_msg)
                raise SheetInfoQueryError(error_msg)
            
            # 检查业务返回码
            if 'code' in data and data['code'] != 0:
                error_msg = f"API returned error code {data['code']}: {data.get('message', 'Unknown error')}"
                logger.error(error_msg)
                raise SheetInfoQueryError(error_msg)
            
            logger.info(f"Successfully queried sheet info for file_id: {file_id}")
            return data
            
        except Exception as e:
            error_msg = f"Unknown error: {e}"
            logger.error(error_msg)
            raise SheetInfoQueryError(error_msg)

# 全局实例
sheet_info_query = SheetInfoQuery()

def get_sheet_info_query() -> SheetInfoQuery:
    """
    获取表格信息查询器实例
    """
    return sheet_info_query
