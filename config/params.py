"""
配置参数管理模块
支持从环境变量和配置文件读取参数
"""
import os
from typing import Optional


class Config:
    """配置管理类"""
    
    def __init__(self):
        """初始化配置"""
        self._load_config()
    
    def _load_config(self):
        """加载配置参数"""
        # OpenAI API配置
        self.openai_api_key = self._get_env_var("OPENAI_API_KEY")
        self.openai_model = self._get_env_var("OPENAI_MODEL", "text-embedding-v3")
        self.openai_base_url = self._get_env_var("OPENAI_BASE_URL")

        # 在线表格API配置 (MCP模式下由agent动态传递)
        self.sheet_base_url = self._get_env_var("SHEET_BASE_URL", "https://docs.qq.com/openapi/spreadsheet/v3/files/")

        # 处理参数
        self.similarity_threshold = float(self._get_env_var("SIMILARITY_THRESHOLD", "0.7"))
        self.min_segment_length = int(self._get_env_var("MIN_SEGMENT_LENGTH", "50"))
        self.max_segment_length = int(self._get_env_var("MAX_SEGMENT_LENGTH", "1000"))

        # 日志配置
        self.log_level = self._get_env_var("LOG_LEVEL", "INFO")
        self.log_file_path = self._get_env_var("LOG_FILE_PATH", "logs/semantic_splitter.log")

        # 临时文件配置
        self.temp_base_dir = self._get_env_var("TEMP_BASE_DIR", "temp")
    
    def _get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """获取环境变量"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_openai_config(self) -> dict:
        """获取OpenAI配置"""
        config = {
            "api_key": self.openai_api_key,
            "model": self.openai_model
        }
        if self.openai_base_url:
            config["base_url"] = self.openai_base_url
        return config
    
    def get_sheet_headers(self, access_token: str, client_id: str, open_id: str) -> dict:
        """
        获取在线表格API请求头

        Args:
            access_token: 访问令牌
            client_id: 应用ID
            open_id: 开放平台用户ID

        Returns:
            请求头字典
        """
        return {
            "Access-Token": access_token,
            "Client-Id": client_id,
            "Open-Id": open_id,
            "Accept": "application/json"
        }


# 全局配置实例
config = Config()

def get_config() -> Config:
    """
    获取配置实例

    Returns:
        Config实例
    """
    return config
