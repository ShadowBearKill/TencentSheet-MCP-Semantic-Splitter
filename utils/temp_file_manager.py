"""
临时文件管理模块
支持临时文件创建/读写/清理，使用UUID命名
"""
import os
import json
import tempfile
import uuid
import shutil
from typing import Optional, Any, Dict
from pathlib import Path


class TempFileManager:
    """临时文件管理类"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化临时文件管理器

        Args:
            base_dir: 基础目录，默认从环境变量TEMP_BASE_DIR读取或使用项目temp目录
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # 从环境变量读取或使用默认值
            env_base_dir = os.getenv("TEMP_BASE_DIR", "temp")
            self.base_dir = Path(env_base_dir) / "sheet-semantic-splitter"
        
        self.session_id = str(uuid.uuid4())
        self.session_dir = self.base_dir / self.session_id
        
        # 创建会话目录
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> str:
        """
        创建临时文件
        
        Args:
            content: 文件内容
            suffix: 文件后缀
        
        Returns:
            临时文件路径
        """
        file_id = str(uuid.uuid4())
        file_path = self.session_dir / f"{file_id}{suffix}"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    def create_temp_json(self, data: Dict[str, Any]) -> str:
        """
        创建临时JSON文件
        
        Args:
            data: JSON数据
        
        Returns:
            临时文件路径
        """
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return self.create_temp_file(content, ".json")
    
    def read_temp_file(self, file_path: str) -> str:
        """
        读取临时文件内容
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def read_temp_json(self, file_path: str) -> Dict[str, Any]:
        """
        读取临时JSON文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            JSON数据
        """
        content = self.read_temp_file(file_path)
        return json.loads(content)
    
    def write_temp_file(self, file_path: str, content: str) -> None:
        """
        写入临时文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def append_temp_file(self, file_path: str, content: str) -> None:
        """
        追加内容到临时文件
        
        Args:
            file_path: 文件路径
            content: 追加内容
        """
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def get_temp_dir(self) -> str:
        """
        获取当前会话的临时目录
        
        Returns:
            临时目录路径
        """
        return str(self.session_dir)
    
    def list_temp_files(self) -> list:
        """
        列出当前会话的所有临时文件
        
        Returns:
            文件路径列表
        """
        if not self.session_dir.exists():
            return []
        
        return [str(f) for f in self.session_dir.iterdir() if f.is_file()]
    
    def cleanup(self) -> None:
        """清理当前会话的所有临时文件"""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)
    
    def cleanup_all(self) -> None:
        """清理所有临时文件（包括其他会话）"""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动清理"""
        self.cleanup()


# 便捷函数
def create_temp_manager(base_dir: Optional[str] = None) -> TempFileManager:
    """
    创建临时文件管理器

    Args:
        base_dir: 基础目录

    Returns:
        TempFileManager实例
    """
    return TempFileManager(base_dir)

def get_temp_file_manager(base_dir: Optional[str] = None) -> TempFileManager:
    """
    获取临时文件管理器实例

    Args:
        base_dir: 基础目录

    Returns:
        TempFileManager实例
    """
    return TempFileManager(base_dir)
