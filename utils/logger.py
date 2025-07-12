"""
日志管理模块
支持不同级别的日志输出，同时输出到控制台和文件
"""
import logging
import os
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理类"""
    
    def __init__(self, name: str = "semantic_splitter", 
                 level: str = "INFO", 
                 log_file: Optional[str] = None):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self.setup_handlers(log_file)
    
    def setup_handlers(self, log_file: Optional[str]):
        """设置日志处理器"""
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            # 使用Path处理路径，确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """调试级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """信息级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """严重错误级别日志"""
        self.logger.critical(message)

    def setLevel(self, level: str):
        """设置日志级别"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
        else:
            self.logger.warning(f"Unknown log level: {level}")

    def close(self):
        """关闭并移除所有handler，防止文件占用"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


def get_logger(name: str = "semantic_splitter",
               level: Optional[str] = None,
               log_file: Optional[str] = None) -> Logger:
    """
    获取日志器实例，支持从环境变量读取配置

    Args:
        name: 日志器名称
        level: 日志级别，默认从环境变量LOG_LEVEL读取
        log_file: 日志文件路径，默认从环境变量LOG_FILE_PATH读取

    Returns:
        Logger实例
    """
    # 从环境变量读取配置
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    if log_file is None:
        log_file = os.getenv("LOG_FILE_PATH", "logs/semantic_splitter.log")

    return Logger(name, level, log_file)


# 全局日志器实例
logger = get_logger()
