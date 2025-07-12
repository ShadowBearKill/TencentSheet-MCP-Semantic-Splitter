"""
OpenAI客户端封装模块
封装OpenAI API调用，支持文本嵌入和语言模型调用，包含错误处理和重试机制
"""
import time
from typing import List, Dict, Any, Optional, Union
import openai
from openai import OpenAI
from utils.logger import logger
from config.params import config


class OpenAIClientError(Exception):
    pass


class OpenAIClient:
    """OpenAI客户端类"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        初始化OpenAI客户端
        
        Args:
            api_key: API密钥，默认从配置读取
            base_url: API基础URL，默认从配置读取
            model: 默认模型名称，默认从配置读取
        """
        # 获取配置
        openai_config = config.get_openai_config()
        
        self.api_key = api_key or openai_config.get("api_key")
        self.base_url = base_url or openai_config.get("base_url")
        self.model = model or openai_config.get("model", "text-embedding-3-small")
        
        if not self.api_key:
            raise OpenAIClientError("OpenAI API key is required")
        
        # 初始化客户端
        client_kwargs = {"api_key": self.api_key, 'base_url': self.base_url}
        
        self.client = OpenAI(**client_kwargs)
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 1.0  # 秒
        self.backoff_factor = 2.0
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def embedding(self, text: str, 
                        model: Optional[str] = None) -> List[float]:
        """
        创建单个文本嵌入
        
        Args:
            text: 要嵌入的文本
            model: 嵌入模型名称，默认使用初始化时的模型
            
        Returns:
            嵌入向量
        """
        model = model or self.model
        texts = [text]
        logger.info(f"Creating embeddings for {len(texts)} text(s) using model: {model}")
        try:
            # 调用API
            response = self.retry(
                lambda: self.client.embeddings.create(
                    input=texts,
                    model=model
                )
            )
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings[0]

            
        except Exception as e:
            error_msg = f"Failed to create embeddings: {e}"
            logger.error(error_msg)
            raise OpenAIClientError(error_msg)
        
    def embeddings(self, texts: List[str], 
                        model: Optional[str] = None) -> List[List[float]]:
        """
        创建多个文本的嵌入
        
        Args:
            texts: 要嵌入的文本列表
            model: 嵌入模型名称，默认使用初始化时的模型
            
        Returns:
            嵌入向量列表
        """
        model = model or self.model
        logger.info(f"Creating embeddings for {len(texts)} text(s) using model: {model}")
        try:
            # 调用API
            response = self.retry(
                lambda: self.client.embeddings.create(
                    input=texts,
                    model=model
                )
            )
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            error_msg = f"Failed to create embeddings: {e}"
            logger.error(error_msg)
            raise OpenAIClientError(error_msg)
    
    def retry(self, func, *args, **kwargs):
        """
        带重试机制的API调用
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数返回值
        """
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except openai.AuthenticationError as e:
                raise OpenAIClientError(f"Authentication error: {e}")
            
            except openai.BadRequestError as e:
                raise OpenAIClientError(f"Bad request error: {e}")
            
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(f"Unexpected error, retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue
                else:
                    raise OpenAIClientError(f"Unexpected error after {self.max_retries} retries: {e}")
        
        raise OpenAIClientError(f"All retries failed. Last exception: {last_exception}")
    

# 全局实例
openai_client = None

def get_openai_client() -> OpenAIClient:
    global openai_client
    if openai_client is None:
        openai_client = OpenAIClient()
    return openai_client
