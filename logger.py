# logger.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认为INFO
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
        
    # 设置日志级别
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制台处理器，设置为WARNING级别，减少控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    
    # 确保日志目录存在
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建文件处理器，保留所有日志级别
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 创建一个过滤器，用于过滤流式处理中的频繁日志
class StreamLogFilter(logging.Filter):
    """
    过滤流式处理中的频繁日志
    """
    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.last_message: Optional[str] = None
        self.repeat_count: int = 0
        self.max_repeats: int = 3  # 最多显示相同消息的次数
        
    def filter(self, record: logging.LogRecord) -> bool:
        # 如果是流式处理相关的日志，进行过滤
        if "实时" in record.getMessage() or "流式" in record.getMessage():
            # 如果是相同的消息，增加计数
            if self.last_message == record.getMessage():
                self.repeat_count += 1
                # 如果超过最大重复次数，不记录
                if self.repeat_count > self.max_repeats:
                    return False
            else:
                # 如果是新消息，重置计数
                self.last_message = record.getMessage()
                self.repeat_count = 1
                
        return True

# 添加全局过滤器
def add_stream_filter(logger: logging.Logger) -> None:
    """
    为日志记录器添加流式日志过滤器
    
    Args:
        logger: 要添加过滤器的日志记录器
    """
    stream_filter = StreamLogFilter()
    logger.addFilter(stream_filter)
