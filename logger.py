# logger.py

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 创建日志目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class StreamUpdateFilter(logging.Filter):
    """
    过滤流式更新日志的过滤器，减少日志输出频率
    """
    def __init__(self, name=''):
        super().__init__(name)
        self.last_update_count = {}
        self.log_interval = 100  # 每50次更新记录一次日志

    def filter(self, record):
        # 如果不是流式更新日志，直接通过
        if "流式任务" not in record.getMessage() or "更新第" not in record.getMessage():
            return True
            
        # 提取任务ID和更新次数
        try:
            message = record.getMessage()
            task_id = message.split("流式任务")[1].split("更新第")[0].strip()
            update_count = int(message.split("更新第")[1].split("次")[0].strip())
            
            # 如果是第一次更新或者达到了记录间隔，则记录日志
            if task_id not in self.last_update_count or \
               update_count - self.last_update_count.get(task_id, 0) >= self.log_interval or \
               "完成" in message:
                self.last_update_count[task_id] = update_count
                return True
            return False
        except:
            # 解析失败，默认通过
            return True

def setup_logger(name):
    """
    配置并返回一个日志器。

    :param name: 日志器的名称
    :return: 配置好的日志器
    """
    # 创建一个日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除可能存在的处理器
    if logger.handlers:
        logger.handlers = []

    # 创建一个文件处理器，将日志写入文件
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 添加流式更新过滤器到控制台处理器
    stream_filter = StreamUpdateFilter()
    console_handler.addFilter(stream_filter)

    # 创建一个日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式器添加到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
