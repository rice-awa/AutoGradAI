#!/usr/bin/env python
"""
作文批改系统启动脚本
"""
import os
import sys
import argparse
from logger import setup_logger

# 配置日志
logger = setup_logger("run")

def run_app(host='0.0.0.0', port=5000, debug=True):
    """启动Flask应用"""
    try:
        from app import app
        logger.info(f"启动Flask应用，地址: {host}:{port}, 调试模式: {debug}")
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"启动应用失败: {str(e)}")
        sys.exit(1)

def create_log_dir():
    """确保日志目录存在"""
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"创建日志目录: {log_dir}")
    return log_dir

def check_dependencies():
    """检查依赖项是否安装"""
    required_modules = [
        'flask', 'langchain_core', 'langchain_openai', 'langchain_deepseek'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"缺少以下依赖项: {', '.join(missing)}")
        logger.info("请运行: pip install -r requirements.txt")
        return False
    
    return True
def check_deepseek_api_key():
    """检查DeepSeek API密钥是否设置"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("未设置DeepSeek API密钥")
        logger.info("请设置DEEPSEEK_API_KEY环境变量")
        return False
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='作文批改系统启动脚本')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--no-debug', action='store_true', help='关闭调试模式')
    
    args = parser.parse_args()
    
    # 创建日志目录
    create_log_dir()
    
    # 检查依赖项
    if not check_dependencies() or not check_deepseek_api_key():
        sys.exit(1)
    
    run_app(args.host, args.port, not args.no_debug)

if __name__ == '__main__':
    main() 