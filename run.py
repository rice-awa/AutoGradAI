#!/usr/bin/env python
"""
作文批改系统启动脚本
"""
import os
import sys
import argparse
from logger import setup_logger
from dotenv import load_dotenv

# 配置日志
logger = setup_logger("run")
# 加载 .env 文件
load_dotenv()  

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
        'flask', 'langchain', 'langchain_core', 'langchain_openai'
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
def check_api_key():
    """检查API密钥是否设置"""
    import json

    # 尝试从config.json读取配置
    config_path = os.path.join(os.getcwd(), 'config.json')
    provider = "deepseek"  # 默认值

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                provider = config.get('model', {}).get('provider', 'deepseek')
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}, 使用默认provider: deepseek")

    provider_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                provider_config = config.get('model', {}).get(provider, {})
        except Exception:
            provider_config = {}

    base_url = str(provider_config.get("base_url", ""))

    # 根据provider检查对应的API密钥
    if provider.lower() == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            logger.error("未设置DeepSeek API密钥")
            logger.info("请设置DEEPSEEK_API_KEY环境变量")
            return False
    elif provider.lower() == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("未设置OpenAI API密钥")
            logger.info("请设置OPENAI_API_KEY环境变量")
            return False
    else:
        # 对于其他provider，使用OpenAI兼容格式
        logger.warning(f"使用未预定义的provider: {provider}，将采用OpenAI兼容格式")
        # 尝试从环境变量获取API密钥，格式为 {PROVIDER}_API_KEY
        api_key_env = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            # 如果没有特定的环境变量，尝试使用通用的API_KEY
            api_key = os.getenv("API_KEY")
            if not api_key:
                # 兼容本地模型网关（例如Ollama）
                if "localhost" in base_url or "127.0.0.1" in base_url:
                    logger.info(f"{provider} 使用本地地址 {base_url}，允许使用占位密钥")
                    return True
                logger.error(f"未设置 {api_key_env} 或 API_KEY 环境变量")
                logger.info(f"请设置 {api_key_env} 或通用的 API_KEY 环境变量")
                return False
            logger.info(f"使用通用环境变量 API_KEY 作为 {provider} 的密钥")

    logger.info(f"使用模型提供商: {provider}")
    return True

def main():
    """主函数"""
    import json

    parser = argparse.ArgumentParser(description='作文批改系统启动脚本')
    parser.add_argument('--host', default=None, help='主机地址')
    parser.add_argument('--port', type=int, default=None, help='端口号')
    parser.add_argument('--no-debug', action='store_true', help='关闭调试模式')

    args = parser.parse_args()

    # 从config.json读取配置
    config_path = os.path.join(os.getcwd(), 'config.json')
    host = args.host or '0.0.0.0'
    port = args.port or 5000
    debug = not args.no_debug

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                server_config = config.get('server', {})
                # 命令行参数优先级高于配置文件
                if args.host is None:
                    host = server_config.get('host', '0.0.0.0')
                if args.port is None:
                    port = server_config.get('port', 5000)
                if not args.no_debug:
                    debug = server_config.get('debug', True)
            logger.info(f"从配置文件加载服务器设置: host={host}, port={port}, debug={debug}")
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}, 使用默认设置")

    # 创建日志目录
    create_log_dir()

    # 检查依赖项
    if not check_dependencies() or not check_api_key():
        sys.exit(1)

    run_app(host, port, debug)

if __name__ == '__main__':
    main() 
