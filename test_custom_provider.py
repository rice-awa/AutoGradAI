#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试自定义provider功能"""
import os
import json
import sys
from dotenv import load_dotenv

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# 加载环境变量
load_dotenv()

def test_custom_provider():
    """测试自定义provider的API密钥检查逻辑"""
    print("=" * 60)
    print("测试自定义Provider功能")
    print("=" * 60)

    # 模拟配置文件
    test_providers = [
        {"name": "deepseek", "env_key": "DEEPSEEK_API_KEY"},
        {"name": "openai", "env_key": "OPENAI_API_KEY"},
        {"name": "ollama", "env_key": "OLLAMA_API_KEY"},
        {"name": "custom", "env_key": "CUSTOM_API_KEY"},
    ]

    for provider_info in test_providers:
        provider = provider_info["name"]
        print(f"\n测试Provider: {provider}")
        print("-" * 60)

        # 检查API密钥逻辑
        if provider.lower() == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                print(f"✓ {provider} - 使用预定义配置")
                print(f"  环境变量: DEEPSEEK_API_KEY")
                print(f"  密钥长度: {len(api_key)}")
            else:
                print(f"✗ {provider} - DEEPSEEK_API_KEY 未设置")

        elif provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print(f"✓ {provider} - 使用预定义配置")
                print(f"  环境变量: OPENAI_API_KEY")
                print(f"  密钥长度: {len(api_key)}")
            else:
                print(f"✗ {provider} - OPENAI_API_KEY 未设置")

        else:
            # 自定义provider逻辑
            print(f"⚠ {provider} - 使用OpenAI兼容格式")

            # 尝试特定环境变量
            api_key_env = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_env)

            if api_key:
                print(f"✓ 找到特定环境变量: {api_key_env}")
                print(f"  密钥长度: {len(api_key)}")
            else:
                print(f"✗ 未找到特定环境变量: {api_key_env}")

                # 尝试通用API_KEY
                api_key = os.getenv("API_KEY")
                if api_key:
                    print(f"✓ 使用通用环境变量: API_KEY")
                    print(f"  密钥长度: {len(api_key)}")
                else:
                    print(f"✗ 未找到通用环境变量: API_KEY")
                    print(f"  建议: 设置 {api_key_env} 或 API_KEY 环境变量")

def test_config_example():
    """测试示例配置文件"""
    print("\n" + "=" * 60)
    print("示例配置文件内容")
    print("=" * 60)

    config_file = "config.example.json"

    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"\n支持的Provider列表:")
        providers = [k for k in config['model'].keys() if k != 'provider']
        for provider in providers:
            provider_config = config['model'][provider]
            print(f"\n  • {provider}:")
            print(f"    - model_name: {provider_config['model_name']}")
            print(f"    - base_url: {provider_config['base_url']}")

        print(f"\n当前配置的Provider: {config['model']['provider']}")
    else:
        print(f"✗ 示例配置文件 {config_file} 不存在")

def show_usage_guide():
    """显示使用指南"""
    print("\n" + "=" * 60)
    print("使用指南")
    print("=" * 60)

    print("\n1. 使用预定义Provider (deepseek/openai):")
    print("   - 修改config.json中的provider为 'deepseek' 或 'openai'")
    print("   - 设置对应的环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

    print("\n2. 使用自定义Provider (如ollama, custom等):")
    print("   - 在config.json中添加provider配置:")
    print("     {")
    print('       "ollama": {')
    print('         "model_name": "qwen2.5:14b",')
    print('         "base_url": "http://localhost:11434/v1",')
    print('         "temperature": 0.0,')
    print('         "max_tokens": null,')
    print('         "request_timeout": 120')
    print("       }")
    print("     }")
    print("   - 修改config.json中的provider为 'ollama'")
    print("   - 设置环境变量 OLLAMA_API_KEY 或通用的 API_KEY")

    print("\n3. 环境变量优先级:")
    print("   - 对于自定义provider,系统会依次查找:")
    print("     1) {PROVIDER}_API_KEY (如 OLLAMA_API_KEY)")
    print("     2) API_KEY (通用密钥)")

    print("\n4. 所有自定义provider都将使用OpenAI兼容格式请求")

if __name__ == "__main__":
    test_custom_provider()
    test_config_example()
    show_usage_guide()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
