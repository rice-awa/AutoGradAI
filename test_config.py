#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试配置加载功能"""
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

# 测试1: 读取config.json
print("=" * 50)
print("测试1: 读取config.json配置文件")
print("=" * 50)

try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"✓ 成功读取config.json")
    print(f"  服务器配置: host={config['server']['host']}, port={config['server']['port']}")
    print(f"  模型提供商: {config['model']['provider']}")

    provider = config['model']['provider']
    provider_config = config['model'][provider]
    print(f"  {provider}配置:")
    print(f"    - model_name: {provider_config['model_name']}")
    print(f"    - base_url: {provider_config['base_url']}")
    print(f"    - temperature: {provider_config['temperature']}")
except Exception as e:
    print(f"✗ 读取配置文件失败: {e}")

# 测试2: 检查API密钥
print("\n" + "=" * 50)
print("测试2: 检查环境变量中的API密钥")
print("=" * 50)

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if deepseek_key:
    print(f"✓ DEEPSEEK_API_KEY 已设置 (长度: {len(deepseek_key)})")
else:
    print("✗ DEEPSEEK_API_KEY 未设置")

if openai_key:
    print(f"✓ OPENAI_API_KEY 已设置 (长度: {len(openai_key)})")
else:
    print("✗ OPENAI_API_KEY 未设置")

# 测试3: 模拟check_api_key函数
print("\n" + "=" * 50)
print("测试3: 模拟API密钥检查逻辑")
print("=" * 50)

try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    provider = config.get('model', {}).get('provider', 'deepseek')

    print(f"从配置文件读取到provider: {provider}")

    if provider.lower() == "deepseek":
        if os.getenv("DEEPSEEK_API_KEY"):
            print(f"✓ DeepSeek API密钥检查通过")
        else:
            print(f"✗ DeepSeek API密钥未设置")
    elif provider.lower() == "openai":
        if os.getenv("OPENAI_API_KEY"):
            print(f"✓ OpenAI API密钥检查通过")
        else:
            print(f"✗ OpenAI API密钥未设置")
    else:
        print(f"✗ 不支持的provider: {provider}")

except Exception as e:
    print(f"✗ 检查失败: {e}")

print("\n" + "=" * 50)
print("测试完成")
print("=" * 50)
