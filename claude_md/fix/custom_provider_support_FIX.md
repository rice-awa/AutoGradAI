# 自定义Provider支持 - 更新说明

## 更新日期
2025-10-07

## 新增功能

系统现在支持任意自定义的OpenAI兼容provider,不再局限于预定义的deepseek和openai。

## 功能说明

### 1. OpenAI兼容格式

对于未预定义的provider(除deepseek和openai外),系统会自动使用**OpenAI兼容格式**进行API请求。这意味着您可以使用任何支持OpenAI API格式的服务,例如:

- **Ollama** (本地大模型服务)
- **vLLM** (推理服务器)
- **LocalAI** (本地AI服务)
- **OpenRouter** (模型路由服务)
- **任何自定义的OpenAI兼容API服务**

### 2. API密钥检查逻辑

对于自定义provider,系统会按以下优先级查找API密钥:

1. **特定环境变量**: `{PROVIDER}_API_KEY` (如 `OLLAMA_API_KEY`)
2. **通用环境变量**: `API_KEY`

如果两者都未设置,系统会报错并提示设置相应的环境变量。

### 3. 实现细节

#### run.py中的检查逻辑:
```python
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
            logger.error(f"未设置 {api_key_env} 或 API_KEY 环境变量")
            logger.info(f"请设置 {api_key_env} 或通用的 API_KEY 环境变量")
            return False
        logger.info(f"使用通用环境变量 API_KEY 作为 {provider} 的密钥")
```

#### main.py中的配置加载:
```python
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
            raise ValueError(f"环境变量 {api_key_env} 或 API_KEY 未设置")
        logger.info(f"使用通用环境变量 API_KEY 作为 {provider} 的密钥")
```

## 使用示例

### 示例1: 使用Ollama本地模型

**1. 配置config.json:**
```json
{
  "model": {
    "provider": "ollama",
    "ollama": {
      "model_name": "qwen2.5:14b",
      "base_url": "http://localhost:11434/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    }
  }
}
```

**2. 设置环境变量:**
```bash
# 方式1: 使用特定环境变量(推荐)
export OLLAMA_API_KEY="your-api-key"  # 或 "ollama" (Ollama通常不需要真实密钥)

# 方式2: 使用通用环境变量
export API_KEY="your-api-key"
```

**3. 启动应用:**
```bash
python run.py
```

### 示例2: 使用自定义API服务

**1. 配置config.json:**
```json
{
  "model": {
    "provider": "custom",
    "custom": {
      "model_name": "my-custom-model",
      "base_url": "https://my-api.example.com/v1",
      "temperature": 0.0,
      "max_tokens": 2000,
      "request_timeout": 180
    }
  }
}
```

**2. 设置环境变量:**
```bash
export CUSTOM_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 示例3: 使用OpenRouter

**1. 配置config.json:**
```json
{
  "model": {
    "provider": "openrouter",
    "openrouter": {
      "model_name": "anthropic/claude-3-sonnet",
      "base_url": "https://openrouter.ai/api/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    }
  }
}
```

**2. 设置环境变量:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxx"
```

## 配置文件示例

项目根目录下的`config.example.json`包含了所有支持的配置示例:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": true
  },
  "model": {
    "provider": "deepseek",
    "deepseek": {
      "model_name": "deepseek-chat",
      "base_url": "https://api.deepseek.com/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    },
    "openai": {
      "model_name": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    },
    "ollama": {
      "model_name": "qwen2.5:14b",
      "base_url": "http://localhost:11434/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    },
    "custom": {
      "model_name": "custom-model",
      "base_url": "http://your-api-endpoint/v1",
      "temperature": 0.0,
      "max_tokens": null,
      "request_timeout": 120
    }
  }
}
```

## 环境变量设置指南

### Windows (.env文件):
```env
# 预定义Provider
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 自定义Provider
OLLAMA_API_KEY=ollama
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# 通用密钥(可选,作为后备选项)
API_KEY=your-fallback-api-key
```

### Linux/Mac:
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
export OLLAMA_API_KEY="ollama"
export API_KEY="your-fallback-api-key"
```

## 测试

运行测试脚本验证配置:

```bash
# 测试基本配置
python test_config.py

# 测试自定义provider
python test_custom_provider.py
```

## 日志输出

使用自定义provider时,系统会输出相应的日志信息:

```
WARNING - 使用未预定义的provider: ollama，将采用OpenAI兼容格式
INFO - 使用通用环境变量 API_KEY 作为 ollama 的密钥
INFO - 使用模型提供商: ollama
```

## 支持的Provider类型

### 预定义Provider (原生支持):
- `deepseek`: DeepSeek官方API
- `openai`: OpenAI官方API

### 自定义Provider (OpenAI兼容):
- `ollama`: 本地Ollama服务
- `vllm`: vLLM推理服务器
- `localai`: LocalAI本地服务
- `openrouter`: OpenRouter模型路由
- 任何其他OpenAI兼容的API服务

## 兼容性说明

- ✅ 所有自定义provider自动使用OpenAI兼容格式
- ✅ 支持灵活的API密钥配置(特定或通用)
- ✅ 向后兼容,不影响现有配置
- ✅ 支持任意base_url和model_name配置

## 优势

1. **灵活性**: 支持任意OpenAI兼容的API服务
2. **本地部署**: 可以使用Ollama等本地模型服务
3. **成本控制**: 可以选择更经济的API服务
4. **隐私保护**: 支持完全本地化部署
5. **易用性**: 只需修改配置文件,无需修改代码

## 注意事项

1. 确保您的自定义provider支持OpenAI API格式
2. 正确配置base_url(必须包含完整的API端点路径)
3. model_name必须是provider支持的模型名称
4. 某些本地服务(如Ollama)可能不需要真实的API密钥,可以设置任意值
5. 如果遇到连接问题,检查base_url和网络配置

## 相关文件

- `config.json`: 主配置文件
- `config.example.json`: 配置示例文件
- `test_custom_provider.py`: 自定义provider测试脚本
- `run.py`: 启动脚本(包含API密钥检查)
- `main.py`: 核心逻辑(包含ModelConfig类)

## 总结

通过这次更新,系统现在支持任意OpenAI兼容的API服务,大大提高了灵活性和可扩展性。用户可以轻松切换到本地模型服务或其他第三方API,而无需修改任何代码。
