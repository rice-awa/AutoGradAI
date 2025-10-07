# 模型配置优化与Bug修复总结

## 修复日期
2025-10-07

## 问题描述

### 1. API密钥检查Bug
- **问题**: `run.py`中的`check_deepseek_api_key()`函数只检查DeepSeek的API密钥，如果用户配置了OpenAI则无法通过检查
- **影响**: 当用户想使用OpenAI作为模型提供商时，启动脚本会因为缺少DeepSeek API密钥而失败

### 2. 模型配置硬编码
- **问题**: `main.py:1023`处硬编码了模型提供商为"deepseek"，无法灵活切换
- **影响**: 切换模型提供商需要修改代码，不够灵活

### 3. 缺少统一配置管理
- **问题**: 服务器配置(host, port)和模型配置分散在不同位置，没有统一的配置文件
- **影响**: 配置管理混乱，修改配置不方便

## 解决方案

### 1. 创建JSON配置文件(`config.json`)

创建了统一的配置文件，包含以下配置项:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": true
  },
  "model": {
    "provider": "deepseek",  // 或 "openai"
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
    }
  }
}
```

**优点**:
- 所有配置集中管理
- 可以轻松切换模型提供商(只需修改`provider`字段)
- 支持自定义每个提供商的详细配置

### 2. 修复API密钥检查逻辑(`run.py`)

**修改前**:
```python
def check_deepseek_api_key():
    """检查DeepSeek API密钥是否设置"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("未设置DeepSeek API密钥")
        logger.info("请设置DEEPSEEK_API_KEY环境变量")
        return False
    return True
```

**修改后**:
```python
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
        logger.error(f"不支持的provider类型: {provider}")
        return False

    logger.info(f"使用模型提供商: {provider}")
    return True
```

**改进**:
- ✅ 根据配置文件中的`provider`动态检查对应的API密钥
- ✅ 支持DeepSeek和OpenAI两种提供商
- ✅ 提供友好的错误提示
- ✅ 兼容配置文件不存在的情况(使用默认值)

### 3. 优化服务器配置加载(`run.py`)

修改了`main()`函数,支持从`config.json`读取服务器配置:

```python
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

    # ...
    run_app(host, port, debug)
```

**优先级**:
1. 命令行参数 (最高优先级)
2. 配置文件
3. 默认值 (最低优先级)

### 4. 扩展ModelConfig类(`main.py`)

添加了`from_config()`类方法,支持从JSON配置文件加载模型配置:

```python
@classmethod
def from_config(cls, config_path: str = "config.json") -> "ModelConfig":
    """
    从JSON配置文件加载模型配置

    Args:
        config_path: 配置文件路径，默认为 "config.json"

    Returns:
        ModelConfig: 模型配置对象

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出异常
        ValueError: 当配置文件格式错误或缺少必要字段时抛出异常
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model_config = config.get('model', {})
        provider = model_config.get('provider', 'deepseek').lower()

        if provider not in model_config:
            raise ValueError(f"配置文件中缺少 {provider} 的配置信息")

        provider_config = model_config[provider]

        # 根据provider获取对应的API密钥
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("环境变量 OPENAI_API_KEY 未设置")
        else:
            raise ValueError(f"不支持的模型提供商: {provider}")

        return cls(
            api_key=api_key,
            model_name=provider_config.get('model_name'),
            base_url=provider_config.get('base_url'),
            temperature=provider_config.get('temperature', 0.0),
            max_tokens=provider_config.get('max_tokens'),
            request_timeout=provider_config.get('request_timeout', 120)
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except KeyError as e:
        raise ValueError(f"配置文件缺少必要字段: {e}")
```

### 5. 优化模型初始化逻辑(`main.py`)

**修改前**:
```python
# 初始化模型配置和模型
model_config = ModelConfig.from_env("deepseek")
model = model_config.create_model()
```

**修改后**:
```python
# 初始化模型配置和模型
# 优先尝试从config.json加载配置，如果失败则使用环境变量
try:
    model_config = ModelConfig.from_config("config.json")
    logger.info("从config.json加载模型配置成功")
except (FileNotFoundError, ValueError) as e:
    logger.warning(f"从config.json加载配置失败: {e}, 尝试从环境变量加载")
    # 回退到环境变量加载，默认使用deepseek
    model_config = ModelConfig.from_env("deepseek")
    logger.info("从环境变量加载模型配置成功")

model = model_config.create_model()
logger.info(f"使用模型: {model_config.model_name}")
```

**改进**:
- ✅ 优先从配置文件加载
- ✅ 配置文件不存在时自动降级到环境变量
- ✅ 提供详细的日志信息

## 测试结果

创建了`test_config.py`测试脚本,验证了以下功能:

### 测试1: DeepSeek配置
```
模型提供商: deepseek
✓ DEEPSEEK_API_KEY 已设置
✓ DeepSeek API密钥检查通过
```

### 测试2: OpenAI配置
修改`config.json`中的`provider`为`"openai"`:
```
模型提供商: openai
✓ OPENAI_API_KEY 已设置
✓ OpenAI API密钥检查通过
```

### 测试3: 服务器配置
```
服务器配置: host=0.0.0.0, port=5000, debug=true
```

## 使用方法

### 1. 切换模型提供商

编辑`config.json`文件,修改`provider`字段:

```json
{
  "model": {
    "provider": "openai"  // 切换到OpenAI
  }
}
```

或

```json
{
  "model": {
    "provider": "deepseek"  // 切换到DeepSeek
  }
}
```

### 2. 修改服务器配置

编辑`config.json`文件:

```json
{
  "server": {
    "host": "127.0.0.1",  // 修改监听地址
    "port": 8080,         // 修改端口
    "debug": false        // 关闭调试模式
  }
}
```

### 3. 自定义模型参数

编辑`config.json`中对应提供商的配置:

```json
{
  "model": {
    "openai": {
      "model_name": "gpt-4-turbo",  // 使用不同的模型
      "temperature": 0.7,            // 调整温度参数
      "max_tokens": 2000,            // 设置最大token数
      "request_timeout": 180         // 调整超时时间
    }
  }
}
```

## 修改文件清单

1. **config.json** (新增)
   - 统一的配置文件

2. **run.py** (修改)
   - `check_deepseek_api_key()` → `check_api_key()`: 修复API密钥检查bug
   - `main()`: 添加从配置文件读取服务器配置的逻辑

3. **main.py** (修改)
   - `ModelConfig.from_config()`: 新增从JSON配置文件加载的类方法
   - 模型初始化逻辑: 优先从配置文件加载,失败则降级到环境变量

4. **test_config.py** (新增)
   - 配置加载测试脚本

## 兼容性说明

- ✅ 向后兼容: 如果没有`config.json`,系统会自动降级到原有的环境变量加载方式
- ✅ API密钥仍然通过环境变量提供(安全性考虑)
- ✅ 命令行参数优先级最高,可以覆盖配置文件设置

## 总结

本次修复实现了:

1. ✅ **修复了API密钥检查bug**: 支持根据配置的provider动态检查对应的API密钥
2. ✅ **实现了统一配置管理**: 通过JSON配置文件管理服务器和模型配置
3. ✅ **优化了模型切换逻辑**: 只需修改配置文件即可切换模型提供商
4. ✅ **提高了系统灵活性**: 支持自定义各种参数,无需修改代码
5. ✅ **保持了向后兼容**: 配置文件不存在时自动降级到环境变量方式

通过这些改进,系统的配置管理更加清晰、灵活和易用。
