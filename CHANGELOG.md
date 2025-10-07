# 更新日志 (CHANGELOG)

## [v2.0.0] - 2025-10-07

### 🎉 重大更新

#### 新增功能
- ✅ **OpenAI兼容服务支持**: 现在支持任何OpenAI兼容的API服务
  - Ollama (本地大模型)
  - vLLM (推理服务器)
  - LocalAI (本地AI服务)
  - OpenRouter (多模型路由)
  - 任何自定义OpenAI兼容API

- ✅ **统一配置管理**: 通过JSON配置文件管理所有设置
  - `config.json`: 统一配置服务器和模型设置
  - `config.example.json`: 配置示例文件
  - 支持灵活的provider切换

- ✅ **灵活的API密钥管理**:
  - 支持provider特定密钥: `{PROVIDER}_API_KEY`
  - 支持通用密钥: `API_KEY` (作为后备)
  - 自动密钥查找和验证

#### Bug修复
- 🐛 修复了`check_deepseek_api_key()`只检查DeepSeek密钥的问题
  - 重命名为`check_api_key()`
  - 根据配置的provider动态检查对应的API密钥
  - 支持自定义provider的密钥检查

#### 改进
- 🔧 优化了模型初始化逻辑
  - 优先从`config.json`加载配置
  - 失败时自动降级到环境变量
  - 提供详细的日志信息

- 🔧 增强了服务器配置
  - 从配置文件读取host、port、debug设置
  - 命令行参数优先级最高
  - 支持灵活的运行时配置

- 📝 完善了文档
  - 更新了README.md
  - 添加了自定义provider使用指南
  - 提供了多个配置示例

### 📋 技术细节

#### 修改的文件
1. **config.json** (新增)
   - 统一的配置文件

2. **config.example.json** (新增)
   - 配置示例，包含多种provider示例

3. **run.py**
   - `check_deepseek_api_key()` → `check_api_key()`
   - 添加了自定义provider支持
   - 从配置文件读取服务器设置

4. **main.py**
   - `ModelConfig.from_config()`: 新增方法
   - 优化了模型初始化逻辑
   - 支持自定义provider的OpenAI兼容格式

5. **README.md** (更新)
   - 添加了配置指南
   - 更新了安装步骤
   - 添加了常见问题解答

#### 新增文件
- `test_config.py`: 基础配置测试脚本
- `test_custom_provider.py`: 自定义provider测试脚本
- `claude_md/fix/model_config_optimization_FIX.md`: 优化总结文档
- `claude_md/fix/custom_provider_support_FIX.md`: 自定义provider支持文档

### 🚀 使用示例

#### 使用Ollama本地模型
```json
{
  "model": {
    "provider": "ollama",
    "ollama": {
      "model_name": "qwen2.5:14b",
      "base_url": "http://localhost:11434/v1"
    }
  }
}
```

```bash
export OLLAMA_API_KEY="ollama"
python run.py
```

#### 使用OpenRouter
```json
{
  "model": {
    "provider": "openrouter",
    "openrouter": {
      "model_name": "anthropic/claude-3-sonnet",
      "base_url": "https://openrouter.ai/api/v1"
    }
  }
}
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
python run.py
```

### 📊 兼容性

- ✅ 向后兼容: 支持原有的环境变量配置方式
- ✅ API密钥安全: 仍然通过环境变量提供API密钥
- ✅ 灵活配置: 配置文件优先,命令行参数覆盖

### 🎯 优势

1. **成本优化**: 可使用Ollama等本地模型,完全免费
2. **隐私保护**: 支持完全本地化部署
3. **灵活性**: 轻松切换不同的模型服务
4. **易用性**: 只需修改配置文件,无需改代码

---

## [v1.0.0] - 之前版本

### 核心功能
- ✅ 英语作文自动批改
- ✅ 错误分析和定位
- ✅ 亮点识别
- ✅ 写作建议
- ✅ 支持DeepSeek和OpenAI模型
- ✅ 同步/异步/流式处理
- ✅ Web界面和API接口
