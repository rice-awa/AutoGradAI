# AI英语作文自动批改系统

> 🎉 **最新更新 (2025-10-07)**: 现已支持任何OpenAI兼容的API服务，包括Ollama本地模型！详见[配置指南](#模型配置)

基于大型语言模型（LLM）的英语作文自动批改系统，为英语学习者提供专业、全面、详细的作文评价和改进建议。

## 项目简介

本项目是一个使用现代NLP技术实现的英语作文自动批改系统，通过整合深度语义分析，提供与专业英语教师相当的批改质量。系统对英语作文进行全方位评价，包括语法、拼写、用词、结构和内容深度，并给出详细的改进建议。

### 主要功能

- **作文评分**：提供15分制专业评分，附带详细评分理由
- **错误分析**：精确定位并分类作文中的拼写错误、语法错误和用词不当问题
- **亮点分析**：识别作文中使用的高级词汇和优秀表达
- **写作建议**：提供针对性的结构、表达和内容改进建议
- **同步/异步处理**：支持同步和异步批改模式，适应不同使用场景
- **流式响应**：实时返回批改结果，提升用户体验
- **多模型支持**：灵活支持DeepSeek、OpenAI及任何OpenAI兼容的API服务
- **统一配置管理**：通过JSON配置文件轻松管理服务器和模型设置

## 技术架构

- **前端**：Flask Web应用，响应式设计
- **后端**：Python + LangChain框架
- **配置管理**：JSON配置文件，支持灵活的模型和服务器配置
- **语言模型**：支持DeepSeek、OpenAI及任何OpenAI兼容的API服务
- **错误检测**：基于高级正则表达式和语义分析的错误检测系统
- **错误位置验证**：智能索引匹配系统，确保精确定位错误位置
- **类型提示**：使用Python类型注解提高代码可读性和可维护性

## 流程展示
- 初始界面提交作文
  
[![pVMrMYF.png](https://s21.ax1x.com/2025/07/07/pVMrMYF.png)](https://imgse.com/i/pVMrMYF)

- 等待批改，流式传输数据，实时返回批改结果

[![pVMr8yR.png](https://s21.ax1x.com/2025/07/07/pVMr8yR.png)](https://imgse.com/i/pVMr8yR)

- 批改完成

[![pVMrGO1.png](https://s21.ax1x.com/2025/07/07/pVMrGO1.png)](https://imgse.com/i/pVMrGO1)

- 鼠标悬停可查看错误，点击即可跳转到错误

[![pVMrQW4.png](https://s21.ax1x.com/2025/07/07/pVMrQW4.png)](https://imgse.com/i/pVMrQW4)
## 安装与使用

### 环境要求

- Python 3.8+
- 网络连接（用于API调用）
- OpenAI/DeepSeek API密钥

### 安装步骤

1. 克隆本仓库
   ```bash
   git clone https://github.com/rice-awa/AutoGradAI.git
   cd AutoGradAI
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 配置应用

   **方式一：使用配置文件（推荐）**

   复制示例配置文件并根据需要修改：
   ```bash
   cp config.example.json config.json
   ```

   编辑 `config.json`：
   ```json
   {
     "server": {
       "host": "0.0.0.0",
       "port": 5000,
       "debug": true
     },
     "model": {
       "provider": "deepseek",  // 可选: deepseek, openai, ollama, 或其他
       "deepseek": {
         "model_name": "deepseek-chat",
         "base_url": "https://api.deepseek.com",
         "temperature": 0.0,
         "request_timeout": 120
       }
     }
   }
   ```

4. 配置环境变量（API密钥）

   创建 `.env` 文件：
   ```bash
   # 根据使用的provider设置对应的API密钥

   # DeepSeek
   DEEPSEEK_API_KEY=your_deepseek_api_key_here

   # OpenAI
   OPENAI_API_KEY=your_openai_api_key_here

   # 自定义provider (可选)
   # 使用特定环境变量: {PROVIDER}_API_KEY
   OLLAMA_API_KEY=ollama

   # 或使用通用密钥 (作为后备)
   API_KEY=your_api_key_here
   ```

   或直接设置环境变量：
   ```bash
   # Linux/MacOS
   export DEEPSEEK_API_KEY=your_deepseek_api_key_here

   # Windows PowerShell
   $env:DEEPSEEK_API_KEY="your_deepseek_api_key_here"

   # Windows CMD
   set DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

### 模型配置

系统支持多种语言模型，通过 `config.json` 轻松切换：

#### 1. 使用预定义模型 (DeepSeek/OpenAI)

编辑 `config.json`，修改 `provider` 字段：

```json
{
  "model": {
    "provider": "deepseek"  // 或 "openai"
  }
}
```

#### 2. 使用自定义OpenAI兼容服务

系统支持任何OpenAI兼容的API服务，例如：

**Ollama (本地模型):**
```json
{
  "model": {
    "provider": "ollama",
    "ollama": {
      "model_name": "qwen2.5:14b",
      "base_url": "http://localhost:11434/v1",
      "temperature": 0.0,
      "request_timeout": 120
    }
  }
}
```
设置环境变量: `OLLAMA_API_KEY=ollama`

**OpenRouter (多模型路由):**
```json
{
  "model": {
    "provider": "openrouter",
    "openrouter": {
      "model_name": "anthropic/claude-3-sonnet",
      "base_url": "https://openrouter.ai/api/v1",
      "temperature": 0.0,
      "request_timeout": 120
    }
  }
}
```
设置环境变量: `OPENROUTER_API_KEY=sk-or-v1-...`

**任何自定义服务:**
```json
{
  "model": {
    "provider": "custom",
    "custom": {
      "model_name": "your-model-name",
      "base_url": "https://your-api.example.com/v1",
      "temperature": 0.0,
      "request_timeout": 120
    }
  }
}
```
设置环境变量: `CUSTOM_API_KEY=your-api-key` 或 `API_KEY=your-api-key`

#### 环境变量优先级
对于自定义provider，系统会按以下顺序查找API密钥：
1. `{PROVIDER}_API_KEY` (如 `OLLAMA_API_KEY`)
2. `API_KEY` (通用密钥，作为后备)

### 启动应用

```bash
python run.py
```

默认情况下，应用将在 http://localhost:5000 启动。

**命令行参数:**

```bash
# 自定义主机和端口
python run.py --host 127.0.0.1 --port 8080

# 关闭调试模式
python run.py --no-debug
```

配置文件中的设置会被命令行参数覆盖。

## 使用方法

### Web界面使用

1. 打开浏览器访问 http://localhost:5000
2. 在文本框中输入或粘贴需要批改的英语作文
3. 点击"提交"按钮
4. 等待批改结果显示（通常在60秒内完成）

### API调用

#### 提交作文（异步）

```bash
curl -X POST http://localhost:5000/api/submit \
  -H "Content-Type: application/json" \
  -d '{"essay":"Your English essay here..."}'
```

返回：
```json
{
  "task_id": "uuid-task-id",
  "status": "queued"
}
```

#### 查询批改状态

```bash
curl http://localhost:5000/api/status/<task_id>
```

返回：
```json
{
  "status": "completed",
  "result": {
    "评分": { ... },
    "错误分析": { ... },
    "亮点分析": { ... },
    "写作建议": { ... }
  }
}
```

## 项目结构

```
AutoGradAI/
├── app.py                 # Flask Web应用和API接口
├── main.py                # 核心批改逻辑和LLM调用
├── prompts.py             # LLM提示词和批改指令模板
├── logger.py              # 日志配置和管理系统
├── run.py                 # 应用启动脚本
├── config.json            # 配置文件 (用户创建)
├── config.example.json    # 配置文件示例
├── .env                   # 环境变量文件 (用户创建)
├── requirements.txt       # 项目依赖
├── static/                # 静态资源文件
│   ├── css/               # 样式表
│   ├── js/                # JavaScript脚本
│   └── img/               # 图片资源
├── templates/             # HTML模板
│   ├── index.html         # 主页面模板
│   ├── result.html        # 结果页面模板
│   └── layout.html        # 布局模板
├── logs/                  # 日志目录
├── tests/                 # 测试代码目录
└── claude_md/             # 文档目录
    └── fix/               # 修复文档
```

### 核心模块详解

- **main.py**：
  - 定义数据结构和类型（ErrorItem, TaskStatus等）
  - 实现ModelConfig类，支持多模型配置和OpenAI兼容服务
  - 提供错误分析验证 (ErrorAnalysisValidator)
  - 实现任务处理逻辑 (TaskHandler)
  - 提供同步、异步、流式处理API

- **run.py**：
  - 应用启动和初始化
  - 配置加载和验证
  - API密钥检查（支持自定义provider）
  - 命令行参数处理

- **prompts.py**：
  - 定义作文批改提示词模板
  - 配置错误模式和规则
  - 管理输出格式指导

- **logger.py**：
  - 配置日志记录系统
  - 实现日志过滤和格式化
  - 支持文件和控制台输出

- **app.py**：
  - 实现Flask Web界面
  - 定义RESTful API接口
  - 管理任务队列和状态跟踪

## 高级功能

### 错误位置验证

系统采用智能错误位置验证机制，通过以下方法确保精确定位：
- 精确文本匹配
- 模糊匹配算法
- 上下文相关性分析

### 异步处理支持

系统支持长文本的异步处理，避免Web请求超时：
- 任务队列管理
- 实时状态查询
- 结果缓存机制

### 流式响应支持
系统支持流式响应处理，实时展示批改结果：
- 实时数据传输
- 流式响应管理
- 自动错误恢复

### 多模型灵活切换
系统支持在不同LLM之间灵活切换：
- 统一的配置接口（config.json）
- 支持任何OpenAI兼容的API服务
- 灵活的API密钥管理
- 模块化模型配置

### OpenAI兼容服务支持
系统支持任何OpenAI兼容的API服务，包括：
- **Ollama**: 本地大模型部署
- **vLLM**: 高性能推理服务器
- **LocalAI**: 本地AI服务
- **OpenRouter**: 多模型路由服务
- 任何自定义的OpenAI兼容API

## 数据类型设计

系统采用TypedDict和类型注解，实现强类型的数据结构：

```python
class EssayFeedback(TypedDict):
    """作文反馈完整数据结构"""
    评分: EvaluationScore
    错误分析: ErrorAnalysis
    亮点分析: HighlightAnalysis
    写作建议: WritingSuggestion
```

## 未来计划

- ✅ 支持更多语言模型集成（已支持任何OpenAI兼容服务）
- ✅ 统一配置管理（已通过config.json实现）
- 添加批量处理功能
- 实现用户历史记录和进度跟踪
- 开发更精细的评分标准和错误分类
- 增加自定义提示词和评分标准
- 添加单元测试和集成测试
- 支持更多语言的作文批改

## 性能优化建议

- 使用异步API处理长文本
- 对于较短文本，使用同步API减少延迟
- 使用本地模型（如Ollama）降低API成本和延迟
- 在高负载情况下配置队列系统
- 定期清理日志文件

## 常见问题

### 如何切换到本地模型？

1. 安装并启动Ollama
2. 编辑 `config.json`，设置provider为"ollama"
3. 配置ollama的base_url和model_name
4. 设置环境变量 `OLLAMA_API_KEY=ollama`

### 如何使用自定义API服务？

1. 确保您的API服务支持OpenAI格式
2. 在 `config.json` 中添加自定义provider配置
3. 设置对应的环境变量（`{PROVIDER}_API_KEY` 或 `API_KEY`）

### 配置文件和命令行参数的优先级？

优先级从高到低：
1. 命令行参数
2. config.json配置文件
3. 默认值

### 支持哪些OpenAI兼容的服务？

理论上支持所有OpenAI兼容的API服务，包括但不限于：
- Ollama (本地)
- Qwen (OpenAI兼容)
- OpenRouter (路由服务)
- 任何自建的OpenAI兼容服务

## 许可证

[MIT](./LICENSE)

## 贡献指南

欢迎提交Issue和Pull Request，共同改进这个项目！

## 联系方式

[issues@rice-awa.top](issues@rice-awa.top)
