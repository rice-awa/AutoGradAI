# AI英语作文自动批改系统

基于大型语言模型（LLM）的英语作文自动批改系统，为英语学习者提供专业、全面、详细的作文评价和改进建议。

## 项目简介

本项目是一个使用现代NLP技术实现的英语作文自动批改系统，通过整合深度语义分析，提供与专业英语教师相当的批改质量。系统对英语作文进行全方位评价，包括语法、拼写、用词、结构和内容深度，并给出详细的改进建议。

### 主要功能

- **作文评分**：提供15分制专业评分，附带详细评分理由
- **错误分析**：精确定位并分类作文中的拼写错误、语法错误和用词不当问题
- **亮点分析**：识别作文中使用的高级词汇和优秀表达
- **写作建议**：提供针对性的结构、表达和内容改进建议
- **同步/异步处理**：支持同步和异步批改模式，适应不同使用场景
- **多模型支持**：灵活支持DeepSeek和OpenAI等多种大语言模型

## 技术架构

- **前端**：Flask Web应用，响应式设计
- **后端**：Python + LangChain框架
- **语言模型**：支持DeepSeek和OpenAI API
- **错误检测**：基于高级正则表达式和语义分析的错误检测系统
- **错误位置验证**：智能索引匹配系统，确保精确定位错误位置
- **类型提示**：使用Python类型注解提高代码可读性和可维护性

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

3. 配置环境变量
   ```bash
   # DeepSeek模型配置 (默认)
   # Linux/MacOS
   export DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # Windows PowerShell
   $env:DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   
   # Windows CMD
   set DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # OpenAI模型配置 (可选)
   # Linux/MacOS
   export OPENAI_API_KEY=your_openai_api_key_here
   
   # Windows PowerShell
   $env:OPENAI_API_KEY="your_openai_api_key_here"
   
   # Windows CMD
   set OPENAI_API_KEY=your_openai_api_key_here

   # 使用 `.env` 文件
   - 在项目根目录下创建`.env`文件
   - 编辑`.env`文件:`DEEPSEEK_API_KEY="your_deepseek_api_key_here"`

   ```

### 切换使用的模型

要切换使用的语言模型，需修改main.py中的模型初始化部分：

```python
# 使用DeepSeek模型 (默认)
model_config = ModelConfig.from_env("deepseek")

# 或使用OpenAI模型
# model_config = ModelConfig.from_env("openai")
```

### 启动应用

```bash
python run.py
```

默认情况下，应用将在 http://localhost:5000 启动。

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
└── tests/                 # 测试代码目录
```

### 核心模块详解

- **main.py**：
  - 定义数据结构和类型（ErrorItem, TaskStatus等）
  - 实现ModelConfig类，支持多模型配置
  - 提供错误分析验证 (ErrorAnalysisValidator)
  - 实现任务处理逻辑 (TaskHandler)
  - 提供同步、异步、流式处理API

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
- 自动检查机制
- 流式响应管理

### 多模型灵活切换
系统支持在不同LLM之间灵活切换：
- 模块化模型配置
- 统一的模型接口
- 环境变量驱动的配置管理

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

- 支持更多语言模型集成（如Claude、Gemini等）
- 添加批量处理功能
- 实现用户历史记录和进度跟踪
- 开发更精细的评分标准和错误分类
- 增加自定义提示词和评分标准
- 添加单元测试和集成测试

## 性能优化建议

- 使用异步API处理长文本
- 对于较短文本，使用同步API减少延迟
- 在高负载情况下配置队列系统
- 定期清理日志文件

## 许可证

[MIT](./LICENSE)

## 贡献指南

欢迎提交Issue和Pull Request，共同改进这个项目！

## 联系方式

[issues@rice-awa.top](issues@rice-awa.top)