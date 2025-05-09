"""
主模块: 提供英语作文自动批改功能
包含模型配置、任务处理、错误分析等核心功能
"""
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator, Protocol, TypedDict, cast
import os
import json
import re
import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import datetime
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import AsyncCallbackHandler

from prompts import letter_correction_prompt, ERROR_PATTERNS
from logger import setup_logger

# 配置日志
logger = setup_logger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class EvaluationScore(TypedDict):
    """评分数据结构"""
    分数: str
    评分理由: str


class ErrorItem(TypedDict):
    """错误项数据结构"""
    错误文本: str
    错误位置: List[int]
    详细分析: str
    正确形式: str


class SpellingError(ErrorItem):
    """拼写错误数据结构"""
    正例: List[str]
    反例: List[str]


class GrammarError(ErrorItem):
    """语法错误数据结构"""
    语法规则: str
    正例: List[str]
    反例: List[str]


class WordChoiceError(ErrorItem):
    """用词错误数据结构"""
    建议用词: str
    正例: List[str]
    反例: List[str]


class HighlightItem(TypedDict):
    """亮点项数据结构"""
    位置: List[int]
    优秀之处: str


class VocabularyHighlight(HighlightItem):
    """词汇亮点数据结构"""
    词汇: str
    其他用法: List[str]


class ExpressionHighlight(HighlightItem):
    """表达亮点数据结构"""
    表达: str
    类似表达: List[str]


class WritingSuggestion(TypedDict):
    """写作建议数据结构"""
    结构建议: str
    表达建议: str
    内容建议: str
    实用技巧: List[str]


class ErrorAnalysis(TypedDict):
    """错误分析数据结构"""
    拼写错误: List[SpellingError]
    语法错误: List[GrammarError]
    用词不当: List[WordChoiceError]


class HighlightAnalysis(TypedDict):
    """亮点分析数据结构"""
    高级词汇: List[VocabularyHighlight]
    亮点表达: List[ExpressionHighlight]


class EssayFeedback(TypedDict):
    """作文反馈完整数据结构"""
    评分: EvaluationScore
    错误分析: ErrorAnalysis
    亮点分析: HighlightAnalysis
    写作建议: WritingSuggestion


class StreamResponse(TypedDict):
    """流式响应数据结构"""
    status: str
    partial_result: Optional[Dict[str, Any]]
    result: Optional[EssayFeedback]
    error: Optional[str]


@dataclass
class ModelConfig:
    """模型配置类"""
    api_key: str
    model_name: str
    base_url: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    request_timeout: int = 120

    @classmethod
    def from_env(cls, model_type: str = "deepseek") -> "ModelConfig":
        """
        从环境变量加载模型配置
        
        Args:
            model_type: 模型类型，支持 "deepseek" 或 "openai"
            
        Returns:
            模型配置对象
        
        Raises:
            ValueError: 当环境变量未设置或模型类型不支持时抛出异常
        """
        if model_type.lower() == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")
            
            return cls(
                api_key=api_key,
                model_name="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                temperature=0.0,
                request_timeout=120
            )
        elif model_type.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("环境变量 OPENAI_API_KEY 未设置")
            
            return cls(
                api_key=api_key,
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
                temperature=0.0,
                request_timeout=120
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def create_model(self) -> BaseChatModel:
        """
        根据配置创建对应的模型实例
        
        Returns:
            BaseChatModel: 大语言模型实例
        """
        if "deepseek" in self.model_name.lower():
            return ChatDeepSeek(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.request_timeout
            )
        else:
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.request_timeout
            )


class LogManager:
    """日志管理类，负责保存和管理LLM响应日志"""
    
    @staticmethod
    def log_llm_response(response: Dict[str, Any], log_type: str = "response") -> None:
        """
        将LLM的响应保存到日志文件中，并打印到主日志
        
        Args:
            response: LLM的响应数据
            log_type: 日志类型，用于文件名
        """
        try:
            # 确保日志目录存在
            log_dir = Path.cwd() / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            # 创建日志文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'llm_{log_type}_{timestamp}.json'
            
            # 以美化格式写入日志
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            
            # 打印响应体到主日志
            logger.info(f"LLM {log_type} 已保存到 {log_file}")
            logger.info(f"LLM 响应体: {json.dumps(response, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"保存LLM {log_type}失败: {str(e)}")


class ErrorAnalysisValidator:
    """
    错误分析验证器类，用于校验和修复错误位置索引
    """
    @staticmethod
    def validate_error_positions(essay: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证错误位置，确保索引准确
        
        Args:
            essay: 原始作文文本
            error_data: 错误分析数据
            
        Returns:
            修正后的错误分析数据
        """
        validated_data = error_data.copy()
        
        # 记录原始错误数据
        LogManager.log_llm_response(error_data, "original_response")
        
        # 遍历所有错误类别
        for category, errors in error_data.get('错误分析', {}).items():
            if not isinstance(errors, list):
                continue
                
            validated_errors = []
            
            for error in errors:
                # 确保错误文本字段存在
                if '错误文本' not in error and '错误位置' in error:
                    start, end = error['错误位置']
                    if 0 <= start < len(essay) and 0 <= end <= len(essay) and start < end:
                        error['错误文本'] = essay[start:end]
                        logger.info(f"添加错误文本: '{error['错误文本']}' 位置: {start}-{end}")
                    else:
                        logger.warning(f"索引超出范围: {start}-{end}, 文章长度: {len(essay)}")
                        continue
                
                # 使用正则表达式定位错误
                if '错误文本' in error:
                    error_text = error['错误文本']
                    original_positions = error.get('错误位置', [0, 0])
                    
                    try:
                        # 准备正则表达式模式 - 处理特殊字符
                        pattern = re.escape(error_text)
                        
                        # 查找所有匹配
                        matches = list(re.finditer(pattern, essay))
                        
                        if matches:
                            # 如果有多个匹配，选择最接近原始位置的匹配
                            if len(matches) > 1 and original_positions != [0, 0]:
                                original_start = original_positions[0]
                                closest_match = min(matches, key=lambda m: abs(m.start() - original_start))
                                error['错误位置'] = [closest_match.start(), closest_match.end()]
                                logger.info(f"找到多个匹配，选择最接近原始位置的匹配: '{error_text}' 在 {closest_match.start()}-{closest_match.end()} 处")
                            else:
                                # 使用第一个匹配
                                first_match = matches[0]
                                error['错误位置'] = [first_match.start(), first_match.end()]
                                logger.info(f"位置已更新: '{error_text}' 在 {first_match.start()}-{first_match.end()} 处找到")
                        else:
                            # 如果没有精确匹配，尝试模糊匹配
                            logger.warning(f"无法精确匹配错误文本: '{error_text}'，尝试模糊匹配")
                            
                            # 移除多余空格和标点符号进行模糊匹配
                            simplified_error = re.sub(r'[^\w\s]', '', error_text).lower().strip()
                            simplified_essay = essay.lower()
                            
                            fuzzy_match = re.search(simplified_error, simplified_essay)
                            if fuzzy_match:
                                # 从模糊匹配位置查找最接近的原始文本段落
                                approx_start = fuzzy_match.start()
                                context_start = max(0, approx_start - 20)
                                context_end = min(len(essay), approx_start + len(simplified_error) + 20)
                                context = essay[context_start:context_end]
                                
                                error['错误位置'] = [approx_start, approx_start + len(simplified_error)]
                                error['模糊匹配'] = True
                                logger.info(f"使用模糊匹配: '{error_text}' 可能在 {approx_start}-{approx_start + len(simplified_error)} 处，上下文: '{context}'")
                            else:
                                logger.warning(f"无法匹配错误文本: '{error_text}'，保留原始位置信息")
                    except Exception as e:
                        logger.error(f"处理错误文本时出错: {str(e)}")
                        continue
                
                validated_errors.append(error)
            
            if '错误分析' in validated_data:
                validated_data['错误分析'][category] = validated_errors
        
        # 记录验证后的错误数据
        LogManager.log_llm_response(validated_data, "validated_response")
        
        return validated_data


class PromptTemplate(Protocol):
    """定义PromptTemplate接口，用于类型提示"""
    def format(self, **kwargs: Any) -> str:
        ...


class Task:
    """任务基类，定义通用任务接口"""
    def __init__(self, prompt: str, input_variables: List[str], format_instruction: str) -> None:
        """
        初始化任务
        
        Args:
            prompt: 提示词模板
            input_variables: 输入变量列表
            format_instruction: 输出格式指导
        """
        self.prompt_template = prompt
        self.input_variables = input_variables
        self.format_instruction = format_instruction

    def get_prompt(self) -> PromptTemplate:
        """
        构造 PromptTemplate 对象
        
        Returns:
            格式化后的提示词模板
        """
        return PromptTemplate(
            template=self.prompt_template,
            input_variables=self.input_variables,
            partial_variables={"format_instructions": self.format_instruction}
        )


class StreamParser:
    """流式解析器，负责解析LLM流式输出的JSON内容"""
    
    @staticmethod
    def try_parse_json(content: str) -> Optional[Dict[str, Any]]:
        """
        尝试解析JSON内容
        
        Args:
            content: 待解析的JSON字符串
        
        Returns:
            解析成功返回字典，失败返回None
        """
        try:
            # 找到最外层JSON对象
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return None
    
    @staticmethod
    def extract_section(content: str, section_name: str) -> Optional[Dict[str, Any]]:
        """
        从内容中提取特定部分
        
        Args:
            content: 文本内容
            section_name: 部分名称
        
        Returns:
            提取的部分内容，失败返回None
        """
        try:
            section_match = re.search(f'"{section_name}"\\s*:\\s*(\\{{[^{{}}]*\\}}|\\[[^\\[\\]]*\\]|"[^"]*")', content)
            if not section_match:
                return None
                
            section_content = section_match.group(1)
            
            # 确保它是有效的JSON
            if section_content.startswith('{') and section_content.endswith('}'):
                return json.loads(section_content)
            elif section_content.startswith('[') and section_content.endswith(']'):
                return json.loads(section_content)
            elif section_content.startswith('"') and section_content.endswith('"'):
                # 处理字符串值
                return section_content.strip('"')
        except json.JSONDecodeError:
            pass
        return None


class TaskHandler:
    """任务处理器，负责任务的同步/异步执行和流式处理"""
    
    def __init__(self, model: BaseChatModel, parser: JsonOutputParser):
        """
        初始化任务处理器
        
        Args:
            model: 语言模型
            parser: JSON输出解析器
        """
        self.model = model
        self.parser = parser
        self.validator = ErrorAnalysisValidator()

    def process_task(self, task: Task, inputs: Dict[str, Any]) -> Any:
        """
        同步处理任务
        
        Args:
            task: 任务对象
            inputs: 输入数据字典
        
        Returns:
            任务处理结果
            
        Raises:
            Exception: 任务处理失败时抛出异常
        """
        prompt = task.get_prompt()
        chain = prompt | self.model | self.parser
        
        # 记录处理开始
        logger.info(f"开始处理任务: {task.__class__.__name__}")
        if 'essay' in inputs:
            logger.info(f"作文长度: {len(inputs['essay'])} 字符")
        
        try:
            # 执行LLM调用
            result = chain.invoke(inputs)
            
            # 记录原始结果
            logger.info("LLM调用成功")
            
            # 如果有essay输入，进行错误位置验证
            if 'essay' in inputs:
                result = self.validator.validate_error_positions(inputs['essay'], result)
                
            return result
        except Exception as e:
            logger.error(f"任务处理失败: {str(e)}")
            raise
    
    async def process_task_async(self, task: Task, inputs: Dict[str, Any]) -> Any:
        """
        异步处理任务
        
        Args:
            task: 任务对象
            inputs: 输入数据字典
        
        Returns:
            任务处理结果
            
        Raises:
            Exception: 任务处理失败时抛出异常
        """
        prompt = task.get_prompt()
        chain = prompt | self.model | self.parser
        
        # 记录处理开始
        logger.info(f"开始异步处理任务: {task.__class__.__name__}")
        if 'essay' in inputs:
            logger.info(f"作文长度: {len(inputs['essay'])} 字符")
        
        try:
            # 执行LLM调用
            result = await chain.ainvoke(inputs)
            
            # 记录原始结果
            logger.info("异步LLM调用成功")
            
            # 如果有essay输入，进行错误位置验证
            if 'essay' in inputs:
                result = self.validator.validate_error_positions(inputs['essay'], result)
                
            return result
        except Exception as e:
            logger.error(f"异步任务处理失败: {str(e)}")
            raise

    async def stream_task(self, task: Task, inputs: Dict[str, Any]) -> AsyncGenerator[StreamResponse, None]:
        """
        流式处理任务，支持实时返回部分结果
        
        Args:
            task: 任务对象
            inputs: 输入数据字典
        
        Yields:
            部分结果或完整结果
            
        Raises:
            Exception: 处理过程中发生错误时抛出异常
        """
        prompt = task.get_prompt()
        chain = prompt | self.model
        
        # 记录处理开始
        logger.info(f"开始流式处理任务: {task.__class__.__name__}")
        if 'essay' in inputs:
            logger.info(f"作文长度: {len(inputs['essay'])} 字符")
        
        collected_content = ""
        result_dict: Dict[str, Any] = {}
        stream_parser = StreamParser()
        
        try:
            # 流式处理LLM调用
            async for chunk in chain.astream(inputs):
                if isinstance(chunk, AIMessage):
                    content = chunk.content
                else:
                    content = str(chunk)
                    
                collected_content += content
                
                # 尝试解析已经收集到的内容
                try:
                    # 首先尝试解析完整JSON
                    full_result = stream_parser.try_parse_json(collected_content)
                    if full_result:
                        result_dict = full_result
                        yield {"status": TaskStatus.PROCESSING, "partial_result": result_dict, "result": None, "error": None}
                        continue
                    
                    # 如果完整解析失败，尝试提取部分键值对
                    # 查找已完成的部分（评分、错误分析、亮点分析、写作建议）
                    updated = False
                    for section in ["评分", "错误分析", "亮点分析", "写作建议"]:
                        section_content = stream_parser.extract_section(collected_content, section)
                        if section_content is not None:
                            result_dict[section] = section_content
                            updated = True
                    
                    # 如果有解析结果，返回部分结果
                    if updated and result_dict:
                        yield {"status": TaskStatus.PROCESSING, "partial_result": result_dict, "result": None, "error": None}
                    
                except Exception as e:
                    logger.error(f"解析流式内容时出错: {str(e)}")
            
            # 流处理完成后，尝试完整解析所有内容
            try:
                final_result = stream_parser.try_parse_json(collected_content)
                if final_result:
                    # 验证错误位置
                    if 'essay' in inputs:
                        final_result = self.validator.validate_error_positions(inputs['essay'], final_result)
                    
                    yield {"status": TaskStatus.COMPLETED, "partial_result": None, "result": cast(EssayFeedback, final_result), "error": None}
                else:
                    # 如果无法解析完整JSON，但有部分结果，也返回
                    if result_dict:
                        logger.warning("无法解析完整JSON，但有部分结果")
                        yield {"status": TaskStatus.COMPLETED, "partial_result": None, "result": cast(EssayFeedback, result_dict), "error": None}
                    else:
                        raise ValueError("无法解析LLM响应")
            except Exception as e:
                logger.error(f"解析最终流式结果时出错: {str(e)}")
                yield {"status": TaskStatus.ERROR, "partial_result": None, "result": None, "error": str(e)}
                
        except Exception as e:
            logger.error(f"流式任务处理失败: {str(e)}")
            yield {"status": TaskStatus.ERROR, "partial_result": None, "result": None, "error": str(e)}


# 初始化模型和解析器
try:
    model_config = ModelConfig.from_env("deepseek")
    model = model_config.create_model()
    parser = JsonOutputParser()
    task_handler = TaskHandler(model, parser)
except Exception as e:
    logger.critical(f"初始化模型失败: {str(e)}")
    raise

# 具体任务定义
task_letter = Task(
    prompt=letter_correction_prompt,
    input_variables=["essay"],
    format_instruction="""
输出格式为如下 json 格式，请确保包含所有必需字段：
{
"评分": {"分数": "xx"},
"错误分析": {...},
"亮点分析": {...},
"写作建议": "xxxxxxxx"
}
"""
)


# 任务处理函数
def handler_letter_correct(essay: str) -> Dict[str, Any]:
    """
    处理书信作文批改任务
    
    Args:
        essay: 待批改文章
    
    Returns:
        Json格式输出(py字典)
        
    Raises:
        Exception: 处理失败时抛出异常
    """
    if not essay or not isinstance(essay, str):
        raise ValueError("作文内容不能为空且必须是字符串")
        
    logger.info(f"收到作文批改请求，作文长度: {len(essay)} 字符")
    
    try:
        result = task_handler.process_task(task_letter, {"essay": essay})
        logger.info(f"作文批改完成，结果: {json.dumps(result, ensure_ascii=False)}")
        return result
    except Exception as e:
        logger.error(f"作文批改失败: {str(e)}")
        raise


async def handler_letter_correct_async(essay: str) -> Dict[str, Any]:
    """
    异步处理书信作文批改任务
    
    Args:
        essay: 待批改文章
    
    Returns:
        Json格式输出(py字典)
        
    Raises:
        ValueError: 当输入无效时抛出
        Exception: 处理失败时抛出异常
    """
    if not essay or not isinstance(essay, str):
        raise ValueError("作文内容不能为空且必须是字符串")
        
    logger.info(f"收到异步作文批改请求，作文长度: {len(essay)} 字符")
    
    try:
        result = await task_handler.process_task_async(task_letter, {"essay": essay})
        logger.info(f"异步作文批改完成，结果: {json.dumps(result, ensure_ascii=False)}")
        return result
    except Exception as e:
        logger.error(f"异步作文批改失败: {str(e)}")
        raise


async def handler_letter_correct_stream(essay: str) -> AsyncGenerator[StreamResponse, None]:
    """
    流式处理书信作文批改任务
    
    Args:
        essay: 待批改文章
    
    Yields:
        部分结果或完整结果
        
    Raises:
        ValueError: 当输入无效时抛出
    """
    if not essay or not isinstance(essay, str):
        raise ValueError("作文内容不能为空且必须是字符串")
        
    logger.info(f"收到流式作文批改请求，作文长度: {len(essay)} 字符")
    
    try:
        async for part_result in task_handler.stream_task(task_letter, {"essay": essay}):
            yield part_result
        
        logger.info("流式作文批改完成")
    except Exception as e:
        logger.error(f"流式作文批改失败: {str(e)}")
        yield {"status": TaskStatus.ERROR, "partial_result": None, "result": None, "error": str(e)}

