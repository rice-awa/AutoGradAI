from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
import os
import json
import re
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from prompts import letter_correction_prompt, ERROR_PATTERNS
from logger import setup_logger

# 配置日志
logger = setup_logger(__name__)
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


# deepseek 配置
class deepseek_config:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    LLM_CONFIG = {
        "model": "deepseek-chat",
        "api_key": API_KEY,
        "base_url": "https://api.deepseek.com/v1",
    }

# 保存LLM响应到日志文件的函数
def log_llm_response(response: Dict[str, Any], log_type: str = "response") -> None:
    """
    将LLM的响应保存到日志文件中，并打印到主日志
    
    :param response: LLM的响应数据
    :param log_type: 日志类型，用于文件名
    """
    try:
        # 确保日志目录存在
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建日志文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'llm_{log_type}_{timestamp}.json')
        
        # 以美化格式写入日志
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # 打印响应体到主日志
        logger.info(f"LLM {log_type} 已保存到 {log_file}")
        logger.info(f"LLM 响应体: {json.dumps(response, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"保存LLM {log_type}失败: {str(e)}")

# 错误分析验证器类
class ErrorAnalysisValidator:
    """
    错误分析验证器类，用于校验和修复错误位置索引
    """
    @staticmethod
    def validate_error_positions(essay: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证错误位置，确保索引准确
        
        :param essay: 原始作文文本
        :param error_data: 错误分析数据
        :return: 修正后的错误分析数据
        """
        validated_data = error_data.copy()
        
        # 记录原始错误数据
        log_llm_response(error_data, "original_response")
        
        # 遍历所有错误类别
        for category, errors in error_data['错误分析'].items():
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
            
            validated_data['错误分析'][category] = validated_errors
        
        # 记录验证后的错误数据
        log_llm_response(validated_data, "validated_response")
        
        return validated_data

# 通用任务类
class BaseTask:
    def __init__(self, prompt: str, input_variables: List[str], format_instruction: str) -> None:
        self.prompt_template = prompt
        self.input_variables = input_variables
        self.format_instruction = format_instruction

    def get_prompt(self) -> PromptTemplate:
        """
        构造 PromptTemplate 对象
        """
        return PromptTemplate(
            template=self.prompt_template,
            input_variables=self.input_variables,
            partial_variables={"format_instructions": self.format_instruction}
        )
    
# 通用处理器类
class TaskHandler:
    def __init__(self, model: Union[ChatOpenAI, ChatDeepSeek], parser: JsonOutputParser):
        self.model = model
        self.parser = parser
        self.validator = ErrorAnalysisValidator()

    def process_task(self, task: BaseTask, inputs: Dict[str, Any]) -> Any:
        """
        通用任务处理方法
        
        :param task: 任务对象
        :param inputs: 输入数据的字典
        :return: 任务处理结果
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
    
    async def process_task_async(self, task: BaseTask, inputs: Dict[str, Any]) -> Any:
        """
        异步处理任务方法
        
        :param task: 任务对象
        :param inputs: 输入数据的字典
        :return: 任务处理结果
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

    async def stream_task(self, task: BaseTask, inputs: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理任务方法，支持实时返回部分结果
        
        :param task: 任务对象
        :param inputs: 输入数据的字典
        :yield: 部分结果数据
        """
        prompt = task.get_prompt()
        chain = prompt | self.model
        
        # 记录处理开始
        logger.info(f"开始流式处理任务: {task.__class__.__name__}")
        if 'essay' in inputs:
            logger.info(f"作文长度: {len(inputs['essay'])} 字符")
        
        collected_content = ""
        result_dict = {}
        
        # 正则表达式匹配键值对
        key_pattern = r'"([^"]+)"\s*:'
        section_pattern = r'"([^"]+)"\s*:\s*(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*")'
        
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
                    if "{" in collected_content and "}" in collected_content:
                        # 尝试提取最外层的JSON部分
                        start_idx = collected_content.find("{")
                        end_idx = collected_content.rfind("}")
                        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                            possible_json = collected_content[start_idx:end_idx+1]
                            try:
                                partial_result = json.loads(possible_json)
                                if isinstance(partial_result, dict):
                                    result_dict = partial_result  # 完全替换为新解析的内容
                                    yield {"status": "processing", "partial_result": result_dict}
                                    continue  # 如果完整解析成功，跳过部分解析
                            except json.JSONDecodeError:
                                pass  # 继续尝试部分解析
                    
                    # 如果完整解析失败，尝试提取部分键值对
                    # 查找已完成的部分（评分、错误分析、亮点分析、写作建议）
                    for section in ["评分", "错误分析", "亮点分析", "写作建议"]:
                        section_match = re.search(f'"{section}"\\s*:\\s*(\\{{[^{{}}]*\\}}|\\[[^\\[\\]]*\\]|"[^"]*")', collected_content)
                        if section_match:
                            try:
                                section_content = section_match.group(1)
                                # 确保它是有效的JSON
                                if section_content.startswith('{') and section_content.endswith('}'):
                                    section_json = json.loads(section_content)
                                    result_dict[section] = section_json
                                elif section_content.startswith('[') and section_content.endswith(']'):
                                    section_json = json.loads(section_content)
                                    result_dict[section] = section_json
                                elif section_content.startswith('"') and section_content.endswith('"'):
                                    # 处理字符串值（如"写作建议"）
                                    result_dict[section] = section_content.strip('"')
                            except json.JSONDecodeError:
                                pass  # 忽略无法解析的部分
                    
                    # 如果有解析结果，返回部分结果
                    if result_dict:
                        yield {"status": "processing", "partial_result": result_dict}
                    
                except Exception as e:
                    logger.error(f"解析流式内容时出错: {str(e)}")
            
            # 流处理完成后，尝试完整解析所有内容
            try:
                # 查找最外层的JSON部分
                start_idx = collected_content.find("{")
                end_idx = collected_content.rfind("}")
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    possible_json = collected_content[start_idx:end_idx+1]
                    final_result = json.loads(possible_json)
                    
                    # 验证错误位置
                    if 'essay' in inputs:
                        final_result = self.validator.validate_error_positions(inputs['essay'], final_result)
                    
                    yield {"status": "completed", "result": final_result}
            except Exception as e:
                logger.error(f"解析最终流式结果时出错: {str(e)}")
                yield {"status": "error", "error": str(e)}
                
        except Exception as e:
            logger.error(f"流式任务处理失败: {str(e)}")
            yield {"status": "error", "error": str(e)}

# 初始化模型和解析器
model = ChatDeepSeek(**deepseek_config.LLM_CONFIG)
parser = JsonOutputParser()
task_handler = TaskHandler(model, parser)

# 具体任务定义
task_letter = BaseTask(
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
    
    :param essay: 待批改文章
    :return: Json格式输出(py字典)
    """
    logger.info(f"收到作文批改请求，作文长度: {len(essay)} 字符")
    result = task_handler.process_task(task_letter, {"essay": essay})
    
    # 直接打印完整响应到主日志
    logger.info(f"作文批改完成，结果: {json.dumps(result, ensure_ascii=False)}")
    
    return result

# 异步任务处理函数
async def handler_letter_correct_async(essay: str) -> Dict[str, Any]:
    """
    异步处理书信作文批改任务
    
    :param essay: 待批改文章
    :return: Json格式输出(py字典)
    """
    logger.info(f"收到异步作文批改请求，作文长度: {len(essay)} 字符")
    result = await task_handler.process_task_async(task_letter, {"essay": essay})
    
    # 直接打印完整响应到主日志
    logger.info(f"异步作文批改完成，结果: {json.dumps(result, ensure_ascii=False)}")
    
    return result

# 流式处理函数
async def handler_letter_correct_stream(essay: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式处理书信作文批改任务
    
    :param essay: 待批改文章
    :yield: 部分结果或完整结果
    """
    logger.info(f"收到流式作文批改请求，作文长度: {len(essay)} 字符")
    
    async for part_result in task_handler.stream_task(task_letter, {"essay": essay}):
        yield part_result
    
    logger.info("流式作文批改完成")
