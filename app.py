from flask import Flask, request, render_template, jsonify, Response, redirect, url_for, send_file
import asyncio
import json
import os
import uuid
import datetime
from threading import Thread
from queue import Queue
from main import handler_letter_correct, handler_letter_correct_async, log_llm_response, handler_letter_correct_stream
from logger import setup_logger
from models import db, CorrectionHistory, init_db
import csv
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///correction_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
init_db(app)

# 配置日志
logger = setup_logger(__name__)

# 任务队列与结果缓存
task_queue = Queue()
results_cache = {}
stream_results = {}  # 存储流式响应结果

# 批改历史记录存储
correction_history = {}  # 使用内存存储批改历史记录

# 注意：在异步函数中使用数据库操作时，必须确保在应用上下文中执行
# 可以使用 with app.app_context(): 或者调用 save_history_async 辅助函数
def save_correction_to_history(essay: str, result: dict, mode: str = "sync") -> str:
    """
    保存批改结果到历史记录
    
    :param essay: 用户提交的作文
    :param result: 批改结果
    :param mode: 批改模式(sync/async/stream)
    :return: 历史记录ID
    """
    try:
        history_id = str(uuid.uuid4())
        
        # 提取评分
        score = "未评分"
        if result and "评分" in result and "分数" in result["评分"]:
            score = result["评分"]["分数"]
        
        # 统计错误数量
        error_count = count_errors(result)
        
        # 创建数据库记录
        history_record = CorrectionHistory(
            id=history_id,
            essay=essay,
            result=result,
            mode=mode,
            score=score,
            error_count=error_count
        )
        
        # 保存到数据库
        db.session.add(history_record)
        db.session.commit()
        
        logger.info(f"批改历史已保存到数据库，ID: {history_id}, 模式: {mode}, 分数: {score}")
        return history_id
    except Exception as e:
        logger.error(f"保存批改历史失败: {str(e)}")
        # 尝试回滚事务
        try:
            db.session.rollback()
        except Exception as rollback_error:
            logger.error(f"回滚事务失败: {str(rollback_error)}")
        return ""

def process_essay(essay: str):
    """
    处理作文并返回批改结果。

    :param essay: 用户提交的作文
    :return: 批改结果
    """
    try:
        # 记录作文内容到日志目录
        save_essay_to_log(essay, "sync")
        
        # 调用批改函数
        result = handler_letter_correct(essay)
        
        # 记录返回数据摘要
        logger.info(f"批改完成，错误数量: {count_errors(result)}")
        
        # 保存到历史记录
        history_id = save_correction_to_history(essay, result, "sync")
        if history_id:
            result["history_id"] = history_id
        
        return result
    except Exception as e:
        logger.error(f"批改失败: {str(e)}")
        raise

def count_errors(result):
    """统计错误数量"""
    total = 0
    for category, errors in result.get('错误分析', {}).items():
        total += len(errors)
    return total

def save_essay_to_log(essay, mode="sync"):
    """保存作文内容到日志文件"""
    try:
        # 确保日志目录存在
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建唯一文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'essay_{mode}_{timestamp}.txt')
        
        # 写入作文内容
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(essay)
        
        logger.info(f"作文内容已保存到 {log_file}")
    except Exception as e:
        logger.error(f"保存作文失败: {str(e)}")

async def process_essay_async(essay: str, task_id: str):
    """
    异步处理作文

    :param essay: 用户提交的作文
    :param task_id: 任务ID
    """
    try:
        # 记录作文内容到日志
        save_essay_to_log(essay, f"async_{task_id}")
        
        # 调用异步批改函数
        result = await handler_letter_correct_async(essay)
        
        # 保存到历史记录（使用辅助函数）
        history_id = await save_history_async(essay, result, "async")
        if history_id:
            result["history_id"] = history_id
        
        # 更新任务状态
        results_cache[task_id].update({
            "status": "completed",
            "result": result
        })
        
        # 记录完成信息
        logger.info(f"任务 {task_id} 完成，错误数量: {count_errors(result)}")
    except Exception as e:
        results_cache[task_id].update({
            "status": "error",
            "error": str(e)
        })
        logger.error(f"任务 {task_id} 失败: {str(e)}")

async def process_stream_essay(essay: str, task_id: str):
    """
    流式处理作文

    :param essay: 用户提交的作文
    :param task_id: 任务ID
    """
    try:
        # 记录作文内容
        save_essay_to_log(essay, f"stream_{task_id}")
        
        # 初始化结果存储
        stream_results[task_id] = {
            "status": "processing",
            "last_chunk": None,
            "partial_result": {},
            "final_result": None,
            "essay": essay,
            "updates_count": 0  # 添加更新计数器
        }
        
        # 调用流式批改函数
        async for chunk in handler_letter_correct_stream(essay):
            # 更新最新的块数据
            stream_results[task_id]["last_chunk"] = chunk
            stream_results[task_id]["updates_count"] += 1
            
            # 如果有部分结果，更新部分结果
            if "partial_result" in chunk:
                stream_results[task_id]["partial_result"] = chunk["partial_result"]  # 完全替换，不是更新
                logger.info(f"流式任务 {task_id} 更新第 {stream_results[task_id]['updates_count']} 次，获得部分结果")
            
            # 如果处理完成，保存最终结果
            if chunk.get("status") == "completed":
                stream_results[task_id]["status"] = "completed"
                stream_results[task_id]["final_result"] = chunk.get("result")
                
                # 保存到历史记录（使用辅助函数）
                if "result" in chunk:
                    history_id = await save_history_async(essay, chunk["result"], "stream")
                    if history_id:
                        stream_results[task_id]["history_id"] = history_id
                
                logger.info(f"流式任务 {task_id} 完成")
                break
            
            # 如果出错，记录错误
            if chunk.get("status") == "error":
                stream_results[task_id]["status"] = "error"
                stream_results[task_id]["error"] = chunk.get("error")
                logger.error(f"流式任务 {task_id} 失败: {chunk.get('error')}")
                break
                
    except Exception as e:
        stream_results[task_id]["status"] = "error"
        stream_results[task_id]["error"] = str(e)
        logger.error(f"流式任务 {task_id} 处理异常: {str(e)}")

# 双引擎批改函数（目前是框架，尚未实现具体功能）
async def process_dual_engine_async(essay: str, task_id: str):
    """
    使用双引擎批改作文（DeepSeek + Qwen）

    :param essay: 用户提交的作文
    :param task_id: 任务ID
    """
    try:
        # 记录作文内容
        save_essay_to_log(essay, f"dual_{task_id}")
        
        # 初始化结果状态
        results_cache[task_id].update({
            "status": "processing_dual_engine",
            "progress": "正在使用双引擎进行批改..."
        })
        
        # 这里应该实现两个模型的并行调用
        # TODO: 实现双引擎调用和结果合并逻辑
        
        # 模拟双引擎处理
        result = await handler_letter_correct_async(essay)
        
        # 保存到历史记录（使用辅助函数）
        history_id = await save_history_async(essay, result, "dual")
        if history_id:
            result["history_id"] = history_id
        
        # 更新任务状态
        results_cache[task_id].update({
            "status": "completed",
            "result": result,
            "engine": "dual"
        })
        
        logger.info(f"双引擎任务 {task_id} 完成")
    except Exception as e:
        results_cache[task_id].update({
            "status": "error",
            "error": str(e)
        })
        logger.error(f"双引擎任务 {task_id} 失败: {str(e)}")

def background_task_processor():
    """
    后台任务处理线程
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        if not task_queue.empty():
            task = task_queue.get()
            essay = task["essay"]
            task_id = task["task_id"]
            task_type = task.get("type", "async")
            
            logger.info(f"开始处理{task_type}任务 {task_id}")
            
            # 更新任务状态为处理中
            if task_type == "stream":
                # 流式处理
                loop.run_until_complete(process_stream_essay(essay, task_id))
            elif task_type == "dual":
                # 双引擎处理
                results_cache[task_id].update({"status": "processing"})
                loop.run_until_complete(process_dual_engine_async(essay, task_id))
            else:
                # 常规异步处理
                results_cache[task_id].update({"status": "processing"})
                loop.run_until_complete(process_essay_async(essay, task_id))
            
            task_queue.task_done()

# 启动后台任务处理线程
background_thread = Thread(target=background_task_processor, daemon=True)
background_thread.start()

@app.route("/", methods=["GET", "POST"])
def index():
    """
    处理首页请求。
    """
    if request.method == "POST":
        essay = request.form.get("essay")
        process_type = request.form.get("process_type", "sync")
        engine_type = request.form.get("engine_type", "single")
        
        if not essay:
            logger.warning("提交了空作文")
            return render_template("index.html", error="请提交一篇作文！")

        logger.info(f"收到作文批改请求，内容长度: {len(essay)} 字符，处理方式: {process_type}，引擎类型: {engine_type}")
        
        # 只有同步处理在这里处理，其他方式通过JavaScript AJAX提交
        if process_type == "sync":
            try:
                # 同步处理模式
                result = process_essay(essay)
                return render_template("result.html", essay=essay, result=result)
            except Exception as e:
                logger.error(f"处理作文时出错: {str(e)}")
                return render_template("index.html", error=f"批改失败: {str(e)}")

    return render_template("index.html")

@app.route("/api/submit", methods=["POST"])
def api_submit():
    """
    API端点 - 提交作文进行异步处理
    """
    essay = request.json.get("essay")
    engine_type = request.json.get("engine", "single")  # 默认使用单引擎
    
    if not essay:
        logger.warning("API收到空作文请求")
        return jsonify({"error": "请提交作文内容"}), 400
    
    logger.info(f"收到异步作文批改请求，内容长度: {len(essay)} 字符，引擎类型: {engine_type}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"生成任务ID: {task_id}")
    
    # 加入任务队列
    task_queue.put({
        "essay": essay,
        "task_id": task_id,
        "type": "dual" if engine_type == "dual" else "async"
    })
    
    # 初始化任务状态
    results_cache[task_id] = {
        "status": "queued",
        "essay": essay,  # 保存原始作文文本
        "engine": engine_type
    }
    
    return jsonify({
        "task_id": task_id,
        "status": "queued"
    })

@app.route("/api/status/<task_id>", methods=["GET"])
def api_status(task_id):
    """
    API端点 - 检查任务状态
    """
    if task_id not in results_cache:
        logger.warning(f"请求了不存在的任务ID: {task_id}")
        return jsonify({"error": "任务不存在"}), 404
    
    status_data = results_cache[task_id]
    logger.debug(f"任务 {task_id} 状态: {status_data.get('status')}")
    
    # 如果任务完成，返回结果
    if status_data.get("status") == "completed":
        return jsonify({
            "status": "completed",
            "result": status_data.get("result")
        })
    
    # 如果任务出错，返回错误信息
    if status_data.get("status") == "error":
        return jsonify({
            "status": "error",
            "error": status_data.get("error")
        })
    
    # 其他状态
    return jsonify({
        "status": status_data.get("status")
    })

@app.route("/async-result/<task_id>")
def async_result(task_id):
    """
    显示异步任务结果页面
    """
    if task_id not in results_cache:
        logger.warning(f"尝试访问不存在的任务结果: {task_id}")
        return render_template("index.html", error="任务不存在")
    
    status_data = results_cache[task_id]
    
    if status_data.get("status") == "completed":
        result = status_data.get("result")
        essay = status_data.get("essay", "")
        logger.info(f"显示任务 {task_id} 的结果")
        return render_template("result.html", essay=essay, result=result)
    
    if status_data.get("status") == "error":
        error = status_data.get("error")
        logger.error(f"显示任务 {task_id} 的错误: {error}")
        return render_template("index.html", error=f"批改失败: {error}")
    
    # 如果任务还在处理中，显示等待页面
    logger.info(f"任务 {task_id} 仍在处理中，显示等待页面")
    return render_template("waiting.html", task_id=task_id)

@app.route("/api/stream", methods=["POST"])
def api_stream_submit():
    """
    API端点 - 提交作文进行流式处理
    """
    essay = request.json.get("essay")
    if not essay:
        logger.warning("API收到空作文流式请求")
        return jsonify({"error": "请提交作文内容"}), 400
    
    logger.info(f"收到流式作文批改请求，内容长度: {len(essay)} 字符")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"生成流式任务ID: {task_id}")
    
    # 加入任务队列
    task_queue.put({
        "essay": essay,
        "task_id": task_id,
        "type": "stream"
    })
    
    # 初始化流式任务状态
    stream_results[task_id] = {
        "status": "queued",
        "essay": essay,
        "partial_result": {}
    }
    
    return jsonify({
        "task_id": task_id,
        "status": "queued"
    })

@app.route("/api/stream-status/<task_id>", methods=["GET"])
def api_stream_status(task_id):
    """
    API端点 - 获取流式处理状态和部分结果
    """
    if task_id not in stream_results:
        logger.warning(f"请求了不存在的流式任务ID: {task_id}")
        return jsonify({"error": "任务不存在"}), 404
    
    status_data = stream_results[task_id]
    logger.debug(f"流式任务 {task_id} 状态: {status_data.get('status')}")
    
    # 构建响应数据
    response_data = {
        "status": status_data.get("status", "unknown"),
        "partial_result": status_data.get("partial_result", {})
    }
    
    # 如果任务完成，返回最终结果
    if status_data.get("status") == "completed":
        response_data["complete"] = True
        response_data["result"] = status_data.get("final_result")
    
    # 如果任务出错，返回错误信息
    if status_data.get("status") == "error":
        response_data["error"] = status_data.get("error")
    
    return jsonify(response_data)

@app.route("/stream-result/<task_id>")
def stream_result(task_id):
    """
    显示流式任务结果页面
    """
    if task_id not in stream_results:
        logger.warning(f"尝试访问不存在的流式任务结果: {task_id}")
        return render_template("index.html", error="流式任务不存在")
    
    status_data = stream_results[task_id]
    essay = status_data.get("essay", "")
    
    # 无论任务是否完成，都直接进入结果页面，通过JavaScript获取实时更新
    logger.info(f"显示流式任务 {task_id} 的结果页面")
    
    # 如果已经完成，传递最终结果
    if status_data.get("status") == "completed":
        result = status_data.get("final_result")
        return render_template("result.html", essay=essay, result=result, task_id=task_id, stream_mode=True, is_completed=True)
    
    # 如果出错，返回错误信息
    if status_data.get("status") == "error":
        error = status_data.get("error")
        logger.error(f"显示流式任务 {task_id} 的错误: {error}")
        return render_template("index.html", error=f"批改失败: {error}")
    
    # 如果任务还在处理中，传递部分结果（如果有）
    partial_result = status_data.get("partial_result", {})
    return render_template("result.html", essay=essay, result=partial_result, task_id=task_id, stream_mode=True, is_completed=False)

@app.route("/api/sample/<sample_id>")
def api_sample(sample_id):
    """
    API端点 - 获取示例作文内容
    
    :param sample_id: 示例作文ID
    :return: 示例作文内容的JSON响应
    """
    try:
        # 安全检查，防止路径遍历
        if sample_id.startswith('.') or '/' in sample_id or '\\' in sample_id:
            logger.warning(f"尝试访问不安全的示例作文ID: {sample_id}")
            return jsonify({"error": "无效的示例作文ID"}), 400
            
        # 构建示例文件路径
        sample_file = os.path.join('static', 'samples', f"{sample_id}.txt")
        
        # 检查文件是否存在
        if not os.path.exists(sample_file):
            logger.warning(f"请求了不存在的示例作文: {sample_id}")
            return jsonify({"error": "示例作文不存在"}), 404
            
        # 读取示例作文内容
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"提供示例作文: {sample_id}")
        return jsonify({"content": content})
    except Exception as e:
        logger.error(f"获取示例作文时出错: {str(e)}")
        return jsonify({"error": f"获取示例作文失败: {str(e)}"}), 500

# 添加批改历史记录路由
@app.route("/history")
def correction_history_page():
    """
    显示批改历史记录列表
    """
    # 从数据库中查询所有历史记录并按时间倒序排序
    history_records = CorrectionHistory.query.order_by(CorrectionHistory.timestamp.desc()).all()
    return render_template("history.html", history_records=history_records)

@app.route("/history/<history_id>")
def view_history_detail(history_id):
    """
    查看特定历史记录的详细信息
    
    :param history_id: 历史记录ID
    """
    # 从数据库中查询指定ID的历史记录
    record = CorrectionHistory.query.get(history_id)
    
    if not record:
        return render_template("error.html", message="未找到该历史记录")
    
    return render_template("result.html", 
                          essay=record.essay, 
                          result=record.result, 
                          from_history=True,
                          history_id=record.id,
                          timestamp=record.formatted_time)

@app.route("/history/delete/<history_id>", methods=["POST"])
def delete_history(history_id):
    """
    删除指定的历史记录
    
    :param history_id: 历史记录ID
    :return: JSON响应
    """
    try:
        # 查询指定ID的历史记录
        record = CorrectionHistory.query.get(history_id)
        
        if not record:
            return jsonify({"success": False, "error": "未找到该历史记录"}), 404
        
        # 从数据库中删除
        db.session.delete(record)
        db.session.commit()
        
        logger.info(f"已删除历史记录，ID: {history_id}")
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"删除历史记录失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/history/clear", methods=["POST"])
def clear_history():
    """
    清空所有历史记录
    
    :return: JSON响应
    """
    try:
        # 删除所有历史记录
        CorrectionHistory.query.delete()
        db.session.commit()
        
        logger.info("已清空所有历史记录")
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"清空历史记录失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/history/export", methods=["GET"])
def export_history():
    """
    导出所有历史记录为CSV文件
    
    :return: CSV文件下载
    """
    try:
        # 查询所有历史记录
        records = CorrectionHistory.query.order_by(CorrectionHistory.timestamp.desc()).all()
        
        # 创建CSV文件
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入CSV头部
        writer.writerow(['ID', '作文内容', '批改时间', '批改模式', '评分', '错误数量', '作文长度'])
        
        # 写入数据
        for record in records:
            writer.writerow([
                record.id,
                record.essay,
                record.formatted_time,
                record.mode,
                record.score,
                record.error_count,
                record.essay_length
            ])
        
        # 准备文件下载
        output.seek(0)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),  # 使用UTF-8-SIG编码以支持Excel正确显示中文
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'批改历史记录_{timestamp}.csv'
        )
    except Exception as e:
        logger.error(f"导出历史记录失败: {str(e)}")
        return render_template("error.html", message=f"导出历史记录失败: {str(e)}")

@app.route("/history/export/<history_id>", methods=["GET"])
def export_single_history(history_id):
    """
    导出单条历史记录为JSON文件
    
    :param history_id: 历史记录ID
    :return: JSON文件下载
    """
    try:
        # 查询指定ID的历史记录
        record = CorrectionHistory.query.get(history_id)
        
        if not record:
            return render_template("error.html", message="未找到该历史记录")
        
        # 转换为字典
        record_dict = record.to_dict()
        
        # 准备JSON文件下载
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            io.BytesIO(json.dumps(record_dict, ensure_ascii=False, indent=2).encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'批改记录_{history_id}_{timestamp}.json'
        )
    except Exception as e:
        logger.error(f"导出单条历史记录失败: {str(e)}")
        return render_template("error.html", message=f"导出历史记录失败: {str(e)}")

async def save_history_async(essay: str, result: dict, mode: str = "sync") -> str:
    """
    在异步环境中安全地保存历史记录
    
    :param essay: 用户提交的作文
    :param result: 批改结果
    :param mode: 批改模式(sync/async/stream)
    :return: 历史记录ID
    """
    try:
        with app.app_context():
            history_id = save_correction_to_history(essay, result, mode)
            return history_id
    except Exception as e:
        logger.error(f"异步环境中保存历史记录失败: {str(e)}")
        return ""

if __name__ == "__main__":
    logger.info("启动应用服务器...")
    app.run(debug=True, port=5000, host='0.0.0.0')
