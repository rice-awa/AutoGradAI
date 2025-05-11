"""
数据库模型定义
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class CorrectionHistory(db.Model):
    """批改历史记录模型"""
    __tablename__ = 'correction_history'
    
    id = db.Column(db.String(36), primary_key=True)
    essay = db.Column(db.Text, nullable=False)
    result_json = db.Column(db.Text, nullable=False)  # 存储JSON格式的批改结果
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    mode = db.Column(db.String(20), nullable=False)  # 批改模式: sync, async, stream
    score = db.Column(db.String(10), default="未评分")
    error_count = db.Column(db.Integer, default=0)
    essay_length = db.Column(db.Integer, default=0)
    
    def __init__(self, id, essay, result, mode, score="未评分", error_count=0):
        """初始化批改历史记录"""
        self.id = id
        self.essay = essay
        self.result_json = json.dumps(result, ensure_ascii=False)
        self.mode = mode
        self.score = score
        self.error_count = error_count
        self.essay_length = len(essay)
    
    @property
    def result(self):
        """获取反序列化的批改结果"""
        return json.loads(self.result_json)
    
    @property
    def formatted_time(self):
        """获取格式化的时间字符串"""
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self):
        """转换为字典表示"""
        return {
            "id": self.id,
            "essay": self.essay,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "formatted_time": self.formatted_time,
            "mode": self.mode,
            "score": self.score,
            "error_count": self.error_count,
            "essay_length": self.essay_length
        }

def init_db(app):
    """初始化数据库"""
    db.init_app(app)
    
    # 创建数据库表
    with app.app_context():
        db.create_all() 