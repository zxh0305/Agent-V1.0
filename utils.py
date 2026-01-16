#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
============

本模块提供智能 Agent 平台的通用工具函数

作者：学生开发团队
版本：1.0.0
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    配置日志系统

    Args:
        log_dir: 日志目录
        level: 日志级别

    Returns:
        配置好的 Logger 实例
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"agent_{datetime.now().strftime('%Y%m%d')}.log"
    )

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("AgentPlatform")


def generate_id(prefix: str = "") -> str:
    """
    生成唯一ID

    Args:
        prefix: ID前缀

    Returns:
        唯一ID字符串
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    hash_part = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{hash_part}" if prefix else f"{timestamp}_{hash_part}"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全的 JSON 解析

    Args:
        json_str: JSON 字符串
        default: 解析失败时的默认值

    Returns:
        解析结果或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    安全的 JSON 序列化

    Args:
        obj: 待序列化的对象
        **kwargs: json.dumps 的额外参数

    Returns:
        JSON 字符串
    """
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError):
        return "{}"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断字符串

    Args:
        s: 原始字符串
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的字符串
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def extract_file_extension(filename: str) -> str:
    """
    提取文件扩展名

    Args:
        filename: 文件名

    Returns:
        小写的扩展名（不含点）
    """
    if '.' in filename:
        return filename.rsplit('.', 1)[-1].lower()
    return ""


def is_valid_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    检查文件类型是否允许

    Args:
        filename: 文件名
        allowed_types: 允许的文件类型列表

    Returns:
        是否允许
    """
    ext = extract_file_extension(filename)
    return ext in [t.lower() for t in allowed_types]


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def format_datetime(dt: Union[datetime, str], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化日期时间

    Args:
        dt: datetime 对象或 ISO 格式字符串
        fmt: 输出格式

    Returns:
        格式化后的日期时间字符串
    """
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt

    return dt.strftime(fmt)


def clean_text(text: str) -> str:
    """
    清理文本

    移除多余空白、特殊字符等

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return text.strip()


def extract_numbers(text: str) -> List[float]:
    """
    从文本中提取数字

    Args:
        text: 原始文本

    Returns:
        提取的数字列表
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    递归合并字典

    Args:
        dict1: 第一个字典
        dict2: 第二个字典

    Returns:
        合并后的字典
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def validate_date_format(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
    """
    验证日期格式

    Args:
        date_str: 日期字符串
        fmt: 期望的格式

    Returns:
        是否有效
    """
    try:
        datetime.strptime(date_str, fmt)
        return True
    except ValueError:
        return False


def get_current_timestamp() -> str:
    """
    获取当前时间戳字符串

    Returns:
        ISO 格式的时间戳字符串
    """
    return datetime.now().isoformat()


class Timer:
    """
    计时器类

    用于测量代码执行时间
    """

    def __init__(self, name: str = "Timer"):
        """
        初始化计时器

        Args:
            name: 计时器名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """进入上下文"""
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """退出上下文"""
        self.end_time = datetime.now()

    @property
    def elapsed(self) -> float:
        """获取经过的秒数"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def elapsed_ms(self) -> float:
        """获取经过的毫秒数"""
        return self.elapsed * 1000


# 测试代码
if __name__ == "__main__":
    # 测试工具函数
    print("=" * 40)
    print("工具函数测试")
    print("=" * 40)

    # 测试 ID 生成
    print(f"\n生成 ID: {generate_id('TEST')}")

    # 测试字符串截断
    long_text = "这是一段很长的文本，需要被截断显示"
    print(f"截断字符串: {truncate_string(long_text, 15)}")

    # 测试文件大小格式化
    print(f"格式化大小: {format_file_size(1234567)}")

    # 测试数字提取
    text = "本次实验数据：温度25.5度，湿度60%，压力101.3kPa"
    print(f"提取数字: {extract_numbers(text)}")

    # 测试计时器
    with Timer("测试计时") as t:
        import time
        time.sleep(0.1)
    print(f"计时结果: {t.elapsed_ms:.2f}ms")
