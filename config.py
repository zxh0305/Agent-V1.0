#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
============

本模块管理智能 Agent 平台的全局配置项

作者：学生开发团队
版本：1.0.0
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DatabaseConfig:
    """数据库配置"""
    task_db_path: str = "data/tasks.db"
    schedule_db_path: str = "data/schedules.db"


@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = "data"
    export_dir: str = "exports"
    upload_dir: str = "uploads"
    log_dir: str = "logs"


@dataclass
class ToolConfig:
    """工具配置"""
    code_execution_timeout: int = 10  # 代码执行超时（秒）
    max_output_length: int = 5000     # 最大输出长度
    allowed_file_types: List[str] = field(default_factory=lambda: [
        "pdf", "xlsx", "xls", "csv", "docx", "doc", "txt", "py"
    ])


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_steps: int = 10               # 最大执行步骤数
    enable_logging: bool = True       # 是否启用日志
    log_level: str = "INFO"           # 日志级别


@dataclass
class UIConfig:
    """界面配置"""
    page_title: str = "智能任务处理 Agent 平台"
    page_icon: str = "🤖"
    theme: str = "light"
    max_history_display: int = 10     # 历史任务显示数量


@dataclass
class AppConfig:
    """
    应用总配置类

    整合所有配置项
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def __post_init__(self):
        """初始化后创建必要的目录"""
        dirs = [
            self.paths.data_dir,
            self.paths.export_dir,
            self.paths.upload_dir,
            self.paths.log_dir
        ]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "database": {
                "task_db_path": self.database.task_db_path,
                "schedule_db_path": self.database.schedule_db_path
            },
            "paths": {
                "data_dir": self.paths.data_dir,
                "export_dir": self.paths.export_dir,
                "upload_dir": self.paths.upload_dir,
                "log_dir": self.paths.log_dir
            },
            "tools": {
                "code_execution_timeout": self.tools.code_execution_timeout,
                "max_output_length": self.tools.max_output_length,
                "allowed_file_types": self.tools.allowed_file_types
            },
            "agent": {
                "max_steps": self.agent.max_steps,
                "enable_logging": self.agent.enable_logging,
                "log_level": self.agent.log_level
            },
            "ui": {
                "page_title": self.ui.page_title,
                "page_icon": self.ui.page_icon,
                "theme": self.ui.theme,
                "max_history_display": self.ui.max_history_display
            }
        }


# 全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
    """获取全局配置实例"""
    return config


def update_config(**kwargs) -> AppConfig:
    """
    更新配置项

    Args:
        **kwargs: 配置键值对

    Returns:
        更新后的配置实例
    """
    global config

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
