#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理模块
============

本模块实现了智能 Agent 平台的任务管理功能，包括：
1. 任务记录：保存用户输入、执行步骤、结果
2. 状态监控：待执行/执行中/完成/失败
3. 结果导出：支持 Markdown 和 Excel 格式导出

作者：学生开发团队
版本：1.0.0
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"         # 待执行
    EXECUTING = "executing"     # 执行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskStep:
    """
    任务执行步骤数据类

    Attributes:
        step_id: 步骤唯一标识
        step_order: 步骤顺序
        description: 步骤描述
        tool_name: 使用的工具名称
        params: 工具参数
        status: 步骤状态
        result: 执行结果
        error_message: 错误信息
        started_at: 开始时间
        completed_at: 完成时间
    """
    step_id: str
    step_order: int
    description: str
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class TaskRecord:
    """
    任务记录数据类

    Attributes:
        task_id: 任务唯一标识
        user_input: 用户原始输入
        parsed_intent: 解析后的意图
        steps: 执行步骤列表
        status: 任务状态
        priority: 任务优先级
        final_result: 最终结果
        error_message: 错误信息
        created_at: 创建时间
        updated_at: 更新时间
        completed_at: 完成时间
        metadata: 额外元数据
    """
    task_id: str
    user_input: str
    parsed_intent: Optional[str] = None
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    final_result: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskStorage:
    """
    任务存储类

    使用 SQLite 数据库持久化存储任务记录

    Attributes:
        db_path: 数据库文件路径
    """

    def __init__(self, db_path: str = "tasks.db"):
        """
        初始化任务存储

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建任务主表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                user_input TEXT NOT NULL,
                parsed_intent TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 2,
                final_result TEXT,
                error_message TEXT,
                created_at TEXT,
                updated_at TEXT,
                completed_at TEXT,
                metadata TEXT
            )
        ''')

        # 创建任务步骤表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_steps (
                step_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                step_order INTEGER NOT NULL,
                description TEXT NOT NULL,
                tool_name TEXT,
                params TEXT,
                status TEXT DEFAULT 'pending',
                result TEXT,
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (task_id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks (created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_steps_task ON task_steps (task_id)')

        conn.commit()
        conn.close()
        logger.info(f"任务数据库初始化完成: {self.db_path}")

    def save_task(self, task: TaskRecord) -> bool:
        """
        保存任务记录

        Args:
            task: 任务记录对象

        Returns:
            是否保存成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 插入或更新任务主记录
            cursor.execute('''
                INSERT OR REPLACE INTO tasks
                (task_id, user_input, parsed_intent, status, priority,
                 final_result, error_message, created_at, updated_at,
                 completed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.user_input,
                task.parsed_intent,
                task.status.value,
                task.priority.value,
                task.final_result,
                task.error_message,
                task.created_at,
                datetime.now().isoformat(),
                task.completed_at,
                json.dumps(task.metadata, ensure_ascii=False)
            ))

            # 删除旧的步骤记录
            cursor.execute('DELETE FROM task_steps WHERE task_id = ?', (task.task_id,))

            # 插入步骤记录
            for step in task.steps:
                cursor.execute('''
                    INSERT INTO task_steps
                    (step_id, task_id, step_order, description, tool_name,
                     params, status, result, error_message, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    step.step_id,
                    task.task_id,
                    step.step_order,
                    step.description,
                    step.tool_name,
                    json.dumps(step.params, ensure_ascii=False),
                    step.status.value,
                    json.dumps(step.result, ensure_ascii=False) if step.result else None,
                    step.error_message,
                    step.started_at,
                    step.completed_at
                ))

            conn.commit()
            conn.close()
            logger.info(f"任务保存成功: {task.task_id}")
            return True

        except Exception as e:
            logger.error(f"任务保存失败: {str(e)}")
            return False

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """
        获取任务记录

        Args:
            task_id: 任务ID

        Returns:
            任务记录对象，不存在返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 查询任务主记录
            cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            # 查询任务步骤
            cursor.execute('''
                SELECT * FROM task_steps
                WHERE task_id = ?
                ORDER BY step_order
            ''', (task_id,))
            step_rows = cursor.fetchall()

            conn.close()

            # 构建任务记录对象
            steps = []
            for step_row in step_rows:
                step = TaskStep(
                    step_id=step_row[0],
                    step_order=step_row[2],
                    description=step_row[3],
                    tool_name=step_row[4],
                    params=json.loads(step_row[5]) if step_row[5] else {},
                    status=TaskStatus(step_row[6]),
                    result=json.loads(step_row[7]) if step_row[7] else None,
                    error_message=step_row[8],
                    started_at=step_row[9],
                    completed_at=step_row[10]
                )
                steps.append(step)

            task = TaskRecord(
                task_id=row[0],
                user_input=row[1],
                parsed_intent=row[2],
                status=TaskStatus(row[3]),
                priority=TaskPriority(row[4]),
                final_result=row[5],
                error_message=row[6],
                created_at=row[7],
                updated_at=row[8],
                completed_at=row[9],
                metadata=json.loads(row[10]) if row[10] else {},
                steps=steps
            )

            return task

        except Exception as e:
            logger.error(f"获取任务失败: {str(e)}")
            return None

    def get_tasks_by_status(self, status: TaskStatus, limit: int = 50) -> List[TaskRecord]:
        """
        按状态获取任务列表

        Args:
            status: 任务状态
            limit: 返回数量限制

        Returns:
            任务记录列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT task_id FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (status.value, limit))

            task_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            tasks = []
            for task_id in task_ids:
                task = self.get_task(task_id)
                if task:
                    tasks.append(task)

            return tasks

        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            return []

    def get_recent_tasks(self, limit: int = 20) -> List[TaskRecord]:
        """
        获取最近的任务列表

        Args:
            limit: 返回数量限制

        Returns:
            任务记录列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT task_id FROM tasks
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))

            task_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            tasks = []
            for task_id in task_ids:
                task = self.get_task(task_id)
                if task:
                    tasks.append(task)

            return tasks

        except Exception as e:
            logger.error(f"获取最近任务失败: {str(e)}")
            return []

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务记录

        Args:
            task_id: 任务ID

        Returns:
            是否删除成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM task_steps WHERE task_id = ?', (task_id,))
            cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"任务删除成功: {task_id}")
            return deleted

        except Exception as e:
            logger.error(f"删除任务失败: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取任务统计信息

        Returns:
            统计信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总任务数
            cursor.execute('SELECT COUNT(*) FROM tasks')
            total = cursor.fetchone()[0]

            # 各状态任务数
            stats = {"total": total}
            for status in TaskStatus:
                cursor.execute(
                    'SELECT COUNT(*) FROM tasks WHERE status = ?',
                    (status.value,)
                )
                stats[status.value] = cursor.fetchone()[0]

            # 今日任务数
            cursor.execute('''
                SELECT COUNT(*) FROM tasks
                WHERE date(created_at) = date('now')
            ''')
            stats["today"] = cursor.fetchone()[0]

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"total": 0}


class TaskExporter:
    """
    任务导出类

    支持将任务记录导出为 Markdown 或 Excel 格式

    Attributes:
        output_dir: 导出文件输出目录
    """

    def __init__(self, output_dir: str = "exports"):
        """
        初始化任务导出器

        Args:
            output_dir: 导出文件输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_to_markdown(self, task: TaskRecord, filename: str = None) -> str:
        """
        导出任务为 Markdown 格式

        Args:
            task: 任务记录对象
            filename: 输出文件名（可选）

        Returns:
            导出文件路径
        """
        if not filename:
            filename = f"task_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        filepath = os.path.join(self.output_dir, filename)

        content = []
        content.append(f"# 任务报告\n")
        content.append(f"**任务ID**: {task.task_id}\n")
        content.append(f"**创建时间**: {task.created_at}\n")
        content.append(f"**状态**: {task.status.value}\n")
        content.append(f"**优先级**: {task.priority.name}\n")
        content.append(f"\n## 用户输入\n")
        content.append(f"{task.user_input}\n")

        if task.parsed_intent:
            content.append(f"\n## 解析意图\n")
            content.append(f"{task.parsed_intent}\n")

        if task.steps:
            content.append(f"\n## 执行步骤\n")
            for step in task.steps:
                status_icon = "✓" if step.status == TaskStatus.COMPLETED else "✗"
                content.append(f"\n### 步骤 {step.step_order}: {step.description}\n")
                content.append(f"- **状态**: {status_icon} {step.status.value}\n")
                if step.tool_name:
                    content.append(f"- **使用工具**: {step.tool_name}\n")
                if step.result:
                    content.append(f"- **执行结果**: \n```json\n{json.dumps(step.result, ensure_ascii=False, indent=2)}\n```\n")
                if step.error_message:
                    content.append(f"- **错误信息**: {step.error_message}\n")

        if task.final_result:
            content.append(f"\n## 最终结果\n")
            content.append(f"{task.final_result}\n")

        if task.error_message:
            content.append(f"\n## 错误信息\n")
            content.append(f"{task.error_message}\n")

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        logger.info(f"任务导出为 Markdown: {filepath}")
        return filepath

    def export_to_excel(self, tasks: List[TaskRecord], filename: str = None) -> str:
        """
        导出任务列表为 Excel 格式

        Args:
            tasks: 任务记录列表
            filename: 输出文件名（可选）

        Returns:
            导出文件路径
        """
        try:
            import pandas as pd

            if not filename:
                filename = f"tasks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            filepath = os.path.join(self.output_dir, filename)

            # 准备任务数据
            task_data = []
            for task in tasks:
                task_data.append({
                    "任务ID": task.task_id,
                    "用户输入": task.user_input,
                    "状态": task.status.value,
                    "优先级": task.priority.name,
                    "步骤数": len(task.steps),
                    "创建时间": task.created_at,
                    "完成时间": task.completed_at or "",
                    "最终结果": task.final_result or ""
                })

            # 准备步骤数据
            step_data = []
            for task in tasks:
                for step in task.steps:
                    step_data.append({
                        "任务ID": task.task_id,
                        "步骤序号": step.step_order,
                        "步骤描述": step.description,
                        "使用工具": step.tool_name or "",
                        "状态": step.status.value,
                        "开始时间": step.started_at or "",
                        "完成时间": step.completed_at or ""
                    })

            # 创建 Excel 文件
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                pd.DataFrame(task_data).to_excel(writer, sheet_name='任务列表', index=False)
                if step_data:
                    pd.DataFrame(step_data).to_excel(writer, sheet_name='执行步骤', index=False)

            logger.info(f"任务导出为 Excel: {filepath}")
            return filepath

        except ImportError:
            logger.warning("pandas 未安装，无法导出 Excel 格式")
            return self._export_to_csv(tasks, filename)

    def _export_to_csv(self, tasks: List[TaskRecord], filename: str = None) -> str:
        """
        导出任务为 CSV 格式（Excel 不可用时的备选方案）

        Args:
            tasks: 任务记录列表
            filename: 输出文件名

        Returns:
            导出文件路径
        """
        import csv

        if not filename:
            filename = f"tasks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['任务ID', '用户输入', '状态', '优先级', '步骤数', '创建时间', '完成时间'])

            for task in tasks:
                writer.writerow([
                    task.task_id,
                    task.user_input,
                    task.status.value,
                    task.priority.name,
                    len(task.steps),
                    task.created_at,
                    task.completed_at or ""
                ])

        logger.info(f"任务导出为 CSV: {filepath}")
        return filepath

    def export_batch_to_markdown(self, tasks: List[TaskRecord], filename: str = None) -> str:
        """
        批量导出任务为单个 Markdown 文件

        Args:
            tasks: 任务记录列表
            filename: 输出文件名

        Returns:
            导出文件路径
        """
        if not filename:
            filename = f"tasks_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        filepath = os.path.join(self.output_dir, filename)

        content = []
        content.append(f"# 任务批量报告\n")
        content.append(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content.append(f"**任务总数**: {len(tasks)}\n")
        content.append(f"\n---\n")

        for i, task in enumerate(tasks, 1):
            content.append(f"\n## 任务 {i}: {task.task_id}\n")
            content.append(f"- **用户输入**: {task.user_input}\n")
            content.append(f"- **状态**: {task.status.value}\n")
            content.append(f"- **创建时间**: {task.created_at}\n")

            if task.final_result:
                content.append(f"- **结果**: {task.final_result[:200]}...\n" if len(task.final_result) > 200 else f"- **结果**: {task.final_result}\n")

            content.append(f"\n---\n")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        logger.info(f"批量任务导出为 Markdown: {filepath}")
        return filepath


class TaskManager:
    """
    任务管理器类

    综合管理任务的创建、执行监控、状态更新和导出

    Attributes:
        storage: 任务存储实例
        exporter: 任务导出实例
        active_tasks: 当前活动的任务字典
    """

    def __init__(self, db_path: str = "tasks.db", export_dir: str = "exports"):
        """
        初始化任务管理器

        Args:
            db_path: 任务数据库路径
            export_dir: 导出文件目录
        """
        self.storage = TaskStorage(db_path)
        self.exporter = TaskExporter(export_dir)
        self.active_tasks: Dict[str, TaskRecord] = {}
        self._task_counter = 0
        logger.info("任务管理器初始化完成")

    def create_task(self, user_input: str, priority: TaskPriority = TaskPriority.MEDIUM) -> TaskRecord:
        """
        创建新任务

        Args:
            user_input: 用户输入
            priority: 任务优先级

        Returns:
            创建的任务记录
        """
        self._task_counter += 1
        task_id = f"TASK_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._task_counter:04d}"

        task = TaskRecord(
            task_id=task_id,
            user_input=user_input,
            priority=priority
        )

        self.active_tasks[task_id] = task
        self.storage.save_task(task)

        logger.info(f"创建任务: {task_id}")
        return task

    def add_step(self, task_id: str, description: str, tool_name: str = None,
                 params: Dict[str, Any] = None) -> Optional[TaskStep]:
        """
        为任务添加执行步骤

        Args:
            task_id: 任务ID
            description: 步骤描述
            tool_name: 使用的工具名称
            params: 工具参数

        Returns:
            创建的步骤对象
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return None

        step_order = len(task.steps) + 1
        step_id = f"{task_id}_STEP_{step_order:02d}"

        step = TaskStep(
            step_id=step_id,
            step_order=step_order,
            description=description,
            tool_name=tool_name,
            params=params or {}
        )

        task.steps.append(step)
        self.storage.save_task(task)

        logger.info(f"添加步骤: {step_id}")
        return step

    def update_task_status(self, task_id: str, status: TaskStatus,
                           error_message: str = None) -> bool:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            error_message: 错误信息（可选）

        Returns:
            是否更新成功
        """
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = status
        task.updated_at = datetime.now().isoformat()

        if error_message:
            task.error_message = error_message

        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now().isoformat()
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

        self.storage.save_task(task)
        logger.info(f"任务状态更新: {task_id} -> {status.value}")
        return True

    def update_step_status(self, task_id: str, step_id: str, status: TaskStatus,
                           result: Dict[str, Any] = None, error_message: str = None) -> bool:
        """
        更新步骤状态

        Args:
            task_id: 任务ID
            step_id: 步骤ID
            status: 新状态
            result: 执行结果
            error_message: 错误信息

        Returns:
            是否更新成功
        """
        task = self.get_task(task_id)
        if not task:
            return False

        for step in task.steps:
            if step.step_id == step_id:
                step.status = status

                if status == TaskStatus.EXECUTING:
                    step.started_at = datetime.now().isoformat()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    step.completed_at = datetime.now().isoformat()

                if result:
                    step.result = result
                if error_message:
                    step.error_message = error_message

                self.storage.save_task(task)
                logger.info(f"步骤状态更新: {step_id} -> {status.value}")
                return True

        return False

    def set_final_result(self, task_id: str, result: str) -> bool:
        """
        设置任务最终结果

        Args:
            task_id: 任务ID
            result: 最终结果

        Returns:
            是否设置成功
        """
        task = self.get_task(task_id)
        if not task:
            return False

        task.final_result = result
        task.updated_at = datetime.now().isoformat()
        self.storage.save_task(task)

        logger.info(f"设置任务结果: {task_id}")
        return True

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """
        获取任务记录

        Args:
            task_id: 任务ID

        Returns:
            任务记录对象
        """
        # 先从活动任务中查找
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        # 从存储中查找
        return self.storage.get_task(task_id)

    def get_pending_tasks(self) -> List[TaskRecord]:
        """
        获取待执行的任务列表

        Returns:
            待执行任务列表
        """
        return self.storage.get_tasks_by_status(TaskStatus.PENDING)

    def get_executing_tasks(self) -> List[TaskRecord]:
        """
        获取执行中的任务列表

        Returns:
            执行中任务列表
        """
        return list(self.active_tasks.values())

    def get_recent_tasks(self, limit: int = 20) -> List[TaskRecord]:
        """
        获取最近的任务列表

        Args:
            limit: 返回数量限制

        Returns:
            任务列表
        """
        return self.storage.get_recent_tasks(limit)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取任务统计信息

        Returns:
            统计信息字典
        """
        stats = self.storage.get_statistics()
        stats["active"] = len(self.active_tasks)
        return stats

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID

        Returns:
            是否删除成功
        """
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

        return self.storage.delete_task(task_id)

    def export_task_markdown(self, task_id: str) -> Optional[str]:
        """
        导出单个任务为 Markdown

        Args:
            task_id: 任务ID

        Returns:
            导出文件路径
        """
        task = self.get_task(task_id)
        if not task:
            return None

        return self.exporter.export_to_markdown(task)

    def export_tasks_excel(self, task_ids: List[str] = None) -> Optional[str]:
        """
        导出任务列表为 Excel

        Args:
            task_ids: 任务ID列表，为空则导出最近的任务

        Returns:
            导出文件路径
        """
        if task_ids:
            tasks = [self.get_task(tid) for tid in task_ids if self.get_task(tid)]
        else:
            tasks = self.get_recent_tasks(50)

        if not tasks:
            return None

        return self.exporter.export_to_excel(tasks)

    def export_batch_markdown(self, task_ids: List[str] = None) -> Optional[str]:
        """
        批量导出任务为 Markdown

        Args:
            task_ids: 任务ID列表

        Returns:
            导出文件路径
        """
        if task_ids:
            tasks = [self.get_task(tid) for tid in task_ids if self.get_task(tid)]
        else:
            tasks = self.get_recent_tasks(50)

        if not tasks:
            return None

        return self.exporter.export_batch_to_markdown(tasks)


# 便捷函数：创建任务管理器实例
def create_task_manager(db_path: str = "tasks.db", export_dir: str = "exports") -> TaskManager:
    """
    创建任务管理器实例

    Args:
        db_path: 数据库路径
        export_dir: 导出目录

    Returns:
        TaskManager 实例
    """
    return TaskManager(db_path, export_dir)


# 测试代码
if __name__ == "__main__":
    # 创建任务管理器
    tm = create_task_manager()

    print("=" * 60)
    print("任务管理模块测试")
    print("=" * 60)

    # 创建测试任务
    print("\n创建测试任务...")
    task1 = tm.create_task("提取test.pdf中的表格并计算均值")
    print(f"  创建任务: {task1.task_id}")

    # 添加步骤
    print("\n添加执行步骤...")
    step1 = tm.add_step(task1.task_id, "提取PDF表格", "file_tool", {"action": "extract_tables"})
    step2 = tm.add_step(task1.task_id, "计算均值", "data_tool", {"action": "mean"})
    print(f"  添加步骤: {step1.step_id}, {step2.step_id}")

    # 更新状态
    print("\n更新任务状态...")
    tm.update_task_status(task1.task_id, TaskStatus.EXECUTING)
    tm.update_step_status(task1.task_id, step1.step_id, TaskStatus.COMPLETED,
                          {"success": True, "data": {"tables": 1}})
    tm.update_step_status(task1.task_id, step2.step_id, TaskStatus.COMPLETED,
                          {"success": True, "data": {"mean": 170.0}})

    # 设置结果
    tm.set_final_result(task1.task_id, "成功提取1个表格，计算得到均值为170.0")
    tm.update_task_status(task1.task_id, TaskStatus.COMPLETED)

    # 获取统计信息
    print("\n任务统计信息:")
    stats = tm.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 导出任务
    print("\n导出任务...")
    md_path = tm.export_task_markdown(task1.task_id)
    print(f"  Markdown导出: {md_path}")

    print("\n" + "=" * 60)
    print("测试完成")
