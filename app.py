#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web 交互界面模块
================

本模块使用 Streamlit 实现智能 Agent 平台的 Web 交互界面，包括：
1. 任务输入框（支持文本+文件上传）
2. 工具开关（可选择启用/禁用工具）
3. 任务执行进度展示
4. 结果展示区（文本+图片+文件下载）
5. 历史任务列表

作者：学生开发团队
版本：1.0.0

运行方式：
    streamlit run app.py
"""

import os
import sys
import json
import time
import base64
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

# Streamlit 界面库
import streamlit as st

# 导入自定义模块
from agent_core import TaskAgent, create_agent, TaskStatus
from tool_pool import ToolManager, create_tool_manager
from task_manager import TaskManager, create_task_manager, TaskStatus as TMTaskStatus, TaskPriority


# ============================================================
# 页面配置和样式
# ============================================================

def setup_page():
    """配置页面基本设置"""
    st.set_page_config(
        page_title="智能任务处理 Agent 平台",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义 CSS 样式
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .status-pending {
            color: #ff9800;
            font-weight: bold;
        }
        .status-executing {
            color: #2196f3;
            font-weight: bold;
        }
        .status-completed {
            color: #4caf50;
            font-weight: bold;
        }
        .status-failed {
            color: #f44336;
            font-weight: bold;
        }
        .tool-card {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .step-card {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 5px 5px 0;
        }
        .result-card {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
        .error-card {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'agent' not in st.session_state:
        st.session_state.agent = create_agent()

    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = create_tool_manager("data/schedules.db")
        st.session_state.agent.set_tool_manager(st.session_state.tool_manager)

    if 'task_manager' not in st.session_state:
        st.session_state.task_manager = create_task_manager("data/tasks.db", "exports")

    if 'current_task' not in st.session_state:
        st.session_state.current_task = None

    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    # 确保数据目录存在
    os.makedirs("data", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)


# ============================================================
# 侧边栏组件
# ============================================================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🛠️ 工具管理")

        # 获取工具列表
        tools = st.session_state.tool_manager.list_tools()

        # 工具开关
        st.markdown("### 启用/禁用工具")
        enabled_tools = []

        for tool in tools:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{get_tool_display_name(tool['name'])}**")
            with col2:
                is_enabled = st.checkbox(
                    "启用",
                    value=tool['enabled'],
                    key=f"tool_{tool['name']}",
                    label_visibility="collapsed"
                )
                if is_enabled:
                    enabled_tools.append(tool['name'])

        # 更新工具管理器的启用状态
        st.session_state.tool_manager.set_enabled_tools(enabled_tools)

        st.markdown("---")

        # 任务统计
        st.markdown("## 📊 任务统计")
        stats = st.session_state.task_manager.get_statistics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("总任务数", stats.get('total', 0))
            st.metric("已完成", stats.get('completed', 0))
        with col2:
            st.metric("今日任务", stats.get('today', 0))
            st.metric("失败", stats.get('failed', 0))

        st.markdown("---")

        # 快捷操作
        st.markdown("## ⚡ 快捷操作")

        if st.button("📥 导出历史任务", use_container_width=True):
            export_path = st.session_state.task_manager.export_batch_markdown()
            if export_path:
                st.success(f"导出成功: {export_path}")
            else:
                st.warning("没有可导出的任务")

        if st.button("🗑️ 清空执行日志", use_container_width=True):
            st.session_state.execution_log = []
            st.success("日志已清空")


def get_tool_display_name(tool_name: str) -> str:
    """获取工具的显示名称"""
    display_names = {
        "file_tool": "📄 文件处理",
        "data_tool": "📊 数据分析",
        "code_tool": "💻 代码运行",
        "paper_tool": "📚 文献查询",
        "schedule_tool": "📅 日程管理",
        "translate_tool": "🌐 文本翻译",
        "summary_tool": "📝 文本摘要",
        "qa_tool": "❓ 知识问答"
    }
    return display_names.get(tool_name, tool_name)


# ============================================================
# 主界面组件
# ============================================================

def render_header():
    """渲染页面头部"""
    st.markdown('<div class="main-header">🤖 智能任务处理 Agent 平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">输入自然语言任务，AI自动规划执行</div>', unsafe_allow_html=True)


def render_task_input():
    """渲染任务输入区域"""
    st.markdown("### 📝 任务输入")

    # 文本输入
    task_text = st.text_area(
        "请描述您的任务：",
        placeholder="例如：提取test.pdf中的表格并计算均值，然后画柱状图",
        height=100,
        key="task_input"
    )

    # 文件上传
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "上传文件（可选）",
            type=["pdf", "xlsx", "xls", "csv", "docx", "txt", "py"],
            accept_multiple_files=True,
            key="file_uploader"
        )

    with col2:
        priority = st.selectbox(
            "任务优先级",
            options=["普通", "高", "紧急"],
            index=0,
            key="task_priority"
        )

    # 处理上传的文件
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.uploaded_files[uploaded_file.name] = file_path
            st.info(f"✅ 已上传: {uploaded_file.name}")

    # 执行按钮
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        execute_button = st.button("🚀 执行任务", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🔄 清空", use_container_width=True)

    if clear_button:
        st.session_state.current_task = None
        st.session_state.execution_log = []
        st.session_state.uploaded_files = {}
        st.rerun()

    if execute_button and task_text.strip():
        execute_task(task_text.strip(), priority)


def save_uploaded_file(uploaded_file) -> str:
    """保存上传的文件"""
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def execute_task(task_text: str, priority: str):
    """执行任务"""
    # 创建任务记录
    priority_map = {
        "普通": TaskPriority.MEDIUM,
        "高": TaskPriority.HIGH,
        "紧急": TaskPriority.URGENT
    }
    task_priority = priority_map.get(priority, TaskPriority.MEDIUM)

    task_record = st.session_state.task_manager.create_task(task_text, task_priority)
    st.session_state.current_task = task_record

    # 添加执行日志
    log_message(f"📋 创建任务: {task_record.task_id}")
    log_message(f"📝 任务内容: {task_text}")

    # 替换任务中的文件路径
    processed_task = task_text
    for filename, filepath in st.session_state.uploaded_files.items():
        if filename in processed_task:
            processed_task = processed_task.replace(filename, filepath)

    # 获取启用的工具列表
    enabled_tools = st.session_state.tool_manager.enabled_tools

    # 执行任务
    st.session_state.task_manager.update_task_status(
        task_record.task_id,
        TMTaskStatus.EXECUTING
    )

    log_message("🔄 开始解析和执行任务...")

    # 调用 Agent 处理任务
    result = st.session_state.agent.process_task(processed_task, enabled_tools)

    if result["success"]:
        st.session_state.task_manager.update_task_status(
            task_record.task_id,
            TMTaskStatus.COMPLETED
        )
        st.session_state.task_manager.set_final_result(
            task_record.task_id,
            result.get("final_result", "")
        )
        log_message("✅ 任务执行完成")
    else:
        st.session_state.task_manager.update_task_status(
            task_record.task_id,
            TMTaskStatus.FAILED,
            result.get("error", "未知错误")
        )
        log_message(f"❌ 任务执行失败: {result.get('error', '未知错误')}")

    # 保存结果到当前任务
    st.session_state.current_task = st.session_state.task_manager.get_task(task_record.task_id)
    st.session_state.current_result = result


def log_message(message: str):
    """添加日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.execution_log.append(f"[{timestamp}] {message}")


def render_execution_progress():
    """渲染执行进度"""
    st.markdown("### ⏳ 执行进度")

    if not st.session_state.execution_log:
        st.info("暂无执行记录，请输入任务并点击执行")
        return

    # 显示执行日志
    log_container = st.container()
    with log_container:
        for log in st.session_state.execution_log[-10:]:  # 显示最近10条
            st.text(log)

    # 如果有当前任务，显示步骤详情
    if hasattr(st.session_state, 'current_result') and st.session_state.current_result:
        result = st.session_state.current_result
        if result.get("plan"):
            plan = result["plan"]
            st.markdown("#### 执行步骤详情")

            for sub_task in plan.sub_tasks:
                status_icon = get_status_icon(sub_task.status)
                with st.expander(f"{status_icon} 步骤 {sub_task.order}: {sub_task.description}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**工具**: {get_tool_display_name(sub_task.matched_tool.value) if sub_task.matched_tool else '无'}")
                    with col2:
                        st.write(f"**状态**: {sub_task.status.value}")

                    if sub_task.params:
                        st.write("**参数**:")
                        st.json(sub_task.params)

                    if sub_task.result:
                        st.write("**结果**:")
                        st.json(sub_task.result)


def get_status_icon(status) -> str:
    """获取状态图标"""
    status_icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.PARSING: "🔍",
        TaskStatus.EXECUTING: "🔄",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌"
    }
    return status_icons.get(status, "❓")


def render_result_area():
    """渲染结果展示区域"""
    st.markdown("### 📊 执行结果")

    if not hasattr(st.session_state, 'current_result') or not st.session_state.current_result:
        st.info("执行任务后，结果将在此显示")
        return

    result = st.session_state.current_result

    if result["success"]:
        # 成功结果
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### ✅ 任务执行成功")
        st.markdown(result.get("final_result", ""))
        st.markdown('</div>', unsafe_allow_html=True)

        # 检查是否有生成的图片
        if result.get("execution_result"):
            for sub_result in result["execution_result"].get("sub_task_results", []):
                if sub_result.get("result", {}).get("data", {}).get("output_path"):
                    output_path = sub_result["result"]["data"]["output_path"]
                    if os.path.exists(output_path) and output_path.endswith(('.png', '.jpg', '.jpeg')):
                        st.image(output_path, caption="生成的图表")

                        # 提供下载按钮
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 下载图表",
                                data=f,
                                file_name=os.path.basename(output_path),
                                mime="image/png"
                            )

    else:
        # 失败结果
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        st.markdown("#### ❌ 任务执行失败")
        st.error(result.get("error", "未知错误"))
        st.markdown('</div>', unsafe_allow_html=True)

    # 导出按钮
    if st.session_state.current_task:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 导出为 Markdown", use_container_width=True):
                export_path = st.session_state.task_manager.export_task_markdown(
                    st.session_state.current_task.task_id
                )
                if export_path:
                    st.success(f"导出成功: {export_path}")
                    with open(export_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="📥 下载报告",
                            data=f.read(),
                            file_name=os.path.basename(export_path),
                            mime="text/markdown"
                        )


def render_history():
    """渲染历史任务列表"""
    st.markdown("### 📜 历史任务")

    tasks = st.session_state.task_manager.get_recent_tasks(10)

    if not tasks:
        st.info("暂无历史任务")
        return

    for task in tasks:
        status_icon = "✅" if task.status == TMTaskStatus.COMPLETED else "❌" if task.status == TMTaskStatus.FAILED else "⏳"
        with st.expander(f"{status_icon} {task.task_id} - {task.user_input[:50]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**状态**: {task.status.value}")
                st.write(f"**创建时间**: {task.created_at}")
            with col2:
                st.write(f"**优先级**: {task.priority.name}")
                st.write(f"**步骤数**: {len(task.steps)}")

            if task.final_result:
                st.write("**结果**:")
                st.write(task.final_result[:200] + "..." if len(task.final_result) > 200 else task.final_result)

            # 操作按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📄 导出", key=f"export_{task.task_id}"):
                    export_path = st.session_state.task_manager.export_task_markdown(task.task_id)
                    if export_path:
                        st.success(f"导出成功")
            with col2:
                if st.button(f"🗑️ 删除", key=f"delete_{task.task_id}"):
                    st.session_state.task_manager.delete_task(task.task_id)
                    st.rerun()


# ============================================================
# 示例任务展示
# ============================================================

def render_examples():
    """渲染示例任务"""
    st.markdown("### 💡 示例任务")

    examples = [
        {
            "title": "📊 数据分析",
            "task": "提取test.pdf中的表格并计算均值",
            "description": "从PDF文件中提取表格数据，并进行统计分析"
        },
        {
            "title": "📈 图表生成",
            "task": "读取data.csv数据并画柱状图",
            "description": "读取CSV数据文件，生成可视化图表"
        },
        {
            "title": "📚 文献搜索",
            "task": "搜索机器学习相关文献",
            "description": "在本地文献库中搜索相关论文"
        },
        {
            "title": "📅 日程管理",
            "task": "添加明天上午10:00的会议日程",
            "description": "创建新的日程提醒"
        },
        {
            "title": "🌐 文本翻译",
            "task": "翻译机器学习为英文",
            "description": "将中文文本翻译为英文或其他语言"
        },
        {
            "title": "📝 文本摘要",
            "task": "总结这篇文章的主要内容",
            "description": "自动生成文本摘要和关键词提取"
        },
        {
            "title": "❓ 知识问答",
            "task": "什么是深度学习",
            "description": "查询编程、AI、数学等领域的知识"
        },
        {
            "title": "💻 代码执行",
            "task": "运行代码 print('Hello, Agent!')",
            "description": "在安全沙箱中执行Python代码"
        }
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{example['title']}**")
                st.caption(example['description'])
                if st.button(f"使用此示例", key=f"example_{i}"):
                    st.session_state.task_input = example['task']
                    st.rerun()


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    # 页面配置
    setup_page()

    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 渲染主界面
    render_header()

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 任务输入
        render_task_input()

        st.markdown("---")

        # 执行进度
        render_execution_progress()

        st.markdown("---")

        # 结果展示
        render_result_area()

    with col2:
        # 示例任务
        render_examples()

        st.markdown("---")

        # 历史任务
        render_history()


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
