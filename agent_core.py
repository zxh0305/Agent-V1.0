#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 核心引擎模块
==================

本模块实现了智能任务处理 Agent 的核心功能，包括：
1. 任务解析：将自然语言任务拆分为可执行子步骤
2. 工具匹配：根据子步骤关键词匹配工具池中的工具
3. 执行规划：确定工具调用顺序（支持串行执行）
4. 结果整合：将工具执行结果汇总为自然语言回答

作者：学生开发团队
版本：1.0.0
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举类"""
    PENDING = "待执行"
    PARSING = "解析中"
    EXECUTING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"


class ToolType(Enum):
    """工具类型枚举类"""
    FILE_TOOL = "file_tool"       # 文件处理工具
    DATA_TOOL = "data_tool"       # 数据分析工具
    CODE_TOOL = "code_tool"       # 代码运行工具
    PAPER_TOOL = "paper_tool"     # 文献查询工具
    SCHEDULE_TOOL = "schedule_tool"  # 日程管理工具


@dataclass
class SubTask:
    """
    子任务数据类

    用于存储解析后的单个子任务信息

    属性:
        task_id: 子任务唯一标识
        description: 子任务描述
        matched_tool: 匹配的工具类型
        params: 工具调用参数
        status: 子任务状态
        result: 执行结果
        order: 执行顺序
    """
    task_id: str
    description: str
    matched_tool: Optional[ToolType] = None
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    order: int = 0


@dataclass
class TaskPlan:
    """
    任务执行计划数据类

    用于存储完整的任务执行计划

    属性:
        plan_id: 计划唯一标识
        original_task: 原始任务描述
        sub_tasks: 子任务列表
        status: 计划状态
        created_at: 创建时间
        completed_at: 完成时间
        final_result: 最终结果
    """
    plan_id: str
    original_task: str
    sub_tasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    final_result: Optional[str] = None


class KeywordMatcher:
    """
    关键词匹配器类

    实现基于规则的关键词匹配，用于将自然语言任务映射到具体工具
    这是本系统的核心原创逻辑之一
    """

    # 工具关键词映射表（核心原创配置）
    TOOL_KEYWORDS = {
        ToolType.FILE_TOOL: {
            "primary": ["pdf", "文件", "文档", "word", "docx", "转换", "提取文本", "读取文件"],
            "secondary": ["打开", "保存", "导出", "表格提取"],
            "weight": 1.0
        },
        ToolType.DATA_TOOL: {
            "primary": ["数据", "excel", "csv", "计算", "分析", "统计", "均值", "方差",
                       "图表", "柱状图", "折线图", "饼图", "画图", "绘图", "可视化"],
            "secondary": ["求和", "平均", "最大", "最小", "排序", "筛选"],
            "weight": 1.0
        },
        ToolType.CODE_TOOL: {
            "primary": ["代码", "运行", "执行", "python", "程序", "脚本", "编程"],
            "secondary": ["调试", "测试", "函数"],
            "weight": 0.9
        },
        ToolType.PAPER_TOOL: {
            "primary": ["文献", "论文", "期刊", "学术", "搜索文献", "查找论文", "引用"],
            "secondary": ["研究", "参考", "摘要", "关键词搜索"],
            "weight": 0.8
        },
        ToolType.SCHEDULE_TOOL: {
            "primary": ["日程", "提醒", "安排", "日历", "待办", "计划", "会议"],
            "secondary": ["时间", "预约", "任务列表", "添加日程", "查询日程", "删除日程"],
            "weight": 0.7
        }
    }

    # 动作关键词映射表
    ACTION_KEYWORDS = {
        "extract": ["提取", "获取", "读取", "抽取", "解析"],
        "convert": ["转换", "转化", "变换", "导出为"],
        "calculate": ["计算", "求", "统计", "分析"],
        "visualize": ["画", "绘制", "生成图", "可视化", "展示"],
        "search": ["搜索", "查找", "查询", "检索"],
        "add": ["添加", "新增", "创建", "设置"],
        "delete": ["删除", "移除", "取消"],
        "run": ["运行", "执行", "启动"]
    }

    def __init__(self):
        """初始化关键词匹配器"""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """
        编译正则表达式模式

        将关键词列表编译为正则表达式，提高匹配效率
        """
        self.compiled_patterns = {}
        for tool_type, keywords_config in self.TOOL_KEYWORDS.items():
            all_keywords = keywords_config["primary"] + keywords_config["secondary"]
            pattern = "|".join(re.escape(kw) for kw in all_keywords)
            self.compiled_patterns[tool_type] = re.compile(pattern, re.IGNORECASE)

    def match_tool(self, text: str) -> Tuple[Optional[ToolType], float]:
        """
        匹配文本中的工具类型

        使用加权评分机制，主关键词权重更高

        Args:
            text: 待匹配的文本

        Returns:
            匹配的工具类型和置信度分数
        """
        scores = {}
        text_lower = text.lower()

        for tool_type, keywords_config in self.TOOL_KEYWORDS.items():
            score = 0.0

            # 主关键词匹配（权重2.0）
            for keyword in keywords_config["primary"]:
                if keyword.lower() in text_lower:
                    score += 2.0

            # 次关键词匹配（权重1.0）
            for keyword in keywords_config["secondary"]:
                if keyword.lower() in text_lower:
                    score += 1.0

            # 应用工具权重
            score *= keywords_config["weight"]
            scores[tool_type] = score

        # 找出最高分的工具
        if scores:
            best_tool = max(scores, key=scores.get)
            best_score = scores[best_tool]
            if best_score > 0:
                # 归一化置信度（0-1范围）
                confidence = min(best_score / 10.0, 1.0)
                return best_tool, confidence

        return None, 0.0

    def extract_action(self, text: str) -> Optional[str]:
        """
        提取文本中的动作类型

        Args:
            text: 待分析的文本

        Returns:
            识别出的动作类型
        """
        text_lower = text.lower()
        for action, keywords in self.ACTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return action
        return None


class TaskParser:
    """
    任务解析器类

    负责将自然语言任务拆分为可执行的子步骤
    这是本系统的核心原创逻辑之一
    """

    # 分隔符模式
    SEPARATOR_PATTERNS = [
        r'[，,]\s*(?:然后|接着|再|并且|同时)',  # 中文分隔
        r'[，,]\s*(?:and|then|also|next)',       # 英文分隔
        r'[。；;]\s*',                            # 标点分隔
        r'\s+(?:然后|接着|再|并且)\s+',           # 连词分隔
        r'\s+(?:and then|then|next)\s+',        # 英文连词
    ]

    # 任务模板模式（用于识别常见任务结构）
    TASK_TEMPLATES = {
        "extract_and_process": r'(?:提取|获取|读取)(.+?)(?:并|然后|再)(.+)',
        "convert_format": r'(?:将|把)(.+?)(?:转换|转化|导出)(?:为|成)(.+)',
        "calculate_data": r'(?:计算|统计|分析)(.+?)(?:的)?(.+)',
        "visualize_data": r'(?:画|绘制|生成)(.+?)(?:图|表)',
        "search_content": r'(?:搜索|查找|检索)(.+)',
        "schedule_action": r'(?:添加|查询|删除)(.+?)(?:日程|提醒|安排)',
    }

    def __init__(self):
        """初始化任务解析器"""
        self.keyword_matcher = KeywordMatcher()
        self._compile_templates()

    def _compile_templates(self) -> None:
        """编译任务模板正则表达式"""
        self.compiled_templates = {}
        for name, pattern in self.TASK_TEMPLATES.items():
            self.compiled_templates[name] = re.compile(pattern, re.IGNORECASE)

    def parse_task(self, task_text: str) -> List[Dict[str, Any]]:
        """
        解析自然语言任务为子步骤列表

        核心解析算法：
        1. 首先尝试模板匹配
        2. 然后尝试分隔符拆分
        3. 最后进行关键词提取

        Args:
            task_text: 原始任务描述

        Returns:
            解析后的子步骤列表，每个元素包含描述、工具类型和参数
        """
        logger.info(f"开始解析任务: {task_text}")

        # 预处理文本
        task_text = self._preprocess_text(task_text)

        # 尝试模板匹配
        template_result = self._try_template_match(task_text)
        if template_result:
            logger.info(f"模板匹配成功，识别出 {len(template_result)} 个子步骤")
            return template_result

        # 尝试分隔符拆分
        segments = self._split_by_separators(task_text)

        # 为每个段落匹配工具和提取参数
        sub_steps = []
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue

            tool_type, confidence = self.keyword_matcher.match_tool(segment)
            action = self.keyword_matcher.extract_action(segment)
            params = self._extract_params(segment, tool_type)

            sub_step = {
                "order": i + 1,
                "description": segment,
                "tool_type": tool_type,
                "action": action,
                "params": params,
                "confidence": confidence
            }
            sub_steps.append(sub_step)

        logger.info(f"解析完成，共识别出 {len(sub_steps)} 个子步骤")
        return sub_steps

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        # 统一标点符号
        text = text.replace('，', ',').replace('。', '.').replace('；', ';')
        return text

    def _try_template_match(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        尝试使用模板匹配任务

        Args:
            text: 任务文本

        Returns:
            匹配成功返回子步骤列表，否则返回None
        """
        for template_name, pattern in self.compiled_templates.items():
            match = pattern.search(text)
            if match:
                return self._process_template_match(template_name, match, text)
        return None

    def _process_template_match(self, template_name: str, match: re.Match,
                                 original_text: str) -> List[Dict[str, Any]]:
        """
        处理模板匹配结果

        Args:
            template_name: 模板名称
            match: 正则匹配对象
            original_text: 原始文本

        Returns:
            子步骤列表
        """
        sub_steps = []
        groups = match.groups()

        if template_name == "extract_and_process":
            # 提取并处理模式
            if len(groups) >= 2:
                # 第一步：提取
                tool1, conf1 = self.keyword_matcher.match_tool(groups[0])
                sub_steps.append({
                    "order": 1,
                    "description": f"提取{groups[0]}",
                    "tool_type": tool1 or ToolType.FILE_TOOL,
                    "action": "extract",
                    "params": self._extract_params(groups[0], tool1),
                    "confidence": conf1
                })
                # 第二步：处理
                tool2, conf2 = self.keyword_matcher.match_tool(groups[1])
                sub_steps.append({
                    "order": 2,
                    "description": groups[1],
                    "tool_type": tool2 or ToolType.DATA_TOOL,
                    "action": self.keyword_matcher.extract_action(groups[1]),
                    "params": self._extract_params(groups[1], tool2),
                    "confidence": conf2
                })

        elif template_name == "convert_format":
            # 格式转换模式
            tool_type, confidence = self.keyword_matcher.match_tool(original_text)
            sub_steps.append({
                "order": 1,
                "description": original_text,
                "tool_type": tool_type or ToolType.FILE_TOOL,
                "action": "convert",
                "params": {
                    "source": groups[0] if groups else "",
                    "target_format": groups[1] if len(groups) > 1 else ""
                },
                "confidence": confidence
            })

        elif template_name in ["calculate_data", "visualize_data"]:
            # 数据计算或可视化模式
            action = "calculate" if template_name == "calculate_data" else "visualize"
            sub_steps.append({
                "order": 1,
                "description": original_text,
                "tool_type": ToolType.DATA_TOOL,
                "action": action,
                "params": self._extract_params(original_text, ToolType.DATA_TOOL),
                "confidence": 0.8
            })

        elif template_name == "search_content":
            # 搜索模式
            sub_steps.append({
                "order": 1,
                "description": original_text,
                "tool_type": ToolType.PAPER_TOOL,
                "action": "search",
                "params": {"query": groups[0] if groups else ""},
                "confidence": 0.7
            })

        elif template_name == "schedule_action":
            # 日程管理模式
            action = self.keyword_matcher.extract_action(original_text)
            sub_steps.append({
                "order": 1,
                "description": original_text,
                "tool_type": ToolType.SCHEDULE_TOOL,
                "action": action,
                "params": self._extract_params(original_text, ToolType.SCHEDULE_TOOL),
                "confidence": 0.8
            })

        return sub_steps if sub_steps else None

    def _split_by_separators(self, text: str) -> List[str]:
        """
        使用分隔符拆分文本

        Args:
            text: 待拆分的文本

        Returns:
            拆分后的文本段落列表
        """
        segments = [text]

        for pattern in self.SEPARATOR_PATTERNS:
            new_segments = []
            for segment in segments:
                parts = re.split(pattern, segment)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments

        return segments if len(segments) > 1 else [text]

    def _extract_params(self, text: str, tool_type: Optional[ToolType]) -> Dict[str, Any]:
        """
        从文本中提取工具参数

        Args:
            text: 任务描述文本
            tool_type: 工具类型

        Returns:
            提取的参数字典
        """
        params = {}

        # 提取文件名/路径
        file_patterns = [
            r'([a-zA-Z0-9_\-]+\.(pdf|xlsx|xls|csv|docx|doc|txt|py))',
            r'[\u4e00-\u9fa5a-zA-Z0-9_\-]+\.(pdf|xlsx|xls|csv|docx|doc|txt|py)',
        ]
        for pattern in file_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                params["file_path"] = match.group(0)
                break

        # 根据工具类型提取特定参数
        if tool_type == ToolType.DATA_TOOL:
            # 提取数据分析相关参数
            if "均值" in text or "平均" in text:
                params["operation"] = "mean"
            elif "方差" in text:
                params["operation"] = "variance"
            elif "求和" in text or "总和" in text:
                params["operation"] = "sum"
            elif "最大" in text:
                params["operation"] = "max"
            elif "最小" in text:
                params["operation"] = "min"

            # 提取图表类型
            if "柱状图" in text:
                params["chart_type"] = "bar"
            elif "折线图" in text:
                params["chart_type"] = "line"
            elif "饼图" in text:
                params["chart_type"] = "pie"
            elif "散点图" in text:
                params["chart_type"] = "scatter"

        elif tool_type == ToolType.SCHEDULE_TOOL:
            # 提取日期时间
            date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
            time_pattern = r'(\d{1,2}:\d{2})'

            date_match = re.search(date_pattern, text)
            time_match = re.search(time_pattern, text)

            if date_match:
                params["date"] = date_match.group(1)
            if time_match:
                params["time"] = time_match.group(1)

            # 提取日程内容
            content_patterns = [
                r'(?:日程|提醒|安排)[：:]\s*(.+)',
                r'(?:内容)[：:]\s*(.+)',
            ]
            for pattern in content_patterns:
                match = re.search(pattern, text)
                if match:
                    params["content"] = match.group(1).strip()
                    break

        elif tool_type == ToolType.PAPER_TOOL:
            # 提取搜索关键词
            params["query"] = text

        return params


class TaskAgent:
    """
    智能任务处理 Agent 核心类

    这是本系统的核心类，负责协调任务解析、工具匹配、执行规划和结果整合

    主要功能：
    1. 接收自然语言任务输入
    2. 调用 TaskParser 解析任务
    3. 生成执行计划
    4. 调用 ToolManager 执行工具
    5. 整合并返回执行结果

    Attributes:
        parser: 任务解析器实例
        tool_manager: 工具管理器实例（需外部注入）
        current_plan: 当前执行计划
        execution_history: 执行历史记录
    """

    def __init__(self, tool_manager=None):
        """
        初始化 TaskAgent

        Args:
            tool_manager: 工具管理器实例，可选
        """
        self.parser = TaskParser()
        self.tool_manager = tool_manager
        self.current_plan: Optional[TaskPlan] = None
        self.execution_history: List[TaskPlan] = []
        self._plan_counter = 0
        logger.info("TaskAgent 初始化完成")

    def set_tool_manager(self, tool_manager) -> None:
        """
        设置工具管理器

        Args:
            tool_manager: 工具管理器实例
        """
        self.tool_manager = tool_manager
        logger.info("工具管理器已设置")

    def process_task(self, task_text: str, enabled_tools: List[str] = None) -> Dict[str, Any]:
        """
        处理用户任务的主入口方法

        完整的任务处理流程：
        1. 解析任务
        2. 创建执行计划
        3. 执行计划
        4. 整合结果

        Args:
            task_text: 用户输入的自然语言任务
            enabled_tools: 启用的工具列表（可选）

        Returns:
            处理结果字典，包含状态、计划详情和最终结果
        """
        logger.info(f"开始处理任务: {task_text}")

        try:
            # 步骤1：解析任务
            sub_steps = self.parser.parse_task(task_text)

            if not sub_steps:
                return {
                    "success": False,
                    "error": "无法解析任务，请重新描述您的需求",
                    "plan": None
                }

            # 步骤2：创建执行计划
            plan = self._create_execution_plan(task_text, sub_steps, enabled_tools)
            self.current_plan = plan

            # 步骤3：执行计划
            execution_result = self._execute_plan(plan)

            # 步骤4：整合结果
            final_result = self._integrate_results(plan)

            # 保存到历史记录
            self.execution_history.append(plan)

            return {
                "success": True,
                "plan": plan,
                "execution_result": execution_result,
                "final_result": final_result
            }

        except Exception as e:
            logger.error(f"任务处理出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "plan": self.current_plan
            }

    def _create_execution_plan(self, original_task: str, sub_steps: List[Dict],
                                enabled_tools: List[str] = None) -> TaskPlan:
        """
        创建任务执行计划

        Args:
            original_task: 原始任务描述
            sub_steps: 解析后的子步骤列表
            enabled_tools: 启用的工具列表

        Returns:
            TaskPlan 对象
        """
        self._plan_counter += 1
        plan_id = f"PLAN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._plan_counter}"

        sub_tasks = []
        for step in sub_steps:
            tool_type = step.get("tool_type")

            # 检查工具是否启用
            if enabled_tools and tool_type:
                if tool_type.value not in enabled_tools:
                    logger.warning(f"工具 {tool_type.value} 未启用，跳过步骤: {step['description']}")
                    continue

            sub_task = SubTask(
                task_id=f"{plan_id}_STEP_{step['order']}",
                description=step["description"],
                matched_tool=tool_type,
                params=step.get("params", {}),
                status=TaskStatus.PENDING,
                order=step["order"]
            )
            sub_tasks.append(sub_task)

        plan = TaskPlan(
            plan_id=plan_id,
            original_task=original_task,
            sub_tasks=sub_tasks,
            status=TaskStatus.PENDING
        )

        logger.info(f"执行计划创建完成: {plan_id}, 包含 {len(sub_tasks)} 个子任务")
        return plan

    def _execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        执行任务计划

        按顺序执行每个子任务，支持任务间结果传递

        Args:
            plan: 执行计划对象

        Returns:
            执行结果汇总
        """
        plan.status = TaskStatus.EXECUTING
        results = []
        previous_result = None

        for sub_task in sorted(plan.sub_tasks, key=lambda x: x.order):
            logger.info(f"执行子任务: {sub_task.task_id} - {sub_task.description}")
            sub_task.status = TaskStatus.EXECUTING

            try:
                # 准备参数，注入上一步的结果
                params = sub_task.params.copy()
                if previous_result:
                    params["previous_result"] = previous_result

                # 调用工具执行
                if self.tool_manager and sub_task.matched_tool:
                    tool_name = sub_task.matched_tool.value
                    result = self.tool_manager.run_tool(tool_name, params)
                else:
                    # 无工具管理器时的模拟执行
                    result = self._mock_execute(sub_task)

                sub_task.result = result
                sub_task.status = TaskStatus.COMPLETED
                previous_result = result
                results.append({
                    "task_id": sub_task.task_id,
                    "success": result.get("success", True),
                    "result": result
                })

            except Exception as e:
                logger.error(f"子任务执行失败: {sub_task.task_id} - {str(e)}")
                sub_task.status = TaskStatus.FAILED
                sub_task.result = {"success": False, "error": str(e)}
                results.append({
                    "task_id": sub_task.task_id,
                    "success": False,
                    "error": str(e)
                })

        # 更新计划状态
        all_completed = all(st.status == TaskStatus.COMPLETED for st in plan.sub_tasks)
        plan.status = TaskStatus.COMPLETED if all_completed else TaskStatus.FAILED
        plan.completed_at = datetime.now()

        return {"sub_task_results": results, "all_completed": all_completed}

    def _mock_execute(self, sub_task: SubTask) -> Dict[str, Any]:
        """
        模拟执行子任务（用于无工具管理器时的测试）

        Args:
            sub_task: 子任务对象

        Returns:
            模拟的执行结果
        """
        return {
            "success": True,
            "message": f"模拟执行完成: {sub_task.description}",
            "tool_used": sub_task.matched_tool.value if sub_task.matched_tool else "none",
            "data": None
        }

    def _integrate_results(self, plan: TaskPlan) -> str:
        """
        整合执行结果为自然语言回答

        Args:
            plan: 执行计划对象

        Returns:
            整合后的自然语言结果
        """
        if not plan.sub_tasks:
            return "任务未能成功解析，请重新描述您的需求。"

        result_parts = []
        result_parts.append(f"任务执行完成，共处理 {len(plan.sub_tasks)} 个步骤：\n")

        for sub_task in sorted(plan.sub_tasks, key=lambda x: x.order):
            status_text = "✓" if sub_task.status == TaskStatus.COMPLETED else "✗"
            result_parts.append(f"\n{status_text} 步骤{sub_task.order}: {sub_task.description}")

            if sub_task.result:
                if sub_task.result.get("success"):
                    if "data" in sub_task.result and sub_task.result["data"]:
                        result_parts.append(f"   结果: {sub_task.result['data']}")
                    elif "message" in sub_task.result:
                        result_parts.append(f"   {sub_task.result['message']}")
                else:
                    result_parts.append(f"   失败原因: {sub_task.result.get('error', '未知错误')}")

        final_result = "\n".join(result_parts)
        plan.final_result = final_result
        return final_result

    def get_plan_status(self) -> Optional[Dict[str, Any]]:
        """
        获取当前计划状态

        Returns:
            计划状态字典
        """
        if not self.current_plan:
            return None

        return {
            "plan_id": self.current_plan.plan_id,
            "status": self.current_plan.status.value,
            "sub_tasks": [
                {
                    "task_id": st.task_id,
                    "description": st.description,
                    "tool": st.matched_tool.value if st.matched_tool else None,
                    "status": st.status.value
                }
                for st in self.current_plan.sub_tasks
            ],
            "created_at": self.current_plan.created_at.isoformat(),
            "completed_at": self.current_plan.completed_at.isoformat() if self.current_plan.completed_at else None
        }

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取执行历史记录

        Args:
            limit: 返回的记录数量限制

        Returns:
            历史记录列表
        """
        history = []
        for plan in self.execution_history[-limit:]:
            history.append({
                "plan_id": plan.plan_id,
                "original_task": plan.original_task,
                "status": plan.status.value,
                "sub_task_count": len(plan.sub_tasks),
                "created_at": plan.created_at.isoformat(),
                "completed_at": plan.completed_at.isoformat() if plan.completed_at else None
            })
        return history


# 便捷函数：快速创建 Agent 实例
def create_agent(tool_manager=None) -> TaskAgent:
    """
    创建 TaskAgent 实例的便捷函数

    Args:
        tool_manager: 可选的工具管理器实例

    Returns:
        TaskAgent 实例
    """
    agent = TaskAgent(tool_manager)
    return agent


# 测试代码
if __name__ == "__main__":
    # 创建 Agent 实例
    agent = create_agent()

    # 测试任务解析
    test_tasks = [
        "提取test.pdf中的表格并计算均值",
        "将data.csv转换为Excel格式，然后画柱状图",
        "搜索机器学习相关文献",
        "添加明天上午10:00的会议日程",
        "运行代码 print('hello world')"
    ]

    print("=" * 60)
    print("任务解析测试")
    print("=" * 60)

    for task in test_tasks:
        print(f"\n原始任务: {task}")
        result = agent.process_task(task)
        if result["success"]:
            print(f"解析结果: {result['final_result']}")
        else:
            print(f"解析失败: {result['error']}")
        print("-" * 40)
