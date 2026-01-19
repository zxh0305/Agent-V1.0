#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型客户端模块
================

本模块实现与大模型API的交互，支持多种大模型提供商：
1. OpenAI (GPT系列)
2. 通义千问 (Qwen)
3. Ollama (本地模型)
4. 智谱AI (GLM系列)

主要功能：
- 意图识别：分析用户输入，识别用户意图
- 工具选择：根据用户意图推荐合适的工具
- 对话生成：支持多轮对话
- 文本摘要：生成文本摘要
- 翻译服务：使用大模型进行翻译

作者：开发团队
版本：3.0.0
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 导入配置管理器
try:
    from settings_manager import get_settings, LLMConfig
except ImportError:
    get_settings = None
    LLMConfig = None

logger = logging.getLogger(__name__)


# ============================================================
# 工具定义
# ============================================================

AVAILABLE_TOOLS = {
    "file_tool": {
        "name": "文件处理工具",
        "description": "用于处理PDF、Word等文件，支持文本提取、表格提取、格式转换等功能",
        "keywords": ["文件", "PDF", "Word", "提取", "转换", "文档"]
    },
    "data_tool": {
        "name": "数据分析工具",
        "description": "用于数据读取、统计分析和图表生成，支持CSV、Excel等格式",
        "keywords": ["数据", "分析", "统计", "图表", "计算", "均值", "柱状图", "折线图"]
    },
    "code_tool": {
        "name": "代码运行工具",
        "description": "在安全沙箱中运行Python代码",
        "keywords": ["代码", "运行", "执行", "Python", "程序"]
    },
    "paper_tool": {
        "name": "文献查询工具",
        "description": "搜索学术文献、论文，支持Semantic Scholar、arXiv等数据源",
        "keywords": ["文献", "论文", "学术", "搜索", "研究", "期刊"]
    },
    "schedule_tool": {
        "name": "日程管理工具",
        "description": "管理日程、提醒、待办事项",
        "keywords": ["日程", "提醒", "安排", "会议", "待办", "日历"]
    },
    "translate_tool": {
        "name": "翻译工具",
        "description": "支持多语言文本翻译，包括中英日韩等语言互译",
        "keywords": ["翻译", "英文", "中文", "日文", "语言"]
    },
    "summary_tool": {
        "name": "文本摘要工具",
        "description": "生成文本摘要、提取关键词、进行文本统计分析",
        "keywords": ["摘要", "总结", "概括", "关键词", "归纳"]
    },
    "qa_tool": {
        "name": "知识问答工具",
        "description": "回答编程、AI、数学等领域的知识问题",
        "keywords": ["问答", "什么是", "如何", "为什么", "解释"]
    }
}


# ============================================================
# 基础LLM客户端抽象类
# ============================================================

class BaseLLMClient(ABC):
    """
    大模型客户端基类

    定义了所有大模型客户端必须实现的接口
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化客户端

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.api_key = self.config.get("api_key", "")
        self.base_url = self.config.get("base_url", "")
        self.model = self.config.get("model", "")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: 其他参数

        Returns:
            响应结果字典
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查服务是否可用

        Returns:
            是否可用
        """
        pass

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        生成文本的便捷方法

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self.chat(messages, **kwargs)
        if result.get("success"):
            return result.get("content", "")
        return ""


# ============================================================
# OpenAI兼容客户端
# ============================================================

class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI兼容API客户端

    支持所有兼容OpenAI API格式的服务，包括：
    - OpenAI官方API
    - 通义千问
    - 智谱AI
    - 其他兼容服务
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化OpenAI兼容客户端"""
        super().__init__(config)
        self.timeout = self.config.get("timeout", 60)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            **kwargs: 其他参数（temperature, max_tokens等）

        Returns:
            响应结果
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "API密钥未配置",
                "content": ""
            }

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }

            url = f"{self.base_url.rstrip('/')}/chat/completions"

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "content": content,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", self.model)
                }
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "content": ""
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "content": ""
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "content": ""
            }
        except Exception as e:
            logger.error(f"LLM请求异常: {str(e)}")
            return {
                "success": False,
                "error": f"请求异常: {str(e)}",
                "content": ""
            }

    def is_available(self) -> bool:
        """检查服务是否可用"""
        if not self.api_key or not self.base_url:
            return False

        try:
            # 发送简单测试请求
            result = self.chat([{"role": "user", "content": "hi"}], max_tokens=5)
            return result.get("success", False)
        except Exception:
            return False


# ============================================================
# Ollama客户端
# ============================================================

class OllamaClient(BaseLLMClient):
    """
    Ollama本地模型客户端

    支持Ollama服务运行的本地大模型
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化Ollama客户端"""
        super().__init__(config)
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.timeout = self.config.get("timeout", 120)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求到Ollama

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            响应结果
        """
        try:
            url = f"{self.base_url.rstrip('/')}/api/chat"

            data = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature)
                }
            }

            response = requests.post(
                url,
                json=data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                return {
                    "success": True,
                    "content": content,
                    "model": result.get("model", self.model)
                }
            else:
                error_msg = f"Ollama请求失败: {response.status_code}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "content": ""
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "无法连接到Ollama服务，请确保Ollama正在运行",
                "content": ""
            }
        except Exception as e:
            logger.error(f"Ollama请求异常: {str(e)}")
            return {
                "success": False,
                "error": f"请求异常: {str(e)}",
                "content": ""
            }

    def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            url = f"{self.base_url.rstrip('/')}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            url = f"{self.base_url.rstrip('/')}/api/tags"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception:
            pass
        return []


# ============================================================
# LLM服务管理器
# ============================================================

class LLMService:
    """
    大模型服务管理器

    统一管理多个大模型提供商，提供高级功能：
    - 意图识别
    - 工具选择
    - 文本生成
    - 摘要生成
    - 翻译服务
    """

    def __init__(self):
        """初始化LLM服务"""
        self.client: Optional[BaseLLMClient] = None
        self.provider: str = ""
        self._initialized = False
        self._load_config()

    def _load_config(self) -> None:
        """加载配置并初始化客户端"""
        try:
            if get_settings:
                settings = get_settings()
                self.provider = settings.get("llm.provider", "qwen")
                provider_config = settings.get(f"llm.{self.provider}", {})

                if self.provider == "ollama":
                    self.client = OllamaClient(provider_config)
                else:
                    # OpenAI兼容的API（包括qwen, openai, zhipu）
                    self.client = OpenAICompatibleClient(provider_config)

                self._initialized = True
                logger.info(f"LLM服务初始化完成，使用提供商: {self.provider}")
            else:
                logger.warning("配置管理器不可用，LLM服务未初始化")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {str(e)}")

    def reload_config(self) -> None:
        """重新加载配置"""
        self._load_config()

    def is_available(self) -> bool:
        """检查LLM服务是否可用"""
        return self._initialized and self.client is not None and self.client.is_available()

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "initialized": self._initialized,
            "provider": self.provider,
            "available": self.is_available(),
            "model": self.client.model if self.client else ""
        }

    def recognize_intent(self, user_input: str, available_tools: List[str] = None) -> Dict[str, Any]:
        """
        识别用户意图并推荐工具

        使用大模型分析用户输入，识别用户意图，并推荐合适的工具

        Args:
            user_input: 用户输入的文本
            available_tools: 可用的工具列表

        Returns:
            意图识别结果，包含推荐的工具和置信度
        """
        if not self._initialized or not self.client:
            # 回退到关键词匹配
            return self._keyword_based_intent(user_input, available_tools)

        # 构建工具描述
        tools_desc = self._build_tools_description(available_tools)

        system_prompt = """你是一个智能任务分析助手。你的任务是分析用户的输入，识别用户的意图，并推荐最合适的工具。

可用的工具列表：
{tools}

请按照以下JSON格式返回分析结果（只返回JSON，不要有其他文字）：
{{
    "intent": "用户意图的简短描述",
    "recommended_tool": "推荐的工具名称（tool_name格式）",
    "confidence": 0.0-1.0之间的置信度,
    "reason": "推荐理由",
    "parameters": {{
        "提取的参数名": "参数值"
    }}
}}

如果用户的请求不需要任何工具（比如普通聊天），recommended_tool返回null。
""".format(tools=tools_desc)

        try:
            result = self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ], temperature=0.3, max_tokens=500)

            if result.get("success"):
                content = result.get("content", "")
                # 解析JSON响应
                try:
                    # 尝试提取JSON
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        intent_data = json.loads(json_str)
                        return {
                            "success": True,
                            "intent": intent_data.get("intent", ""),
                            "recommended_tool": intent_data.get("recommended_tool"),
                            "confidence": intent_data.get("confidence", 0.5),
                            "reason": intent_data.get("reason", ""),
                            "parameters": intent_data.get("parameters", {}),
                            "method": "llm"
                        }
                except json.JSONDecodeError:
                    logger.warning("无法解析LLM返回的JSON，回退到关键词匹配")

        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")

        # 回退到关键词匹配
        return self._keyword_based_intent(user_input, available_tools)

    def _build_tools_description(self, available_tools: List[str] = None) -> str:
        """构建工具描述文本"""
        tools = available_tools or list(AVAILABLE_TOOLS.keys())
        desc_parts = []
        for tool_name in tools:
            if tool_name in AVAILABLE_TOOLS:
                tool_info = AVAILABLE_TOOLS[tool_name]
                desc_parts.append(f"- {tool_name}: {tool_info['name']} - {tool_info['description']}")
        return "\n".join(desc_parts)

    def _keyword_based_intent(self, user_input: str, available_tools: List[str] = None) -> Dict[str, Any]:
        """
        基于关键词的意图识别（回退方案）

        Args:
            user_input: 用户输入
            available_tools: 可用工具列表

        Returns:
            意图识别结果
        """
        tools = available_tools or list(AVAILABLE_TOOLS.keys())
        user_input_lower = user_input.lower()

        best_match = None
        best_score = 0

        for tool_name in tools:
            if tool_name not in AVAILABLE_TOOLS:
                continue

            tool_info = AVAILABLE_TOOLS[tool_name]
            score = 0

            for keyword in tool_info.get("keywords", []):
                if keyword.lower() in user_input_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = tool_name

        if best_match and best_score > 0:
            confidence = min(0.9, 0.3 + best_score * 0.15)
            return {
                "success": True,
                "intent": f"使用{AVAILABLE_TOOLS[best_match]['name']}",
                "recommended_tool": best_match,
                "confidence": confidence,
                "reason": f"基于关键词匹配，匹配度: {best_score}",
                "parameters": {},
                "method": "keyword"
            }

        return {
            "success": True,
            "intent": "普通对话",
            "recommended_tool": None,
            "confidence": 0.5,
            "reason": "未匹配到明确的工具需求",
            "parameters": {},
            "method": "keyword"
        }

    def chat(self, user_input: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        与大模型对话

        Args:
            user_input: 用户输入
            context: 上下文消息列表

        Returns:
            对话结果
        """
        if not self._initialized or not self.client:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "content": ""
            }

        messages = context or []
        messages.append({"role": "user", "content": user_input})

        return self.client.chat(messages)

    def summarize(self, text: str, max_length: int = 200) -> Dict[str, Any]:
        """
        使用大模型生成文本摘要

        Args:
            text: 原始文本
            max_length: 摘要最大长度

        Returns:
            摘要结果
        """
        if not self._initialized or not self.client:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "summary": ""
            }

        system_prompt = f"""你是一个专业的文本摘要助手。请对用户提供的文本生成简洁的摘要。
要求：
1. 摘要长度不超过{max_length}字
2. 保留核心信息和关键观点
3. 使用简洁流畅的语言
4. 只返回摘要内容，不要有其他说明"""

        result = self.client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请总结以下文本：\n\n{text}"}
        ], temperature=0.5, max_tokens=max_length * 2)

        if result.get("success"):
            return {
                "success": True,
                "summary": result.get("content", ""),
                "original_length": len(text),
                "method": "llm"
            }

        return {
            "success": False,
            "error": result.get("error", "摘要生成失败"),
            "summary": ""
        }

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> Dict[str, Any]:
        """
        使用大模型进行翻译

        Args:
            text: 待翻译文本
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            翻译结果
        """
        if not self._initialized or not self.client:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "translation": ""
            }

        lang_names = {
            "zh": "中文",
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "fr": "法文",
            "de": "德文",
            "es": "西班牙文"
        }

        target_name = lang_names.get(target_lang, target_lang)
        source_name = lang_names.get(source_lang, "自动检测") if source_lang != "auto" else "自动检测"

        system_prompt = f"""你是一个专业的翻译助手。请将用户提供的文本翻译成{target_name}。
要求：
1. 翻译要准确、流畅、自然
2. 保持原文的语气和风格
3. 只返回翻译结果，不要有其他说明或解释"""

        result = self.client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ], temperature=0.3)

        if result.get("success"):
            return {
                "success": True,
                "translation": result.get("content", ""),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "method": "llm"
            }

        return {
            "success": False,
            "error": result.get("error", "翻译失败"),
            "translation": ""
        }

    def answer_question(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        使用大模型回答问题

        Args:
            question: 用户问题
            context: 可选的上下文信息

        Returns:
            回答结果
        """
        if not self._initialized or not self.client:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "answer": ""
            }

        system_prompt = """你是一个知识渊博的助手，擅长回答编程、人工智能、数学、科学等领域的问题。
请根据用户的问题提供准确、详细、有帮助的回答。
如果问题超出你的知识范围，请诚实地说明。"""

        user_content = question
        if context:
            user_content = f"参考信息：{context}\n\n问题：{question}"

        result = self.client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

        if result.get("success"):
            return {
                "success": True,
                "answer": result.get("content", ""),
                "method": "llm"
            }

        return {
            "success": False,
            "error": result.get("error", "回答生成失败"),
            "answer": ""
        }


# ============================================================
# 全局LLM服务实例
# ============================================================

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取全局LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reload_llm_service() -> None:
    """重新加载LLM服务配置"""
    global _llm_service
    if _llm_service:
        _llm_service.reload_config()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("LLM客户端测试")
    print("=" * 50)

    # 获取LLM服务
    llm = get_llm_service()

    # 检查状态
    status = llm.get_status()
    print(f"\n服务状态: {status}")

    # 测试意图识别
    test_inputs = [
        "帮我翻译一下这段话",
        "提取这个PDF文件中的表格",
        "什么是机器学习",
        "帮我画一个柱状图",
        "明天上午10点有个会议，帮我添加到日程"
    ]

    print("\n" + "-" * 40)
    print("意图识别测试：")
    for input_text in test_inputs:
        result = llm.recognize_intent(input_text)
        print(f"\n输入: {input_text}")
        print(f"  意图: {result.get('intent')}")
        print(f"  推荐工具: {result.get('recommended_tool')}")
        print(f"  置信度: {result.get('confidence')}")
        print(f"  方法: {result.get('method')}")
