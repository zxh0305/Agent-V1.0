#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理器模块
==============

本模块实现配置文件的加载、验证、保存和热更新功能

作者：学生开发团队
版本：3.0.0
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = "settings.yaml"


@dataclass
class LLMConfig:
    """大模型配置"""
    provider: str = "qwen"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class TranslateConfig:
    """翻译服务配置"""
    provider: str = "baidu"
    app_id: str = ""
    secret_key: str = ""
    api_url: str = ""


@dataclass
class PaperSearchConfig:
    """文献搜索配置"""
    provider: str = "semantic_scholar"
    api_url: str = ""
    api_key: str = ""


class SettingsManager:
    """
    配置管理器

    负责加载、验证、保存和管理所有配置项

    Attributes:
        config_path: 配置文件路径
        settings: 配置字典
    """

    _instance = None

    def __new__(cls, config_path: str = DEFAULT_CONFIG_PATH):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        if self._initialized:
            return

        self.config_path = Path(config_path)
        self.settings: Dict[str, Any] = {}
        self._load_config()
        self._initialized = True
        logger.info(f"配置管理器初始化完成，配置文件: {self.config_path}")

    def _load_config(self) -> None:
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.settings = yaml.safe_load(f) or {}
                logger.info("配置文件加载成功")
            except Exception as e:
                logger.error(f"配置文件加载失败: {e}")
                self.settings = self._get_default_settings()
        else:
            logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            self.settings = self._get_default_settings()
            self._save_config()

    def _get_default_settings(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm": {
                "provider": "qwen",
                "openai": {
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "qwen": {
                    "api_key": "",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-turbo",
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "qwen2:7b",
                    "temperature": 0.7
                }
            },
            "translate": {
                "provider": "baidu",
                "baidu": {
                    "app_id": "",
                    "secret_key": "",
                    "api_url": "https://fanyi-api.baidu.com/api/trans/vip/translate"
                }
            },
            "paper_search": {
                "provider": "semantic_scholar",
                "semantic_scholar": {
                    "api_url": "https://api.semanticscholar.org/graph/v1",
                    "api_key": ""
                }
            },
            "ui": {
                "page_title": "智能Agent平台 V3.0",
                "page_icon": "🤖",
                "theme": "light",
                "show_debug_info": False
            },
            "system": {
                "log_level": "INFO",
                "log_dir": "logs"
            }
        }

    def _save_config(self) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.settings, f, default_flow_style=False, allow_unicode=True)
            logger.info("配置文件保存成功")
            return True
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False

    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()
        logger.info("配置已重新加载")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        支持点号分隔的嵌套键，如 "llm.openai.api_key"

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.settings

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        设置配置项

        Args:
            key: 配置键（支持点号分隔）
            value: 配置值
            save: 是否立即保存到文件

        Returns:
            是否设置成功
        """
        keys = key.split('.')
        target = self.settings

        try:
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

            if save:
                return self._save_config()
            return True
        except Exception as e:
            logger.error(f"设置配置项失败: {e}")
            return False

    def get_llm_config(self) -> LLMConfig:
        """获取大模型配置"""
        provider = self.get("llm.provider", "qwen")
        provider_config = self.get(f"llm.{provider}", {})

        return LLMConfig(
            provider=provider,
            api_key=provider_config.get("api_key", ""),
            base_url=provider_config.get("base_url", ""),
            model=provider_config.get("model", ""),
            temperature=provider_config.get("temperature", 0.7),
            max_tokens=provider_config.get("max_tokens", 2000)
        )

    def get_translate_config(self) -> TranslateConfig:
        """获取翻译配置"""
        provider = self.get("translate.provider", "baidu")
        provider_config = self.get(f"translate.{provider}", {})

        return TranslateConfig(
            provider=provider,
            app_id=provider_config.get("app_id", ""),
            secret_key=provider_config.get("secret_key", provider_config.get("secret_key", "")),
            api_url=provider_config.get("api_url", "")
        )

    def get_paper_search_config(self) -> PaperSearchConfig:
        """获取文献搜索配置"""
        provider = self.get("paper_search.provider", "semantic_scholar")
        provider_config = self.get(f"paper_search.{provider}", {})

        return PaperSearchConfig(
            provider=provider,
            api_url=provider_config.get("api_url", ""),
            api_key=provider_config.get("api_key", "")
        )

    def validate_llm_config(self) -> Dict[str, Any]:
        """验证大模型配置"""
        config = self.get_llm_config()
        issues = []

        if config.provider in ["openai", "qwen", "zhipu"]:
            if not config.api_key:
                issues.append(f"{config.provider} API密钥未配置")
        elif config.provider == "ollama":
            if not config.base_url:
                issues.append("Ollama 服务地址未配置")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "provider": config.provider
        }

    def validate_translate_config(self) -> Dict[str, Any]:
        """验证翻译配置"""
        config = self.get_translate_config()
        issues = []

        if config.provider == "baidu":
            if not config.app_id:
                issues.append("百度翻译 APP ID 未配置")
            if not config.secret_key:
                issues.append("百度翻译密钥未配置")
        elif config.provider == "youdao":
            if not config.app_id:
                issues.append("有道翻译 APP Key 未配置")
            if not config.secret_key:
                issues.append("有道翻译密钥未配置")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "provider": config.provider
        }

    def to_dict(self) -> Dict[str, Any]:
        """返回所有配置"""
        return self.settings.copy()

    def update_from_dict(self, config_dict: Dict[str, Any], save: bool = True) -> bool:
        """
        从字典更新配置

        Args:
            config_dict: 配置字典
            save: 是否保存

        Returns:
            是否更新成功
        """
        def deep_update(base: dict, update: dict):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value

        try:
            deep_update(self.settings, config_dict)
            if save:
                return self._save_config()
            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False


# 全局配置管理器实例
_settings_manager: Optional[SettingsManager] = None


def get_settings() -> SettingsManager:
    """获取全局配置管理器实例"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def reload_settings() -> None:
    """重新加载配置"""
    global _settings_manager
    if _settings_manager:
        _settings_manager.reload()


# 测试代码
if __name__ == "__main__":
    settings = get_settings()

    print("=" * 50)
    print("配置管理器测试")
    print("=" * 50)

    # 测试获取配置
    print(f"\n大模型提供商: {settings.get('llm.provider')}")
    print(f"翻译提供商: {settings.get('translate.provider')}")

    # 测试配置验证
    llm_valid = settings.validate_llm_config()
    print(f"\n大模型配置验证: {llm_valid}")

    translate_valid = settings.validate_translate_config()
    print(f"翻译配置验证: {translate_valid}")
