#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译服务模块
============

本模块实现真实的翻译API接入，支持多种翻译服务提供商：
1. 百度翻译API
2. 有道翻译API
3. 大模型翻译（调用LLM）

作者：开发团队
版本：3.0.0
"""

import hashlib
import random
import time
import logging
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# 导入配置管理器
try:
    from settings_manager import get_settings
except ImportError:
    get_settings = None

# 导入LLM服务
try:
    from llm_client import get_llm_service
except ImportError:
    get_llm_service = None

logger = logging.getLogger(__name__)


# ============================================================
# 语言代码映射
# ============================================================

# 统一语言代码到各平台的映射
LANG_CODE_MAP = {
    "baidu": {
        "zh": "zh",
        "en": "en",
        "ja": "jp",
        "ko": "kor",
        "fr": "fra",
        "de": "de",
        "es": "spa",
        "ru": "ru",
        "pt": "pt",
        "it": "it",
        "auto": "auto"
    },
    "youdao": {
        "zh": "zh-CHS",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "ru": "ru",
        "pt": "pt",
        "it": "it",
        "auto": "auto"
    }
}

# 语言名称
LANG_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "fr": "法文",
    "de": "德文",
    "es": "西班牙文",
    "ru": "俄文",
    "pt": "葡萄牙文",
    "it": "意大利文",
    "auto": "自动检测"
}


# ============================================================
# 翻译服务基类
# ============================================================

class BaseTranslator(ABC):
    """翻译服务基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化翻译器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    @abstractmethod
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> Dict[str, Any]:
        """
        翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言代码
            target_lang: 目标语言代码

        Returns:
            翻译结果字典
        """
        pass

    @abstractmethod
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        检测文本语言

        Args:
            text: 待检测文本

        Returns:
            检测结果字典
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


# ============================================================
# 百度翻译服务
# ============================================================

class BaiduTranslator(BaseTranslator):
    """
    百度翻译API客户端

    使用百度翻译开放平台API进行翻译
    API文档: https://fanyi-api.baidu.com/doc/21
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化百度翻译器"""
        super().__init__(config)
        self.app_id = self.config.get("app_id", "")
        self.secret_key = self.config.get("secret_key", "")
        self.api_url = self.config.get("api_url", "https://fanyi-api.baidu.com/api/trans/vip/translate")
        self.timeout = 10

    def _make_sign(self, query: str, salt: str) -> str:
        """
        生成签名

        Args:
            query: 查询文本
            salt: 随机数

        Returns:
            MD5签名
        """
        sign_str = f"{self.app_id}{query}{salt}{self.secret_key}"
        return hashlib.md5(sign_str.encode('utf-8')).hexdigest()

    def _get_lang_code(self, lang: str) -> str:
        """获取百度翻译语言代码"""
        return LANG_CODE_MAP["baidu"].get(lang, lang)

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> Dict[str, Any]:
        """
        使用百度翻译API翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            翻译结果
        """
        if not text.strip():
            return {
                "success": False,
                "error": "翻译文本不能为空",
                "translation": ""
            }

        if not self.app_id or not self.secret_key:
            return {
                "success": False,
                "error": "百度翻译API未配置，请在settings.yaml中配置app_id和secret_key",
                "translation": ""
            }

        try:
            # 生成签名
            salt = str(random.randint(32768, 65536))
            sign = self._make_sign(text, salt)

            # 构建请求参数
            params = {
                "q": text,
                "from": self._get_lang_code(source_lang),
                "to": self._get_lang_code(target_lang),
                "appid": self.app_id,
                "salt": salt,
                "sign": sign
            }

            # 发送请求
            response = requests.get(self.api_url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()

                if "trans_result" in result:
                    # 成功
                    translations = result["trans_result"]
                    translated_text = "\n".join([item["dst"] for item in translations])

                    return {
                        "success": True,
                        "translation": translated_text,
                        "source_lang": result.get("from", source_lang),
                        "target_lang": result.get("to", target_lang),
                        "provider": "baidu"
                    }
                else:
                    # API返回错误
                    error_code = result.get("error_code", "unknown")
                    error_msg = self._get_error_message(error_code)
                    return {
                        "success": False,
                        "error": f"百度翻译API错误: {error_msg} (错误码: {error_code})",
                        "translation": ""
                    }
            else:
                return {
                    "success": False,
                    "error": f"请求失败: HTTP {response.status_code}",
                    "translation": ""
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "translation": ""
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "translation": ""
            }
        except Exception as e:
            logger.error(f"百度翻译异常: {str(e)}")
            return {
                "success": False,
                "error": f"翻译异常: {str(e)}",
                "translation": ""
            }

    def _get_error_message(self, error_code: str) -> str:
        """获取错误信息"""
        error_messages = {
            "52001": "请求超时",
            "52002": "系统错误",
            "52003": "未授权用户",
            "54000": "必填参数为空",
            "54001": "签名错误",
            "54003": "访问频率受限",
            "54004": "账户余额不足",
            "54005": "长query请求频繁",
            "58000": "客户端IP非法",
            "58001": "译文语言方向不支持",
            "58002": "服务当前已关闭",
            "90107": "认证未通过或未生效"
        }
        return error_messages.get(str(error_code), "未知错误")

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        检测文本语言（使用百度翻译的自动检测功能）

        Args:
            text: 待检测文本

        Returns:
            检测结果
        """
        # 百度翻译会在translate响应中返回检测到的语言
        result = self.translate(text, source_lang="auto", target_lang="en")
        if result.get("success"):
            detected = result.get("source_lang", "unknown")
            return {
                "success": True,
                "language": detected,
                "language_name": LANG_NAMES.get(detected, detected),
                "confidence": 0.9,
                "provider": "baidu"
            }
        return {
            "success": False,
            "error": result.get("error", "检测失败"),
            "language": ""
        }

    def is_available(self) -> bool:
        """检查百度翻译服务是否可用"""
        return bool(self.app_id and self.secret_key)


# ============================================================
# 有道翻译服务
# ============================================================

class YoudaoTranslator(BaseTranslator):
    """
    有道翻译API客户端

    使用有道智云翻译API进行翻译
    API文档: https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化有道翻译器"""
        super().__init__(config)
        self.app_key = self.config.get("app_key", "")
        self.app_secret = self.config.get("app_secret", "")
        self.api_url = self.config.get("api_url", "https://openapi.youdao.com/api")
        self.timeout = 10

    def _make_sign(self, query: str, salt: str, curtime: str) -> str:
        """
        生成签名

        Args:
            query: 查询文本
            salt: 随机数
            curtime: 当前时间戳

        Returns:
            SHA256签名
        """
        # 处理input：如果长度>20，取前10+长度+后10
        if len(query) > 20:
            input_str = query[:10] + str(len(query)) + query[-10:]
        else:
            input_str = query

        sign_str = f"{self.app_key}{input_str}{salt}{curtime}{self.app_secret}"
        return hashlib.sha256(sign_str.encode('utf-8')).hexdigest()

    def _get_lang_code(self, lang: str) -> str:
        """获取有道翻译语言代码"""
        return LANG_CODE_MAP["youdao"].get(lang, lang)

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> Dict[str, Any]:
        """
        使用有道翻译API翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            翻译结果
        """
        if not text.strip():
            return {
                "success": False,
                "error": "翻译文本不能为空",
                "translation": ""
            }

        if not self.app_key or not self.app_secret:
            return {
                "success": False,
                "error": "有道翻译API未配置，请在settings.yaml中配置app_key和app_secret",
                "translation": ""
            }

        try:
            # 生成签名
            salt = str(random.randint(1, 65536))
            curtime = str(int(time.time()))
            sign = self._make_sign(text, salt, curtime)

            # 构建请求参数
            params = {
                "q": text,
                "from": self._get_lang_code(source_lang),
                "to": self._get_lang_code(target_lang),
                "appKey": self.app_key,
                "salt": salt,
                "sign": sign,
                "signType": "v3",
                "curtime": curtime
            }

            # 发送请求
            response = requests.post(self.api_url, data=params, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()

                if result.get("errorCode") == "0":
                    # 成功
                    translations = result.get("translation", [])
                    translated_text = "\n".join(translations)

                    return {
                        "success": True,
                        "translation": translated_text,
                        "source_lang": result.get("l", "").split("2")[0] if result.get("l") else source_lang,
                        "target_lang": target_lang,
                        "provider": "youdao"
                    }
                else:
                    # API返回错误
                    error_code = result.get("errorCode", "unknown")
                    error_msg = self._get_error_message(error_code)
                    return {
                        "success": False,
                        "error": f"有道翻译API错误: {error_msg} (错误码: {error_code})",
                        "translation": ""
                    }
            else:
                return {
                    "success": False,
                    "error": f"请求失败: HTTP {response.status_code}",
                    "translation": ""
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "translation": ""
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "translation": ""
            }
        except Exception as e:
            logger.error(f"有道翻译异常: {str(e)}")
            return {
                "success": False,
                "error": f"翻译异常: {str(e)}",
                "translation": ""
            }

    def _get_error_message(self, error_code: str) -> str:
        """获取错误信息"""
        error_messages = {
            "101": "缺少必填参数",
            "102": "不支持的语言类型",
            "103": "翻译文本过长",
            "108": "应用ID无效",
            "111": "访问频率受限",
            "112": "请求服务无效",
            "113": "查询为空",
            "202": "签名检验失败",
            "203": "访问IP地址不在可访问IP列表",
            "301": "辞典查询失败",
            "302": "翻译查询失败",
            "303": "服务端的其它异常",
            "401": "账户已欠费",
            "411": "访问频率受限",
            "412": "长请求过于频繁"
        }
        return error_messages.get(str(error_code), "未知错误")

    def detect_language(self, text: str) -> Dict[str, Any]:
        """检测文本语言"""
        result = self.translate(text, source_lang="auto", target_lang="en")
        if result.get("success"):
            detected = result.get("source_lang", "unknown")
            return {
                "success": True,
                "language": detected,
                "language_name": LANG_NAMES.get(detected, detected),
                "confidence": 0.9,
                "provider": "youdao"
            }
        return {
            "success": False,
            "error": result.get("error", "检测失败"),
            "language": ""
        }

    def is_available(self) -> bool:
        """检查有道翻译服务是否可用"""
        return bool(self.app_key and self.app_secret)


# ============================================================
# LLM翻译服务
# ============================================================

class LLMTranslator(BaseTranslator):
    """
    使用大模型进行翻译

    当其他翻译服务不可用时，可以使用LLM进行翻译
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化LLM翻译器"""
        super().__init__(config)
        self.llm_service = None
        if get_llm_service:
            self.llm_service = get_llm_service()

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> Dict[str, Any]:
        """
        使用LLM翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            翻译结果
        """
        if not self.llm_service:
            return {
                "success": False,
                "error": "LLM服务不可用",
                "translation": ""
            }

        result = self.llm_service.translate(text, source_lang, target_lang)
        if result.get("success"):
            return {
                "success": True,
                "translation": result.get("translation", ""),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "provider": "llm"
            }

        return {
            "success": False,
            "error": result.get("error", "LLM翻译失败"),
            "translation": ""
        }

    def detect_language(self, text: str) -> Dict[str, Any]:
        """使用简单规则检测语言"""
        import re

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))

        total = chinese_chars + english_chars + japanese_chars + 1

        if chinese_chars / total > 0.3:
            detected = "zh"
        elif japanese_chars / total > 0.1:
            detected = "ja"
        else:
            detected = "en"

        return {
            "success": True,
            "language": detected,
            "language_name": LANG_NAMES.get(detected, detected),
            "confidence": 0.7,
            "provider": "rule"
        }

    def is_available(self) -> bool:
        """检查LLM翻译服务是否可用"""
        return self.llm_service is not None and self.llm_service.is_available()


# ============================================================
# 翻译服务管理器
# ============================================================

class TranslateService:
    """
    翻译服务管理器

    统一管理多个翻译服务提供商，自动选择可用的服务
    """

    def __init__(self):
        """初始化翻译服务"""
        self.translators: Dict[str, BaseTranslator] = {}
        self.primary_provider: str = ""
        self._load_config()

    def _load_config(self) -> None:
        """加载配置并初始化翻译器"""
        try:
            if get_settings:
                settings = get_settings()
                self.primary_provider = settings.get("translate.provider", "baidu")

                # 初始化百度翻译
                baidu_config = settings.get("translate.baidu", {})
                if baidu_config:
                    self.translators["baidu"] = BaiduTranslator(baidu_config)

                # 初始化有道翻译
                youdao_config = settings.get("translate.youdao", {})
                if youdao_config:
                    self.translators["youdao"] = YoudaoTranslator(youdao_config)

                # 初始化LLM翻译
                llm_config = settings.get("translate.llm", {})
                if llm_config.get("enabled", True):
                    self.translators["llm"] = LLMTranslator(llm_config)

                logger.info(f"翻译服务初始化完成，主要提供商: {self.primary_provider}")
            else:
                logger.warning("配置管理器不可用，翻译服务未完全初始化")
                # 至少初始化LLM翻译作为后备
                self.translators["llm"] = LLMTranslator({})
                self.primary_provider = "llm"

        except Exception as e:
            logger.error(f"翻译服务初始化失败: {str(e)}")

    def reload_config(self) -> None:
        """重新加载配置"""
        self.translators.clear()
        self._load_config()

    def get_available_providers(self) -> List[str]:
        """获取可用的翻译服务提供商列表"""
        available = []
        for name, translator in self.translators.items():
            if translator.is_available():
                available.append(name)
        return available

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh",
                  provider: str = None) -> Dict[str, Any]:
        """
        翻译文本

        Args:
            text: 待翻译文本
            source_lang: 源语言
            target_lang: 目标语言
            provider: 指定的翻译服务提供商

        Returns:
            翻译结果
        """
        if not text.strip():
            return {
                "success": False,
                "error": "翻译文本不能为空",
                "data": None
            }

        # 确定使用的翻译器
        translator = None
        provider_name = provider or self.primary_provider

        if provider_name in self.translators:
            translator = self.translators[provider_name]
            if not translator.is_available():
                logger.warning(f"指定的翻译服务 {provider_name} 不可用，尝试其他服务")
                translator = None

        # 如果指定的不可用，尝试其他可用的
        if translator is None:
            for name, trans in self.translators.items():
                if trans.is_available():
                    translator = trans
                    provider_name = name
                    break

        if translator is None:
            return {
                "success": False,
                "error": "没有可用的翻译服务，请检查配置",
                "data": None
            }

        # 执行翻译
        result = translator.translate(text, source_lang, target_lang)

        if result.get("success"):
            return {
                "success": True,
                "message": f"翻译完成 ({source_lang} -> {target_lang})",
                "data": {
                    "original_text": text,
                    "translated_text": result.get("translation", ""),
                    "source_language": result.get("source_lang", source_lang),
                    "target_language": target_lang,
                    "provider": result.get("provider", provider_name)
                }
            }

        return {
            "success": False,
            "error": result.get("error", "翻译失败"),
            "data": None
        }

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        检测文本语言

        Args:
            text: 待检测文本

        Returns:
            检测结果
        """
        if not text.strip():
            return {
                "success": False,
                "error": "文本不能为空",
                "data": None
            }

        # 使用主要翻译器进行检测
        translator = self.translators.get(self.primary_provider)
        if translator and translator.is_available():
            result = translator.detect_language(text)
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"检测到文本语言为: {result.get('language_name', '')}",
                    "data": result
                }

        # 回退到简单规则检测
        llm_translator = self.translators.get("llm")
        if llm_translator:
            result = llm_translator.detect_language(text)
            return {
                "success": True,
                "message": f"检测到文本语言为: {result.get('language_name', '')}",
                "data": result
            }

        return {
            "success": False,
            "error": "语言检测失败",
            "data": None
        }

    def list_languages(self) -> Dict[str, Any]:
        """列出支持的语言"""
        return {
            "success": True,
            "message": f"支持 {len(LANG_NAMES)} 种语言",
            "data": {
                "languages": LANG_NAMES,
                "total": len(LANG_NAMES)
            }
        }


# ============================================================
# 全局翻译服务实例
# ============================================================

_translate_service: Optional[TranslateService] = None


def get_translate_service() -> TranslateService:
    """获取全局翻译服务实例"""
    global _translate_service
    if _translate_service is None:
        _translate_service = TranslateService()
    return _translate_service


def reload_translate_service() -> None:
    """重新加载翻译服务配置"""
    global _translate_service
    if _translate_service:
        _translate_service.reload_config()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("翻译服务测试")
    print("=" * 50)

    # 获取翻译服务
    service = get_translate_service()

    # 显示可用的提供商
    providers = service.get_available_providers()
    print(f"\n可用的翻译服务: {providers}")

    # 测试翻译
    test_texts = [
        ("Hello, World!", "auto", "zh"),
        ("机器学习是人工智能的一个分支", "zh", "en"),
        ("深度学习", "auto", "ja"),
    ]

    print("\n" + "-" * 40)
    print("翻译测试：")
    for text, src, tgt in test_texts:
        result = service.translate(text, src, tgt)
        print(f"\n原文: {text}")
        print(f"目标语言: {tgt}")
        if result.get("success"):
            print(f"译文: {result['data']['translated_text']}")
            print(f"提供商: {result['data']['provider']}")
        else:
            print(f"错误: {result.get('error')}")

    # 测试语言检测
    print("\n" + "-" * 40)
    print("语言检测测试：")
    detect_texts = ["Hello world", "你好世界", "こんにちは"]
    for text in detect_texts:
        result = service.detect_language(text)
        print(f"\n文本: {text}")
        if result.get("success"):
            print(f"检测结果: {result['data']['language_name']}")
        else:
            print(f"错误: {result.get('error')}")
