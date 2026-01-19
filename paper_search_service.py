#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文献搜索服务模块
================

本模块实现真实的学术文献搜索API接入，支持：
1. Semantic Scholar API
2. arXiv API
3. CrossRef API

作者：开发团队
版本：3.0.0
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from urllib.parse import quote_plus
import time

# 导入配置管理器
try:
    from settings_manager import get_settings
except ImportError:
    get_settings = None

logger = logging.getLogger(__name__)


# ============================================================
# 文献搜索基类
# ============================================================

class BasePaperSearcher(ABC):
    """文献搜索服务基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化搜索器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.timeout = 30

    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        搜索文献

        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
            **kwargs: 其他参数

        Returns:
            搜索结果字典
        """
        pass

    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文详情

        Args:
            paper_id: 论文ID

        Returns:
            论文详情字典
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


# ============================================================
# Semantic Scholar API
# ============================================================

class SemanticScholarSearcher(BasePaperSearcher):
    """
    Semantic Scholar API客户端

    Semantic Scholar是一个免费的学术搜索引擎
    API文档: https://api.semanticscholar.org/api-docs/
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化Semantic Scholar搜索器"""
        super().__init__(config)
        self.api_url = self.config.get("api_url", "https://api.semanticscholar.org/graph/v1")
        self.api_key = self.config.get("api_key", "")
        self.rate_limit_delay = 1  # API有频率限制

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        搜索Semantic Scholar文献

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            **kwargs: 其他参数（year, fields_of_study等）

        Returns:
            搜索结果
        """
        if not query.strip():
            return {
                "success": False,
                "error": "搜索关键词不能为空",
                "papers": []
            }

        try:
            # 构建请求URL
            url = f"{self.api_url}/paper/search"
            params = {
                "query": query,
                "limit": min(limit, 100),  # API最大限制100
                "fields": "paperId,title,abstract,year,authors,citationCount,url,venue,publicationDate"
            }

            # 添加年份过滤
            year = kwargs.get("year")
            if year:
                params["year"] = year

            # 发送请求
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                papers = []

                for item in result.get("data", []):
                    paper = {
                        "id": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", "")[:500] if item.get("abstract") else "",
                        "year": item.get("year"),
                        "authors": [a.get("name", "") for a in item.get("authors", [])[:5]],
                        "citations": item.get("citationCount", 0),
                        "url": item.get("url", ""),
                        "venue": item.get("venue", ""),
                        "source": "semantic_scholar"
                    }
                    papers.append(paper)

                return {
                    "success": True,
                    "total": result.get("total", len(papers)),
                    "papers": papers,
                    "provider": "semantic_scholar"
                }

            elif response.status_code == 429:
                return {
                    "success": False,
                    "error": "API请求频率超限，请稍后重试",
                    "papers": []
                }
            else:
                return {
                    "success": False,
                    "error": f"API请求失败: HTTP {response.status_code}",
                    "papers": []
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "papers": []
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "papers": []
            }
        except Exception as e:
            logger.error(f"Semantic Scholar搜索异常: {str(e)}")
            return {
                "success": False,
                "error": f"搜索异常: {str(e)}",
                "papers": []
            }

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文详情

        Args:
            paper_id: Semantic Scholar论文ID

        Returns:
            论文详情
        """
        try:
            url = f"{self.api_url}/paper/{paper_id}"
            params = {
                "fields": "paperId,title,abstract,year,authors,citationCount,referenceCount,url,venue,publicationDate,fieldsOfStudy,citations,references"
            }

            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                item = response.json()
                return {
                    "success": True,
                    "paper": {
                        "id": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "year": item.get("year"),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "citations": item.get("citationCount", 0),
                        "references": item.get("referenceCount", 0),
                        "url": item.get("url", ""),
                        "venue": item.get("venue", ""),
                        "fields": item.get("fieldsOfStudy", []),
                        "source": "semantic_scholar"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"获取论文详情失败: HTTP {response.status_code}",
                    "paper": None
                }

        except Exception as e:
            logger.error(f"获取论文详情异常: {str(e)}")
            return {
                "success": False,
                "error": f"获取详情异常: {str(e)}",
                "paper": None
            }

    def is_available(self) -> bool:
        """检查Semantic Scholar服务是否可用"""
        try:
            response = requests.get(
                f"{self.api_url}/paper/search",
                params={"query": "test", "limit": 1},
                headers=self._get_headers(),
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================
# arXiv API
# ============================================================

class ArxivSearcher(BasePaperSearcher):
    """
    arXiv API客户端

    arXiv是一个免费的预印本数据库
    API文档: https://arxiv.org/help/api/
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化arXiv搜索器"""
        super().__init__(config)
        self.api_url = self.config.get("api_url", "http://export.arxiv.org/api/query")

    def search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        搜索arXiv文献

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            **kwargs: 其他参数

        Returns:
            搜索结果
        """
        if not query.strip():
            return {
                "success": False,
                "error": "搜索关键词不能为空",
                "papers": []
            }

        try:
            # 构建搜索查询
            search_query = f"all:{quote_plus(query)}"

            # 构建请求参数
            params = {
                "search_query": search_query,
                "start": kwargs.get("start", 0),
                "max_results": min(limit, 100),
                "sortBy": kwargs.get("sort_by", "relevance"),
                "sortOrder": kwargs.get("sort_order", "descending")
            }

            # 发送请求
            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                # 解析XML响应
                papers = self._parse_arxiv_response(response.text)
                return {
                    "success": True,
                    "total": len(papers),
                    "papers": papers,
                    "provider": "arxiv"
                }
            else:
                return {
                    "success": False,
                    "error": f"API请求失败: HTTP {response.status_code}",
                    "papers": []
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "papers": []
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "papers": []
            }
        except Exception as e:
            logger.error(f"arXiv搜索异常: {str(e)}")
            return {
                "success": False,
                "error": f"搜索异常: {str(e)}",
                "papers": []
            }

    def _parse_arxiv_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """解析arXiv API的XML响应"""
        papers = []

        try:
            # 定义命名空间
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            root = ET.fromstring(xml_text)

            for entry in root.findall('atom:entry', namespaces):
                # 提取ID
                id_elem = entry.find('atom:id', namespaces)
                arxiv_id = ""
                url = ""
                if id_elem is not None:
                    url = id_elem.text or ""
                    arxiv_id = url.split('/abs/')[-1] if '/abs/' in url else url

                # 提取标题
                title_elem = entry.find('atom:title', namespaces)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

                # 提取摘要
                summary_elem = entry.find('atom:summary', namespaces)
                abstract = summary_elem.text.strip()[:500] if summary_elem is not None else ""

                # 提取作者
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                # 提取发布日期
                published_elem = entry.find('atom:published', namespaces)
                year = None
                if published_elem is not None:
                    year = int(published_elem.text[:4])

                # 提取分类
                categories = []
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term', '')
                    if term:
                        categories.append(term)

                paper = {
                    "id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors[:5],
                    "citations": 0,  # arXiv不提供引用数
                    "url": url,
                    "venue": "arXiv",
                    "categories": categories,
                    "source": "arxiv"
                }
                papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"XML解析错误: {str(e)}")

        return papers

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文详情

        Args:
            paper_id: arXiv论文ID

        Returns:
            论文详情
        """
        try:
            params = {
                "id_list": paper_id,
                "max_results": 1
            }

            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                papers = self._parse_arxiv_response(response.text)
                if papers:
                    return {
                        "success": True,
                        "paper": papers[0]
                    }
                else:
                    return {
                        "success": False,
                        "error": "未找到论文",
                        "paper": None
                    }
            else:
                return {
                    "success": False,
                    "error": f"获取论文详情失败: HTTP {response.status_code}",
                    "paper": None
                }

        except Exception as e:
            logger.error(f"获取arXiv论文详情异常: {str(e)}")
            return {
                "success": False,
                "error": f"获取详情异常: {str(e)}",
                "paper": None
            }

    def is_available(self) -> bool:
        """检查arXiv服务是否可用"""
        try:
            response = requests.get(
                self.api_url,
                params={"search_query": "all:test", "max_results": 1},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================
# CrossRef API
# ============================================================

class CrossRefSearcher(BasePaperSearcher):
    """
    CrossRef API客户端

    CrossRef是一个DOI注册机构，提供学术元数据
    API文档: https://api.crossref.org/
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化CrossRef搜索器"""
        super().__init__(config)
        self.api_url = self.config.get("api_url", "https://api.crossref.org/works")
        self.email = self.config.get("email", "")

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        # CrossRef建议在请求中包含邮箱以获得更好的服务
        if self.email:
            headers["User-Agent"] = f"AgentPlatform/3.0 (mailto:{self.email})"
        return headers

    def search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        搜索CrossRef文献

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            **kwargs: 其他参数

        Returns:
            搜索结果
        """
        if not query.strip():
            return {
                "success": False,
                "error": "搜索关键词不能为空",
                "papers": []
            }

        try:
            params = {
                "query": query,
                "rows": min(limit, 100),
                "select": "DOI,title,abstract,author,published-print,container-title,is-referenced-by-count,URL"
            }

            response = requests.get(
                self.api_url,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                papers = []

                for item in result.get("message", {}).get("items", []):
                    # 提取年份
                    year = None
                    published = item.get("published-print", {}) or item.get("published-online", {})
                    if published and "date-parts" in published:
                        date_parts = published["date-parts"]
                        if date_parts and date_parts[0]:
                            year = date_parts[0][0]

                    # 提取作者
                    authors = []
                    for author in item.get("author", [])[:5]:
                        name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                        if name:
                            authors.append(name)

                    paper = {
                        "id": item.get("DOI", ""),
                        "title": item.get("title", [""])[0] if item.get("title") else "",
                        "abstract": (item.get("abstract", "") or "")[:500],
                        "year": year,
                        "authors": authors,
                        "citations": item.get("is-referenced-by-count", 0),
                        "url": item.get("URL", ""),
                        "venue": item.get("container-title", [""])[0] if item.get("container-title") else "",
                        "doi": item.get("DOI", ""),
                        "source": "crossref"
                    }
                    papers.append(paper)

                return {
                    "success": True,
                    "total": result.get("message", {}).get("total-results", len(papers)),
                    "papers": papers,
                    "provider": "crossref"
                }

            else:
                return {
                    "success": False,
                    "error": f"API请求失败: HTTP {response.status_code}",
                    "papers": []
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "papers": []
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"网络错误: {str(e)}",
                "papers": []
            }
        except Exception as e:
            logger.error(f"CrossRef搜索异常: {str(e)}")
            return {
                "success": False,
                "error": f"搜索异常: {str(e)}",
                "papers": []
            }

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文详情

        Args:
            paper_id: DOI

        Returns:
            论文详情
        """
        try:
            url = f"{self.api_url}/{paper_id}"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                item = result.get("message", {})

                # 提取年份
                year = None
                published = item.get("published-print", {}) or item.get("published-online", {})
                if published and "date-parts" in published:
                    date_parts = published["date-parts"]
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]

                # 提取作者
                authors = []
                for author in item.get("author", []):
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)

                return {
                    "success": True,
                    "paper": {
                        "id": item.get("DOI", ""),
                        "title": item.get("title", [""])[0] if item.get("title") else "",
                        "abstract": item.get("abstract", ""),
                        "year": year,
                        "authors": authors,
                        "citations": item.get("is-referenced-by-count", 0),
                        "url": item.get("URL", ""),
                        "venue": item.get("container-title", [""])[0] if item.get("container-title") else "",
                        "doi": item.get("DOI", ""),
                        "source": "crossref"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"获取论文详情失败: HTTP {response.status_code}",
                    "paper": None
                }

        except Exception as e:
            logger.error(f"获取CrossRef论文详情异常: {str(e)}")
            return {
                "success": False,
                "error": f"获取详情异常: {str(e)}",
                "paper": None
            }

    def is_available(self) -> bool:
        """检查CrossRef服务是否可用"""
        try:
            response = requests.get(
                self.api_url,
                params={"query": "test", "rows": 1},
                headers=self._get_headers(),
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


# ============================================================
# 文献搜索服务管理器
# ============================================================

class PaperSearchService:
    """
    文献搜索服务管理器

    统一管理多个文献搜索服务提供商
    """

    def __init__(self):
        """初始化文献搜索服务"""
        self.searchers: Dict[str, BasePaperSearcher] = {}
        self.primary_provider: str = ""
        self._load_config()

    def _load_config(self) -> None:
        """加载配置并初始化搜索器"""
        try:
            if get_settings:
                settings = get_settings()
                self.primary_provider = settings.get("paper_search.provider", "semantic_scholar")

                # 初始化Semantic Scholar
                ss_config = settings.get("paper_search.semantic_scholar", {})
                self.searchers["semantic_scholar"] = SemanticScholarSearcher(ss_config)

                # 初始化arXiv
                arxiv_config = settings.get("paper_search.arxiv", {})
                self.searchers["arxiv"] = ArxivSearcher(arxiv_config)

                # 初始化CrossRef
                crossref_config = settings.get("paper_search.crossref", {})
                self.searchers["crossref"] = CrossRefSearcher(crossref_config)

                logger.info(f"文献搜索服务初始化完成，主要提供商: {self.primary_provider}")
            else:
                logger.warning("配置管理器不可用，使用默认配置")
                self.searchers["semantic_scholar"] = SemanticScholarSearcher({})
                self.searchers["arxiv"] = ArxivSearcher({})
                self.searchers["crossref"] = CrossRefSearcher({})
                self.primary_provider = "semantic_scholar"

        except Exception as e:
            logger.error(f"文献搜索服务初始化失败: {str(e)}")

    def reload_config(self) -> None:
        """重新加载配置"""
        self.searchers.clear()
        self._load_config()

    def get_available_providers(self) -> List[str]:
        """获取可用的搜索服务提供商列表"""
        available = []
        for name, searcher in self.searchers.items():
            try:
                if searcher.is_available():
                    available.append(name)
            except Exception:
                pass
        return available

    def search(self, query: str, limit: int = 10, provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        搜索文献

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            provider: 指定的搜索服务提供商
            **kwargs: 其他参数

        Returns:
            搜索结果
        """
        if not query.strip():
            return {
                "success": False,
                "error": "搜索关键词不能为空",
                "data": None
            }

        # 确定使用的搜索器
        searcher = None
        provider_name = provider or self.primary_provider

        if provider_name in self.searchers:
            searcher = self.searchers[provider_name]

        # 如果指定的不可用，尝试其他
        if searcher is None:
            for name, search in self.searchers.items():
                searcher = search
                provider_name = name
                break

        if searcher is None:
            return {
                "success": False,
                "error": "没有可用的文献搜索服务",
                "data": None
            }

        # 执行搜索
        result = searcher.search(query, limit, **kwargs)

        if result.get("success"):
            papers = result.get("papers", [])
            return {
                "success": True,
                "message": f"找到 {len(papers)} 篇相关文献",
                "data": {
                    "query": query,
                    "total": result.get("total", len(papers)),
                    "papers": papers,
                    "provider": result.get("provider", provider_name)
                }
            }

        return {
            "success": False,
            "error": result.get("error", "搜索失败"),
            "data": None
        }

    def get_paper_details(self, paper_id: str, provider: str = None) -> Dict[str, Any]:
        """
        获取论文详情

        Args:
            paper_id: 论文ID
            provider: 提供商

        Returns:
            论文详情
        """
        provider_name = provider or self.primary_provider
        searcher = self.searchers.get(provider_name)

        if searcher is None:
            return {
                "success": False,
                "error": f"提供商 {provider_name} 不可用",
                "data": None
            }

        result = searcher.get_paper_details(paper_id)

        if result.get("success"):
            return {
                "success": True,
                "message": "获取论文详情成功",
                "data": result.get("paper")
            }

        return {
            "success": False,
            "error": result.get("error", "获取详情失败"),
            "data": None
        }

    def multi_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        多源搜索（从多个数据源搜索并合并结果）

        Args:
            query: 搜索关键词
            limit: 每个数据源的结果数量

        Returns:
            合并的搜索结果
        """
        all_papers = []
        errors = []

        for name, searcher in self.searchers.items():
            try:
                result = searcher.search(query, limit)
                if result.get("success"):
                    all_papers.extend(result.get("papers", []))
                else:
                    errors.append(f"{name}: {result.get('error', '未知错误')}")
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        if all_papers:
            # 按引用数排序
            all_papers.sort(key=lambda x: x.get("citations", 0), reverse=True)

            return {
                "success": True,
                "message": f"从多个数据源找到 {len(all_papers)} 篇文献",
                "data": {
                    "query": query,
                    "total": len(all_papers),
                    "papers": all_papers,
                    "provider": "multi"
                },
                "warnings": errors if errors else None
            }

        return {
            "success": False,
            "error": "所有搜索源均失败: " + "; ".join(errors),
            "data": None
        }


# ============================================================
# 全局文献搜索服务实例
# ============================================================

_paper_search_service: Optional[PaperSearchService] = None


def get_paper_search_service() -> PaperSearchService:
    """获取全局文献搜索服务实例"""
    global _paper_search_service
    if _paper_search_service is None:
        _paper_search_service = PaperSearchService()
    return _paper_search_service


def reload_paper_search_service() -> None:
    """重新加载文献搜索服务配置"""
    global _paper_search_service
    if _paper_search_service:
        _paper_search_service.reload_config()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("文献搜索服务测试")
    print("=" * 50)

    # 获取服务
    service = get_paper_search_service()

    # 显示可用的提供商
    providers = service.get_available_providers()
    print(f"\n可用的搜索服务: {providers}")

    # 测试搜索
    test_queries = [
        "machine learning",
        "deep learning neural network",
        "natural language processing transformer"
    ]

    print("\n" + "-" * 40)
    print("单源搜索测试：")
    for query in test_queries[:1]:
        print(f"\n搜索: {query}")
        result = service.search(query, limit=3)
        if result.get("success"):
            print(f"找到 {result['data']['total']} 篇文献")
            for paper in result['data']['papers'][:3]:
                print(f"  - {paper['title'][:60]}... ({paper.get('year', 'N/A')})")
                print(f"    作者: {', '.join(paper.get('authors', [])[:3])}")
                print(f"    引用: {paper.get('citations', 0)}")
        else:
            print(f"错误: {result.get('error')}")

    print("\n" + "-" * 40)
    print("多源搜索测试：")
    result = service.multi_search("transformer attention mechanism", limit=2)
    if result.get("success"):
        print(f"找到 {result['data']['total']} 篇文献")
        for paper in result['data']['papers'][:5]:
            print(f"  - [{paper.get('source')}] {paper['title'][:50]}...")
    else:
        print(f"错误: {result.get('error')}")
