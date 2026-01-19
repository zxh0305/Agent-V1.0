# 智能任务处理 Agent 平台 V3.0 配置说明

## 概述

本文档说明了智能Agent平台的所有配置项，配置文件为 `settings.yaml`。

## 配置文件位置

配置文件位于项目根目录：`settings.yaml`

## 配置项详解

### 1. 大模型配置 (llm)

用于配置AI大模型，支持意图识别、智能问答、文本摘要等功能。

```yaml
llm:
  # 选择使用的提供商: openai / qwen / ollama / zhipu
  provider: "qwen"

  # OpenAI 配置
  openai:
    api_key: "sk-your-openai-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 2000

  # 通义千问配置 (阿里云)
  qwen:
    api_key: "sk-your-qwen-api-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: "qwen-turbo"
    temperature: 0.7
    max_tokens: 2000

  # Ollama 本地模型配置
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen2:7b"
    temperature: 0.7

  # 智谱AI配置
  zhipu:
    api_key: "your-zhipu-api-key"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    model: "glm-4-flash"
    temperature: 0.7
```

**获取API密钥：**
- **OpenAI**: https://platform.openai.com/api-keys
- **通义千问**: https://dashscope.console.aliyun.com/
- **智谱AI**: https://open.bigmodel.cn/
- **Ollama**: 本地部署，无需API密钥

### 2. 翻译服务配置 (translate)

用于配置文本翻译功能。

```yaml
translate:
  # 选择提供商: baidu / youdao / llm
  provider: "baidu"

  # 百度翻译API
  baidu:
    app_id: "your-baidu-app-id"
    secret_key: "your-baidu-secret-key"
    api_url: "https://fanyi-api.baidu.com/api/trans/vip/translate"

  # 有道翻译API
  youdao:
    app_key: "your-youdao-app-key"
    app_secret: "your-youdao-app-secret"
    api_url: "https://openapi.youdao.com/api"

  # 使用大模型进行翻译
  llm:
    enabled: true
```

**获取API密钥：**
- **百度翻译**: https://fanyi-api.baidu.com/
- **有道翻译**: https://ai.youdao.com/

### 3. 文献搜索配置 (paper_search)

用于配置学术文献搜索功能。

```yaml
paper_search:
  # 选择提供商: semantic_scholar / arxiv / crossref
  provider: "semantic_scholar"

  # Semantic Scholar API (免费)
  semantic_scholar:
    api_url: "https://api.semanticscholar.org/graph/v1"
    api_key: ""  # 可选，申请后可提高请求限制

  # arXiv API (免费，无需API Key)
  arxiv:
    api_url: "http://export.arxiv.org/api/query"

  # CrossRef API (免费)
  crossref:
    api_url: "https://api.crossref.org/works"
    email: "your-email@example.com"  # 可选，提供后可获得更好的服务
```

**说明：**
- Semantic Scholar、arXiv、CrossRef 均为免费服务
- 配置邮箱可获得更高的API请求配额

### 4. 代码执行配置 (code_execution)

```yaml
code_execution:
  timeout: 30  # 执行超时时间(秒)
  max_output_length: 10000  # 最大输出长度
  allowed_modules:  # 允许导入的模块
    - math
    - random
    - statistics
    - datetime
    - json
    - collections
    - itertools
    - functools
    - re
    - string
    - numpy
    - pandas
```

### 5. 文件处理配置 (file_processing)

```yaml
file_processing:
  upload_dir: "uploads"  # 上传目录
  export_dir: "exports"  # 导出目录
  max_file_size_mb: 50   # 最大文件大小(MB)
  allowed_extensions:    # 允许的文件类型
    - pdf
    - xlsx
    - xls
    - csv
    - docx
    - doc
    - txt
```

### 6. 界面配置 (ui)

```yaml
ui:
  page_title: "智能Agent平台 V3.0"
  page_icon: "🤖"
  theme: "light"
  show_debug_info: false
  max_history_display: 20
```

### 7. 系统配置 (system)

```yaml
system:
  log_level: "INFO"  # 日志级别: DEBUG/INFO/WARNING/ERROR
  log_dir: "logs"    # 日志目录
  enable_cache: true
  cache_ttl_seconds: 3600
```

## 快速配置指南

### 最小化配置

如果只想使用基本功能，只需配置大模型即可：

```yaml
llm:
  provider: "qwen"
  qwen:
    api_key: "your-api-key-here"
```

### 推荐配置

建议同时配置大模型和翻译服务：

```yaml
llm:
  provider: "qwen"
  qwen:
    api_key: "your-qwen-api-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: "qwen-turbo"

translate:
  provider: "baidu"
  baidu:
    app_id: "your-baidu-app-id"
    secret_key: "your-baidu-secret-key"
```

## 配置热更新

修改 `settings.yaml` 后，可以通过以下方式重新加载配置：

1. 重启应用
2. 在代码中调用 `reload_settings()` 函数

## 注意事项

1. **安全性**：不要将包含API密钥的配置文件提交到公开仓库
2. **API限制**：注意各服务的API调用频率限制
3. **费用**：部分服务（如OpenAI、百度翻译）可能产生费用，请注意用量
4. **网络**：部分服务需要网络访问，请确保网络畅通

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 检查网络连接
   - 查看日志获取详细错误信息

2. **翻译服务不可用**
   - 确认已配置百度翻译或有道翻译的API密钥
   - 检查API配额是否用尽

3. **LLM服务无响应**
   - 检查大模型提供商配置
   - 确认API密钥有效
   - 对于Ollama，确保服务已启动

### 日志查看

日志文件位于 `logs/` 目录，可查看详细的错误信息。
