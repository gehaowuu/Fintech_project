# Finance Middleware

金融市场数据分析中间件，整合宏观、流动性、情绪和技术面四大维度数据，通过LLM生成深度市场分析报告，为买方机构的宏观策略分析师和跨资产投资组合经理提供决策支持。

## ✨ 核心特性

- 📊 **四大分析模块**
  - **Fundamentals（宏观基本面）**：利率、利差、美元指数、经济日历、财经新闻
  - **Liquidity（流动性）**：美联储资产负债表、净流动性、信用利差
  - **Sentiment（市场情绪）**：恐惧贪婪指数、VIX、Put/Call比率、市场广度
  - **Technicals（技术面）**：主要资产的技术指标、SPY期权数据

- 🤖 **LLM深度分析**
  - 集成Google Gemini 3 Pro，支持thinking模式
  - 单模块分析：每个模块独立深度分析
  - 综合分析：跨模块整合，生成多资产交易观点
  - 对话功能：基于快照的多轮问答

- 🏷️ **智能标签系统**
  - 启发式规则自动生成Regime标签
  - LLM可覆盖标签并提供详细推理
  - 数据质量监控和缺失字段追踪

- 📁 **数据持久化**
  - 快照数据自动保存到磁盘
  - LLM分析结果持久化存储
  - 支持历史快照查询和分析

## 🚀 快速开始

### 环境要求

- Python 3.10 或 3.11
- 必要的API密钥（见下方配置说明）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

复制 `env.example` 为 `.env` 并填入必要的API密钥：

```bash
# 必需配置
FRED_API_KEY=your_fred_api_key          # FRED经济数据API（免费注册）
GEMINI_API_KEY=your_gemini_api_key      # Google Gemini API（LLM分析）

# 可选配置
APP_PASSWORD=your_password               # 后端访问密码（不设置则不鉴权）
TE_CLIENT_KEY=your_te_key                # TradingEconomics（经济日历）
TE_CLIENT_SECRET=your_te_secret
DATA_DIR=data                            # 数据存储目录
CORS_ALLOW_ORIGINS=http://localhost:3000 # 前端域名（生产环境必设）

# LLM配置（可选）
GEMINI_MODEL=gemini-3-pro-preview        # 默认模型
GEMINI_THINKING_LEVEL=high               # low/high，默认high
```

### 本地运行

**方式1：直接运行**
```bash
python main_v3.py
```

**方式2：使用uvicorn（推荐开发）**
```bash
uvicorn main_v3:app --reload --host 0.0.0.0 --port 8000
```

访问API文档：http://127.0.0.1:8000/docs

### PowerShell测试示例

```powershell
$base = "http://127.0.0.1:8000"

# 1. 健康检查
Invoke-RestMethod -Method Get -Uri "$base/health"

# 2. 登录
$login = Invoke-RestMethod -Method Post -Uri "$base/auth/login" `
    -ContentType "application/json" `
    -Body (@{ password = "your_password" } | ConvertTo-Json)
$token = $login.token

# 3. 生成快照
$snap = Invoke-RestMethod -Method Post -Uri "$base/v3/snapshot/run" `
    -Headers @{ Authorization = "Bearer $token" }
$snapshot_id = $snap.id

# 4. 获取快照
Invoke-RestMethod -Method Get -Uri "$base/v3/snapshot/$snapshot_id" `
    -Headers @{ Authorization = "Bearer $token" }

# 5. 模块分析
Invoke-RestMethod -Method Post -Uri "$base/v3/analysis/module" `
    -Headers @{ Authorization = "Bearer $token" } `
    -ContentType "application/json" `
    -Body (@{ snapshot_id = $snapshot_id; module = "fundamentals"; call_llm = $true } | ConvertTo-Json)

# 6. 综合分析
Invoke-RestMethod -Method Post -Uri "$base/v3/analysis/overall" `
    -Headers @{ Authorization = "Bearer $token" } `
    -ContentType "application/json" `
    -Body (@{ snapshot_id = $snapshot_id; call_llm = $true; include_module_summaries = $true } | ConvertTo-Json)
```

## 📡 API接口

### 公开接口

- `GET /health` - 健康检查
- `GET /v1/context?date=YYYY-MM-DD` - 获取上下文数据（无需鉴权）

### 受保护接口（需要Bearer Token）

- `POST /auth/login` - 登录获取token
- `POST /auth/logout` - 登出
- `POST /v3/snapshot/run?date=YYYY-MM-DD` - 生成市场快照
- `GET /v3/snapshot/{snapshot_id}` - 获取快照详情
- `POST /v3/analysis/module` - 单模块LLM分析
- `POST /v3/analysis/overall` - 综合LLM分析
- `POST /v3/chat` - 基于快照的对话

详细API文档请访问：http://127.0.0.1:8000/docs

## 📊 数据格式化工具

项目提供了 `format_snapshot.py` 工具，可将JSON格式的快照数据转化为可读性强的文本报告。

### 使用方法

```bash
# 格式化指定快照
python format_snapshot.py <snapshot_id>

# 格式化最新快照
python format_snapshot.py --latest

# 格式化所有快照
python format_snapshot.py --all

# 列出所有快照
python format_snapshot.py --list
```

### 输出示例

报告会保存到 `formatted_reports/` 目录，包含：
- 📅 快照元数据
- 🏷️ 市场体制标签摘要
- 📈 各模块详细数据（表格形式）
- ⚠️ 数据质量报告
- 🤖 LLM分析结果（如已生成）

## 🚢 部署到Render

### 1. 创建Web Service

- **Service Type**: Web Service
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main_v3:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`

### 2. 配置环境变量

在Render控制台设置以下环境变量：

**必需：**
- `FRED_API_KEY` - FRED API密钥
- `GEMINI_API_KEY` - Gemini API密钥
- `APP_PASSWORD` - 访问密码（生产环境强烈建议设置）
- `DATA_DIR` - 数据存储目录（如 `/var/data`，需挂载Persistent Disk）

**可选：**
- `TE_CLIENT_KEY` / `TE_CLIENT_SECRET` - TradingEconomics凭证
- `CORS_ALLOW_ORIGINS` - 前端域名（逗号分隔）
- `GEMINI_MODEL` - LLM模型名称
- `GEMINI_THINKING_LEVEL` - 思考级别

### 3. 挂载持久化存储

1. 在Render控制台：Disks -> Add Disk
2. Mount Path设为 `/var/data`
3. 在环境变量中设置 `DATA_DIR=/var/data`

这样快照和LLM分析结果会持久化保存，服务重启后仍可访问。

### 4. 注意事项

- Render免费版会在长时间无请求后休眠，可用定时任务ping `/health`保活
- 多实例部署时session不共享，建议先保持单实例
- 生产环境务必设置 `APP_PASSWORD` 和 `CORS_ALLOW_ORIGINS`

## 📚 相关文档

- **[后端对接指导](./主要功能说明&render前后端对接指导_后端部分.txt)** - 详细的API说明、数据结构、前端对接指南
- **[数据与判定标准](./主要数据&判定标准介绍及LLM分析框架说明.txt)** - 数据字段详解、判定标准、LLM分析框架

## 🗂️ 项目结构

```
finance-middleware/
├── main_v3.py                          # 主服务文件（FastAPI应用）
├── format_snapshot.py                  # 数据格式化工具
├── requirements.txt                     # Python依赖
├── env.example                         # 环境变量示例
├── readme.md                           # 本文件
├── data/                               # 数据存储目录
│   ├── snapshots/                      # 快照JSON文件
│   └── llm_logs/                       # LLM分析日志
│       ├── module/                     # 模块分析日志
│       └── overall/                    # 综合分析日志
├── formatted_reports/                  # 格式化报告输出目录
└── 主要功能说明&render前后端对接指导_后端部分.txt
```

## 🔑 API密钥获取

### FRED API（必需）
1. 访问 https://fred.stlouisfed.org/
2. 注册免费账号
3. 在账户设置中获取API Key

### Google Gemini API（必需）
1. 访问 https://aistudio.google.com/app/apikey
2. 创建API密钥
3. 复制密钥到 `.env` 文件

### TradingEconomics（可选）
1. 访问 https://tradingeconomics.com/
2. 注册免费账号
3. 在API设置中获取Client Key和Secret

## 🛠️ 技术栈

- **框架**: FastAPI + Uvicorn
- **数据处理**: Pandas + NumPy
- **数据源**: 
  - FRED API（美联储经济数据）
  - Yahoo Finance（市场数据）
  - TradingEconomics（经济日历）
  - CNN Fear & Greed Index（市场情绪）
  - CBOE（期权数据）
  - Google News RSS（财经新闻）
- **LLM**: Google Gemini 3 Pro
- **存储**: 文件系统（JSON格式）

## 📝 数据源说明

### 数据更新频率
- **FRED数据**: 日度/周度更新（工作日）
- **Yahoo Finance**: 实时/日度更新
- **经济日历**: 每日更新
- **市场广度**: 盘中实时更新

### 数据质量
- 系统会自动标记缺失字段
- 数据质量报告包含在快照中
- LLM分析会考虑数据完整性

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 🔗 相关链接

- [FRED API文档](https://fred.stlouisfed.org/docs/api/)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [Render部署文档](https://render.com/docs)
- [Gemini API文档](https://ai.google.dev/docs)

---

**版本**: v3.0  
**最后更新**: 2025-12-18
