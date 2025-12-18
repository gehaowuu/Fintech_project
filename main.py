import os
import logging
import time
import json
import io
import ssl
import datetime as dt
from datetime import datetime, timedelta, date, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Cookie, Response, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
from fredapi import Fred
from google import genai
from google.genai import types

# 确保无论从哪里启动（uvicorn/app-dir）都能加载到同目录下的 .env
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Finance Middleware", version="0.1.0")

# -----------------------------------------------------------------------------
# LLM Prompt Templates (统一管理，便于前端/产品对齐)
#
# 说明：
# - A) GLOBAL_SYSTEM_PROMPT：所有端点共用的系统提示词（硬约束/风格/目标）
# - B) build_module_user_prompt：按钮1「模块AI简析」使用的用户提示词模板
# - C) build_overall_user_prompt：按钮2「综合AI专业分析」使用的用户提示词模板
# - D) build_chat_system_prompt：Chat 端点使用的系统提示词（可注入 last_overall_report）
#

GLOBAL_SYSTEM_PROMPT = """
你是一名买方宏观与跨资产策略分析师（Multi-Asset Macro PM）。你的输出将用于投委会/交易晨会决策支持。

硬约束（必须遵守）：
1) 严格基于输入数据推断，不得臆造未提供的事实、数值、日期、网页内容；若未提供/无法确认，必须明确写“缺失/无法确认”。
2) 如输入包含 supplemental（外部补充数据），仅当 supplemental 字段带有“来源URL + 抓取日期/数据日期”且与快照日期逻辑一致时才可使用；否则视为不可用并说明原因。
3) 允许你对输入中的“启发式标签(heuristic_label)”进行覆盖，但必须给出：scorecard 打分、覆盖理由、以及覆盖所依赖的关键证据。
4) 输出必须结构化、可复核：每个结论都要能追溯到“指标 → 推理链条 → 结论”。
5) 输出中文，要点化；避免空话与泛泛科普；避免给出“下一步必须盯的3个数据点”这种独立章节。
   如有缺失数据，用条件句表达：例如“后续关注X；若X上行/下行至Y，将提高/降低Z结论置信度”。

目标：
- 给出当日市场环境（regime）与跨资产传导逻辑；
- 在不完整数据下，仍完成“最低可用”的分析闭环，并清晰标注不确定性来自何处。
""".strip()

# -----------------------------------------------------------------------------
# CORS (给前端调用用)
#
# 浏览器环境（Render 静态站点/自建前端）请求后端通常是跨域的，需要开启 CORS。
# 生产环境建议把 CORS_ALLOW_ORIGINS 设置为你的前端域名列表（逗号分隔）。
#
# 例：
# - CORS_ALLOW_ORIGINS=https://your-frontend.onrender.com,https://www.yourdomain.com
#
CORS_ALLOW_ORIGINS_RAW = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if CORS_ALLOW_ORIGINS_RAW:
    CORS_ALLOW_ORIGINS = [o.strip() for o in CORS_ALLOW_ORIGINS_RAW.split(",") if o.strip()]
else:
    # 默认放开，方便本地/快速联调；上线请务必收紧
    CORS_ALLOW_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,  # 本项目推荐前端用 Bearer token；跨站 cookie 在现代浏览器更容易踩坑
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Session & Snapshot Management
#
# The original middleware exposed only unauthenticated endpoints.  To prevent
# anyone on the public internet from exhausting your data providers or AI
# quotas, this patch introduces a simple password‐based login.  Upon
# successful login a secure random token is issued and stored in memory.
# Subsequent protected endpoints require the session cookie.  See `/auth/login`
# and `require_auth()` below for details.

# In addition, a snapshot cache is maintained to avoid recomputing the same
# market snapshot repeatedly.  Each snapshot stores the raw modules, the
# heuristic labels and a set of lightweight signals/data quality summaries
# intended for LLM consumption.

APP_PASSWORD = os.getenv("APP_PASSWORD")

# In-memory session store: maps session token to a dictionary containing
# metadata (creation time).  In a real application you might persist this
# elsewhere or implement expiry.  Here we keep it simple.
_sessions: Dict[str, Dict[str, Any]] = {}

# In-memory snapshot cache (cache only): maps snapshot_id to a snapshot object.
# IMPORTANT: The source-of-truth is the filesystem (DATA_DIR). This in-memory
# dict is only a performance cache for hot snapshots.
_snapshot_cache: Dict[str, Dict[str, Any]] = {}

#
# Persistent storage (Render Persistent Disk)
# - Render 上建议配置 Persistent Disk mount 到 /var/data，然后设置：
#   DATA_DIR=/var/data
# - 本地不设置则默认写到项目目录下的 ./data
#
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR") or os.path.join(_BASE_DIR, "data"))
SNAPSHOT_STORE_DIR = os.path.join(DATA_DIR, "snapshots")
LLM_LOG_DIR = os.path.join(DATA_DIR, "llm_logs")

PERSIST_ENABLED = True
try:
    os.makedirs(SNAPSHOT_STORE_DIR, exist_ok=True)
    os.makedirs(LLM_LOG_DIR, exist_ok=True)
except Exception as exc:
    # 若磁盘不可写，降级到内存（仍可跑通，但重启会丢数据）
    logging.warning("DATA_DIR 不可写，持久化被禁用: %s", exc)
    PERSIST_ENABLED = False


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically (safe for crash/restart)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _snapshot_path(snapshot_id: str) -> str:
    return os.path.join(SNAPSHOT_STORE_DIR, f"{snapshot_id}.json")


def _save_snapshot_to_disk(snapshot: Dict[str, Any]) -> None:
    if not PERSIST_ENABLED:
        return
    sid = str(snapshot.get("id") or "").strip()
    if not sid:
        return
    try:
        snapshot = {**snapshot, "persisted_ts": time.time()}
        _atomic_write_json(_snapshot_path(sid), snapshot)
    except Exception as exc:
        logging.warning("snapshot 落盘失败（sid=%s）: %s", sid, exc)


def _load_snapshot_from_disk(snapshot_id: str) -> Optional[Dict[str, Any]]:
    if not PERSIST_ENABLED:
        return None
    path = _snapshot_path(snapshot_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        logging.warning("snapshot 读取失败（sid=%s）: %s", snapshot_id, exc)
        return None


def _get_snapshot(snapshot_id: str) -> Optional[Dict[str, Any]]:
    """Get snapshot by id (memory cache first, then disk)."""
    snap = _snapshot_cache.get(snapshot_id)
    if snap:
        return snap
    snap = _load_snapshot_from_disk(snapshot_id)
    if snap:
        _snapshot_cache[snapshot_id] = snap
    return snap


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not PERSIST_ENABLED:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_llm_event(kind: str, snapshot_id: str, record: Dict[str, Any], module_name: Optional[str] = None) -> None:
    """Append an LLM event record to disk (JSONL)."""
    if not PERSIST_ENABLED:
        return
    sid = (snapshot_id or "").strip()
    if not sid:
        return
    try:
        base = os.path.join(LLM_LOG_DIR, kind, sid)
        if module_name:
            path = os.path.join(base, f"{module_name}.jsonl")
        else:
            path = os.path.join(base, "events.jsonl")
        record = {"ts": time.time(), **record}
        _append_jsonl(path, record)
    except Exception as exc:
        logging.warning("LLM log 失败(kind=%s sid=%s): %s", kind, snapshot_id, exc)


FRED_API_KEY = os.getenv("FRED_API_KEY")
TE_CLIENT_KEY = os.getenv("TE_CLIENT_KEY")
TE_CLIENT_SECRET = os.getenv("TE_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_DEFAULT_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3-pro-preview").strip() or "gemini-3-pro-preview"
GEMINI_DEFAULT_THINKING_LEVEL = (os.getenv("GEMINI_THINKING_LEVEL") or "high").strip().lower()
if GEMINI_DEFAULT_THINKING_LEVEL not in {"low", "high"}:
    GEMINI_DEFAULT_THINKING_LEVEL = "high"

def _normalize_gemini_model_name(model: Optional[str]) -> str:
    """Normalize user-provided model names.

    FastAPI /docs 会把可选字符串字段展示为占位符 \"string\"；如果用户不改就提交，
    会导致调用 `models/string` 从而报 404。这里将这些占位符/空值统一回退到默认模型。
    """
    if model is None:
        return GEMINI_DEFAULT_MODEL
    m = str(model).strip()
    if not m:
        return GEMINI_DEFAULT_MODEL
    if m.lower() in {"string", "none", "null", "undefined"}:
        return GEMINI_DEFAULT_MODEL
    return m


def _should_retry_gemini_exception(exc: Exception) -> bool:
    """Heuristic retry policy for transient network/TLS issues.

    We only retry on low-level transport errors (e.g. SSL handshake, connect
    reset, timeouts). API errors like 4xx/5xx are handled by the SDK and should
    not be retried here.
    """
    if isinstance(exc, ssl.SSLError):
        return True
    # httpx/httpcore transport errors (google-genai uses httpx under the hood)
    try:
        import httpx  # type: ignore

        if isinstance(exc, httpx.HTTPError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "handshake failure",
            "sslv3_alert_handshake_failure",
            "tls",
            "ssl",
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "temporarily unavailable",
        )
    )


def _format_signal_value(val: Any) -> str:
    """Format values for prompts to avoid ugly float repr like 0.6199999997."""
    if val is None:
        return "n/a"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        # keep 4 decimals for rates/spreads; 2 for others is too lossy sometimes
        return _fmt_num(float(val), 4)
    if isinstance(val, dt.date):
        return str(val)
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_format_signal_value(x) for x in val[:20]) + (", ..." if len(val) > 20 else "") + "]"
    if isinstance(val, dict):
        items = []
        for k, v in list(val.items())[:30]:
            items.append(f"{k}:{_format_signal_value(v)}")
        suffix = ", ..." if len(val) > 30 else ""
        return "{" + ", ".join(items) + suffix + "}"
    return str(val)


def build_module_user_prompt(
    module_name: str,
    as_of_date: str,
    signals: List[Dict[str, Any]],
    heuristic_label: Optional[str],
    missing: List[str],
    supplemental: Optional[str] = None,
) -> str:
    """按钮1：模块短评 Prompt（四模块通用模板，按规范输出）。"""
    module_lower = (module_name or "").lower().strip()
    module_display = module_lower or module_name

    # signals -> text
    parts: List[str] = []
    for s in signals:
        name = s.get("name")
        val = s.get("value")
        if isinstance(val, dict):
            # compact dict
            inner = ", ".join(f"{k}:{_format_signal_value(v)}" for k, v in val.items())
            parts.append(f"{name}({inner})")
        else:
            parts.append(f"{name}={_format_signal_value(val)}")
    signals_text = "; ".join(parts) if parts else "(无)"

    missing_text = ", ".join(missing) if missing else "无"
    supplemental_text = (supplemental or "").strip() or "（空）"
    heuristic_text = heuristic_label if heuristic_label is not None else "（无）"

    # candidate set by module
    candidate_map = {
        "fundamentals": "Hawkish / Neutral / Dovish",
        "liquidity": "Expansionary / Neutral / Tightening",
        "sentiment": "Risk-On / Neutral / Risk-Off",
        "technicals": "Uptrend / Range / Downtrend",
    }
    candidates = candidate_map.get(module_lower, "（按模块自定，但必须从预定义集合中选）")

    # Suggested scorecard anchors (helps controllability / dashboard)
    scorecard_hint_map = {
        "fundamentals": [
            "期限结构（10Y-2Y、曲线形态）",
            "政策定价（FFR-2Y 或相关替代）",
            "美元（DXY）",
            "交叉验证（Real10Y / Breakeven / 信用利差 OAS 若有；缺失则 NA）",
        ],
        "liquidity": [
            "净流动性水平（Net Liquidity）",
            "净流动性变化（Δ4w）",
            "组成项驱动（WALCL/RRP/TGA）",
            "交叉验证（信用比率 HYG/IEI 或 OAS 若有；缺失则 NA）",
        ],
        "sentiment": [
            "Fear&Greed（FGI）",
            "VIX 期限结构（VIX vs VIX3M / contango/backwardation）",
            "Put/Call（或其均值）",
            "市场广度（涨跌家数比 A/D Ratio、新高新低比；若有）",
            "交叉验证（风险价差 SPY-XLU / HYG-IEF / BTC-Gold 之一；缺失则 NA）",
        ],
        "technicals": [
            "趋势（MA20/50/200 距离 + trend_label）",
            "波动/结构（ATR%、boll_width）",
            "广度（RSP-SPX）",
            "期权指标（SPY期权PCR成交量/OI、IV偏斜；若有）",
            "交叉验证（风格比 XLK/XLP 或相关；缺失则 NA）",
        ],
    }
    hint_lines = scorecard_hint_map.get(module_lower, [])
    hint_text = "\n".join([f"  - {x}" for x in hint_lines]) if hint_lines else "  - （由你自行选取 2-6 个关键指标）"

    return f"""
你将分析单个模块：{module_display}
快照日期：{as_of_date}

【输入】
- signals（原始信号）：{signals_text}
- heuristic_label（启发式标签）：{heuristic_text}
- data_quality_missing（缺失项）：{missing_text}
- supplemental（外部补充数据，可能为空）：{supplemental_text}

【输出要求：必须按以下标题与顺序输出】

1) 数据可用性与关键不确定性（≤6条要点）
- 说明哪些信号日期不一致/口径不清会影响结论
- 若 supplemental 存在：只在“来源+日期可核对”时使用；否则写明“无法确认而未采用”

2) 模块内部推理链条（至少2条，写成因果箭头）
- 每条链条格式：
  - 证据（指标/方向） → 机制解释（简短） → 对{module_display}的含义（偏宽松/偏紧/偏风险偏好等）

3) Regime 判定（必须给 scorecard）
- 候选集合（按模块自动选择）：{candidates}
- Scorecard（每条：+1 / 0 / -1 / NA，并用一句话解释；建议选用如下锚点）：
{hint_text}
- 计算：总分 = …（你自己算）
- 最终 Regime：…（必须来自候选集合）
- 置信度（0-100）：…（必须解释置信度来自“数据完整度+一致性”）
- Override Check：是否与 heuristic_label 一致？若不一致：必须解释为何覆盖，以及覆盖依赖的关键证据；同时写明“若补齐哪些数据将更确定”。

4) 对主要风险资产的含义（只写条件句，2-4条）
- 格式：后续关注X；若X走向Y，则该模块对风险资产的影响将更偏向Z（例如 risk-on/risk-off/高波动/轮动）。
""".strip()


def build_overall_user_prompt(
    as_of_date: str,
    prompt_context_text: str,
    labels: Dict[str, str],
    module_reports_or_empty: str,
    supplemental_overall: Optional[str] = None,
) -> str:
    """按钮2：综合专业版 Prompt（固定结构 + 可复核链条 + 多资产细分）。"""
    labels_text = (
        f"- Macro: {labels.get('macro_regime')}\n"
        f"- Liquidity: {labels.get('liquidity_regime')}\n"
        f"- Sentiment: {labels.get('sentiment_regime')}\n"
        f"- Technical: {labels.get('technical_regime')}"
    )
    supplemental_text = (supplemental_overall or "").strip() or "（空）"
    module_reports_text = (module_reports_or_empty or "").strip() or "（无：尚未运行按钮1模块短评）"

    return f"""
你将获得一份市场快照（时间标签：{as_of_date}）。请仅基于该快照与可核对的 supplemental 完成分析。

【输入】
A) 快照汇总（你必须视为唯一事实来源）：
{prompt_context_text}

B) 既有标签（可能不准，允许覆盖但要给证据）：
{labels_text}

C) 可选：模块AI短评输出（如果上一步按钮1已经跑过，则会提供；没有也没关系）：
{module_reports_text}

D) supplemental（外部补充数据，可能为空；仅可在“来源+日期可核对”时使用）：
{supplemental_text}

【输出：必须按以下固定框架、标题一致、按序输出】

1) 数据可用性与“结论可靠性边界”
1.1 可用数据概况
- 用一句话概括：当前快照覆盖了哪些维度（宏观/流动性/情绪/技术）以及缺什么
1.2 关键缺失与影响（用“影响链条”写，不要列“下一步必须盯3个”）
- 对每个关键缺失，写：缺失项 → 会影响哪条推理链 → 可能导致结论偏向哪里
- 若 supplemental 存在但不可核对：说明“未采用”的原因（日期/来源/口径）

2) Regime 拼图（跨模块一致性评估）
2.1 四模块最终 regime（若模块短评为空，你需用快照自行判定并给 scorecard 逻辑）
- Fundamentals：…（简短一句话理由）
- Liquidity：…
- Sentiment：…
- Technicals：…
2.2 一致性评分（0-100）与冲突解释
- 评分规则（你自洽即可）：一致性高 + 数据完整 → 高分；相互打架或关键缺失多 → 低分
- 用要点列出：最主要的 1-3 个“打架点”（例如：油价通缩信号 vs 黄金强势；BTC弱 vs 股强）

3) 跨资产传导图谱（必须是“可复核链条”，至少3条）
- 每条链条写成：起点指标（含方向） → 中介变量（含方向） → 终点资产（含方向）
- 链条类型至少覆盖：
  A) 利率/实际利率 → 美元 → 黄金（或相反）
  B) 流动性 → 权益/信用/加密（区分敏感度）
  C) 风险偏好/波动结构 → 资产内部轮动（小盘/等权/行业风格）

4) 市场环境结论（专业、可交易，但不要求具体标的）
4.1 Base Case（1-2周）
- 用一句话给出：risk-on / risk-off / 区间轮动（必须选一个主基调）
- 用条件句描述“可靠性边界”：后续关注X；若X出现Y，则Base Case 更强/更弱/转向
4.2 1-3个月主线（增长/通胀/政策/流动性四要素）
- 你必须明确：主导变量是哪一个（例如政策路径/流动性/增长担忧/通胀回落）
- 同样用条件句描述：若出现…则主线切换
4.3 Bull / Bear 场景（各2-3条触发条件 + 1条应对原则）
- 触发条件必须落在“你已有的数据维度”或“可核对的 supplemental”上；缺失则写“需数据验证”

5) 多资产细分分析（给你固定的专业框架）
你必须分别输出：债市、股市（分3-5个行业/风格桶）、黄金/白银、BTC。
每个资产用同一模板输出（必须按序）：
A) 当前驱动拆解（宏观/流动性/技术或情绪各1条，总计3条）
B) 当前偏向（多/空/中性/区间）+ 置信度（0-100）
C) 关键条件句（2条）：后续关注X；若X→Y，则偏向更…/转为…
D) 风控原则（2条）：仓位纪律/对冲思路/避免的行为（不许空泛）

6) 结论摘要（用于Dashboard展示，≤10行）
- 用“结论-证据”对照写：每行=一个结论 + 1个最关键证据
- 禁止出现“下一步必须盯的3个数据点”这种独立段落
""".strip()


def build_chat_system_prompt(
    as_of_date: str,
    prompt_context_text: str,
    last_overall_report_text_or_summary: Optional[str],
) -> str:
    """Chat Prompt（可选：把按钮2输出接进去，形成“连续问答”体验）。"""
    report_text = (last_overall_report_text_or_summary or "").strip() or "（无：尚未生成综合专业报告）"
    return (
        GLOBAL_SYSTEM_PROMPT
        + "\n\n"
        + f"""
你是投研助理，正在对同一份快照（{as_of_date}）提供持续问答支持。
你必须严格基于以下两类信息回答：
1) 快照原始数据（snapshot）
2) 已生成的综合专业报告（last_overall_report）
不得引入外部未提供信息，不得臆造网页内容。

【snapshot】
{prompt_context_text}

【last_overall_report】
{report_text}

对用户问题的回答规则：
- 先给结论（1-2句），再给证据点（最多3条，均需来自 snapshot 或 last_overall_report）
- 若用户问到报告未覆盖且快照也没有的数据：明确说缺失，并用条件句说明“若未来补齐该数据，判断可能如何变化”
""".strip()
    )

fred_client: Optional[Fred] = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# Gemini API 配置
gemini_client: Optional[Any] = None
if GEMINI_API_KEY:
    try:
        try:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except TypeError:
            # 兼容部分版本不支持显式 api_key 参数的情况
            # 新版 SDK 默认读取环境变量 GOOGLE_API_KEY；这里兼容你当前的 GEMINI_API_KEY 命名
            os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
            gemini_client = genai.Client()
        logging.info(f"Gemini API 已配置成功，使用模型: {GEMINI_DEFAULT_MODEL}")
    except Exception as e:
        logging.warning(f"Gemini API 配置失败: {e}")
        gemini_client = None
else:
    logging.warning("未检测到 GEMINI_API_KEY，LLM 分析功能将不可用。")

if not FRED_API_KEY:
    logging.warning("未检测到 FRED_API_KEY，FRED 数据将被跳过。")
if not TE_CLIENT_KEY or not TE_CLIENT_SECRET:
    logging.warning("未检测到 TradingEconomics 凭证，经济日历将返回空列表。")


# -------- Gemini LLM helpers --------
def call_gemini(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 10240,
    model: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    调用 Gemini API 生成分析内容。
    
    返回 (response_text, error_message)
    - 成功时: (回复内容, None)
    - 失败时: (None, 错误信息)
    """
    if not gemini_client:
        return None, "Gemini API 未配置（缺少 GEMINI_API_KEY）"
    
    try:
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        
        model_name = _normalize_gemini_model_name(model)
        tl = (thinking_level or GEMINI_DEFAULT_THINKING_LEVEL).strip().lower()
        if tl not in {"low", "high"}:
            tl = GEMINI_DEFAULT_THINKING_LEVEL

        # 优先使用强类型 Config；如遇到版本差异，再回退到 dict
        try:
            cfg = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=tl),
            )
        except Exception:
            cfg = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
                "thinking_config": {"thinking_level": tl},
            }

        # 网络/SSL 在部分环境会偶发握手失败；这里做轻量重试，避免一次性失败让体验很差
        max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "2")))
        last_exc: Optional[Exception] = None
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=cfg,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries or not _should_retry_gemini_exception(exc):
                    raise
                sleep_s = 0.8 * (2**attempt)
                logging.warning(
                    "Gemini 调用失败（%s），%.1fs 后重试（%d/%d）",
                    type(exc).__name__,
                    sleep_s,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_s)
        if last_exc is not None or response is None:
            raise last_exc or RuntimeError("Gemini call failed")
        
        if getattr(response, "text", None):
            return response.text, None
        else:
            # 检查是否有安全过滤
            prompt_feedback = getattr(response, "prompt_feedback", None)
            if prompt_feedback:
                return None, f"内容被安全过滤: {prompt_feedback}"
            # Gemini 3 可能会消耗较多“思考 token”。如果 max_output_tokens 太小，会出现候选返回但无可见文本。
            try:
                cands = getattr(response, "candidates", None) or []
                fr = getattr(cands[0], "finish_reason", None) if cands else None
                if fr and "MAX_TOKENS" in str(fr):
                    usage = getattr(response, "usage_metadata", None)
                    thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
                    return (
                        None,
                        f"Gemini 输出为空（max_output_tokens 可能过小，thoughts_token_count={thoughts}）。"
                        "请调大 max_tokens 或降低 thinking_level。",
                    )
            except Exception:
                pass
            return None, "Gemini 返回空响应"
            
    except Exception as e:
        logging.error(f"Gemini API 调用失败: {e}")
        return None, f"Gemini API 调用失败: {str(e)}"


def call_gemini_chat(
    system_prompt: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 10240,
    model: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    调用 Gemini API 进行多轮对话。
    
    messages 格式: [{"role": "user"|"assistant", "content": "..."}]
    
    返回 (response_text, error_message)
    """
    if not gemini_client:
        return None, "Gemini API 未配置（缺少 GEMINI_API_KEY）"
    
    try:
        model_name = _normalize_gemini_model_name(model)
        tl = (thinking_level or GEMINI_DEFAULT_THINKING_LEVEL).strip().lower()
        if tl not in {"low", "high"}:
            tl = GEMINI_DEFAULT_THINKING_LEVEL

        try:
            cfg = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=tl),
            )
        except Exception:
            cfg = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
                "thinking_config": {"thinking_level": tl},
            }

        # 重要：该服务端是“无状态”的。每次请求都会重建 history，因此 system_prompt 必须每次都注入，
        # 否则用户第二轮开始就会丢失“市场快照上下文”。
        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": system_prompt}]}
        ]
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        # 兼容：若 SDK 版本对 structured contents 支持不一致，则回退到拼接文本
        def _build_fallback_text() -> str:
            lines: List[str] = [system_prompt, "---"]
            for msg in messages:
                r = "USER" if msg.get("role") == "user" else "ASSISTANT"
                lines.append(f"{r}: {msg.get('content', '')}")
            return "\n".join(lines)

        max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "2")))
        last_exc: Optional[Exception] = None
        response = None
        for attempt in range(max_retries + 1):
            try:
                try:
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=cfg,
                    )
                except Exception:
                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=_build_fallback_text(),
                        config=cfg,
                    )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries or not _should_retry_gemini_exception(exc):
                    raise
                sleep_s = 0.8 * (2**attempt)
                logging.warning(
                    "Gemini Chat 调用失败（%s），%.1fs 后重试（%d/%d）",
                    type(exc).__name__,
                    sleep_s,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_s)
        if last_exc is not None or response is None:
            raise last_exc or RuntimeError("Gemini chat call failed")
        
        if getattr(response, "text", None):
            return response.text, None
        else:
            prompt_feedback = getattr(response, "prompt_feedback", None)
            if prompt_feedback:
                return None, f"内容被安全过滤: {prompt_feedback}"
            try:
                cands = getattr(response, "candidates", None) or []
                fr = getattr(cands[0], "finish_reason", None) if cands else None
                if fr and "MAX_TOKENS" in str(fr):
                    usage = getattr(response, "usage_metadata", None)
                    thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
                    return (
                        None,
                        f"Gemini 输出为空（max_output_tokens 可能过小，thoughts_token_count={thoughts}）。"
                        "请调大 max_tokens 或降低 thinking_level。",
                    )
            except Exception:
                pass
            return None, "Gemini 返回空响应"
            
    except Exception as e:
        logging.error(f"Gemini Chat API 调用失败: {e}")
        return None, f"Gemini API 调用失败: {str(e)}"


NEWS_QUERY = "Federal Reserve OR CPI OR inflation"
NEWS_RSS = (
    f"https://news.google.com/rss/search?q={NEWS_QUERY.replace(' ', '+')}+when:3d"
    "&hl=en-US&gl=US&ceid=US:en"
)

PCR_URLS = [
    "https://cdn.cboe.com/data/us/options/market_statistics/daily_put_call_ratios.csv",
    "https://cdn.cboe.com/data/PUT/pc.csv",
    "https://cdn.cboe.com/resources/options/volume_and_put_call_ratios/totalpc.csv",
]

# CBOE Put/Call CSV URLs (新增)
CBOE_PCR_URLS = {
    "total": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "equity": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "vix": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/vixpc.csv",
}

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TE_CACHE_PATH = os.getenv("TE_CACHE_PATH") or os.path.join(DATA_DIR, "cache_te_calendar.json")
TE_CIRCUIT_PATH = os.getenv("TE_CIRCUIT_PATH") or os.path.join(DATA_DIR, "cache_te_circuit.json")

# -------- TradingEconomics helpers --------
def te_credential_string() -> str:
    key = (TE_CLIENT_KEY or "").strip()
    secret = (TE_CLIENT_SECRET or "").strip()
    if key and secret:
        return f"{key}:{secret}"
    return key


def te_login() -> Tuple[bool, Optional[str]]:
    """兼容仅 key 或 key:secret 的登录方式（部分账号没有 secret 也能用）。"""
    try:
        import tradingeconomics as te  # type: ignore
    except Exception as exc:
        return False, f"tradingeconomics package not available: {exc}"
    key = (TE_CLIENT_KEY or "").strip()
    secret = (TE_CLIENT_SECRET or "").strip()
    if not key:
        return False, "TE_CLIENT_KEY missing"
    try:
        if secret:
            te.login(f"{key}:{secret}")
        else:
            te.login(key)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _circuit_open(now_ts: float) -> bool:
    data = _load_json(TE_CIRCUIT_PATH, {})
    until = float(data.get("blocked_until_ts", 0))
    return now_ts < until


def _set_circuit(hours: int, reason: str) -> None:
    now_ts = time.time()
    _save_json(
        TE_CIRCUIT_PATH,
        {
            "blocked_until_ts": now_ts + hours * 3600,
            "reason": reason,
            "set_at": dt.datetime.utcnow().isoformat() + "Z",
        },
    )


def get_te_calendar_events(
    countries: tuple = ("united states",),
    lookback_days: int = 7,
    lookforward_days: int = 7,
    importance_min: Optional[int] = 1,
    timeout: int = 15,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    使用 TradingEconomics REST 接口拉取经济日历，带缓存和断路器，避免 403/限频导致反复失败。
    - 窗口：回看+前看（避免当天无数据时为空）
    - 1 req/s throttle，409 退避重试
    - 403 时开启断路器（6h），优先返回缓存
    """
    cred = te_credential_string()
    if not cred:
        return [], "TradingEconomics credentials missing (TE_CLIENT_KEY/TE_CLIENT_SECRET)."

    now_ts = time.time()
    circuit = _load_json(TE_CIRCUIT_PATH, {})
    circuit_open = now_ts < float(circuit.get("blocked_until_ts", 0))
    last_try_date = circuit.get("last_try_date")
    today_str = dt.date.today().isoformat()
    if circuit_open and last_try_date == today_str:
        cached = _load_json(TE_CACHE_PATH, {})
        if cached.get("events"):
            return cached["events"], "TradingEconomics calendar skipped (circuit open). Using cached events."
        return [], "TradingEconomics calendar skipped (circuit open). No cache available."

    today = dt.date.today()
    start = (today - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=lookforward_days)).strftime("%Y-%m-%d")
    country_path = ",".join([c.replace(" ", "%20") for c in countries])
    url = f"https://api.tradingeconomics.com/calendar/country/{country_path}/{start}/{end}"

    params = {"c": cred, "f": "json"}
    if importance_min is not None:
        params["importance"] = str(importance_min)

    headers = {"User-Agent": "finance-middleware/1.0"}

    # 1 req/s throttle
    time.sleep(1.05)

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 409:
            time.sleep(1.5)
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 403:
            _set_circuit(hours=6, reason="TE 403 Forbidden (quota/blocked)")
            circuit["last_try_date"] = today_str
            _save_json(TE_CIRCUIT_PATH, circuit)
            cached = _load_json(TE_CACHE_PATH, {})
            if cached.get("events"):
                return cached["events"], "TradingEconomics 403; using cached events."
            return [], "TradingEconomics unauthorized/forbidden (status=403). Check TE quota/plan or unblock; no cache."
        if r.status_code == 401:
            return [], "TradingEconomics 401 Unauthorized. Check TE key/secret."
        r.raise_for_status()
        raw = r.json() or []
    except Exception as e:
        cached = _load_json(TE_CACHE_PATH, {})
        if cached.get("events"):
            return cached["events"], f"TradingEconomics calendar failed ({e}); using cached events."
        return [], f"TradingEconomics calendar request failed: {e}"

    events: List[Dict[str, Any]] = []
    for x in raw:
        events.append(
            {
                "datetime": x.get("Date"),
                "country": x.get("Country"),
                "category": x.get("Category"),
                "event": x.get("Event"),
                "actual": x.get("Actual"),
                "previous": x.get("Previous"),
                "forecast": x.get("Forecast") or x.get("TEForecast"),
                "importance": x.get("Importance"),
                "unit": x.get("Unit"),
                "currency": x.get("Currency"),
                "source": x.get("Source"),
                "source_url": x.get("SourceURL"),
                "ticker": x.get("Ticker") or x.get("Symbol"),
                "te_url": x.get("URL"),
                "last_update": x.get("LastUpdate"),
            }
        )

    _save_json(
        TE_CACHE_PATH,
        {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "window": {"start": start, "end": end, "importance_min": importance_min, "countries": list(countries)},
            "events": events,
        },
    )
    circuit["last_try_date"] = today_str
    _save_json(TE_CIRCUIT_PATH, circuit)

    note = f"TradingEconomics calendar window: {start} to {end}, importance>={importance_min}."
    return events, note


# -------- FRED helpers --------
def fred_latest_observation(series_id: str, timeout: int = 15, max_rows: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    直接调用 FRED observations，按时间倒序取最近有效值。
    返回 {"value": float, "date": "YYYY-MM-DD"} 或 (None, note)
    """
    if not FRED_API_KEY:
        return None, "FRED_API_KEY missing."

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": series_id,
        "sort_order": "desc",
        "limit": max_rows,
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        obs = (r.json() or {}).get("observations", []) or []
        for o in obs:
            v = o.get("value", None)
            if v is None:
                continue
            v = str(v).strip()
            if v in ("", ".", "nan", "NaN"):
                continue
            return {"value": float(v), "date": o.get("date")}, None
        return None, f"No valid observation found for {series_id}."
    except Exception as e:
        return None, f"FRED observations failed for {series_id}: {e}"


# -------- FRED releases fallback for events --------
EVENT_RELEASES = [
    {"name": "CPI", "release_id": 10},
    {"name": "Employment Situation", "release_id": 50},
    {"name": "Retail Sales", "release_id": 31},
    {"name": "GDP", "release_id": 53},
]


def fred_last_release_date(release_id: int, timeout: int = 15) -> Optional[str]:
    if not FRED_API_KEY:
        return None
    url = "https://api.stlouisfed.org/fred/release/dates"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "release_id": release_id,
        "sort_order": "desc",
        "limit": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json() or {}
        dates = data.get("release_dates", []) or []
        if dates:
            return dates[0].get("date")
    except Exception:
        return None
    return None


def build_events_fred(timeout: int = 15) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for item in EVENT_RELEASES:
        d = fred_last_release_date(item["release_id"], timeout=timeout)
        if d:
            events.append({"event": item["name"], "last_release_date": d, "source": "FRED"})
    return events


# -------- Investing.com Economic Calendar --------
def get_investing_calendar_events(
    country: str = "united states",
    lookback_days: int = 1,
    lookforward_days: int = 7,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    从Investing.com经济日历获取事件数据。
    使用POST请求调用其内部API获取数据。
    
    Returns:
        Tuple[List[Dict], Optional[str]]: (事件列表, 备注信息)
    """
    try:
        from datetime import datetime, timedelta
        
        # 计算日期范围
        today = datetime.now()
        date_from = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=lookforward_days)).strftime("%Y-%m-%d")
        
        # Investing.com AJAX API endpoint
        url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.investing.com",
            "Referer": "https://www.investing.com/economic-calendar/",
        }
        
        # 国家ID映射 (Investing.com内部ID)
        country_ids = {
            "united states": "5",
            "usa": "5",
            "us": "5",
        }
        country_id = country_ids.get(country.lower(), "5")
        
        # POST数据
        data = {
            "country[]": country_id,
            "dateFrom": date_from,
            "dateTo": date_to,
            "timeZone": "8",  # UTC+8，可根据需要调整
            "timeFilter": "timeRemain",
            "currentTab": "today",
            "submitFilters": "1",
            "limit_from": "0",
        }
        
        r = requests.post(url, headers=headers, data=data, timeout=timeout)
        
        if r.status_code != 200:
            return [], f"Investing.com API returned status {r.status_code}"
        
        # 解析JSON响应
        try:
            resp = r.json()
            html_data = resp.get("data", "")
        except:
            html_data = r.text
        
        # 解析HTML表格数据
        events = _parse_investing_calendar_html(html_data, country)
        
        if events:
            return events, "Investing.com Economic Calendar"
        else:
            return [], "Investing.com: No events found or parsing failed"
            
    except requests.exceptions.Timeout:
        return [], "Investing.com calendar request timed out"
    except requests.exceptions.RequestException as e:
        return [], f"Investing.com calendar request failed: {str(e)}"
    except Exception as e:
        logging.warning("Investing.com calendar error: %s", e)
        return [], f"Investing.com calendar error: {str(e)}"


def _parse_investing_calendar_html(html: str, country_filter: str = "united states") -> List[Dict[str, Any]]:
    """
    解析Investing.com经济日历的HTML数据。
    
    Args:
        html: HTML字符串（AJAX返回的数据）
        country_filter: 国家过滤器
    
    Returns:
        事件列表
    """
    import re
    events: List[Dict[str, Any]] = []
    
    if not html:
        return events
    
    try:
        # 查找所有事件行
        rows = re.findall(r'<tr[^>]*eventRowId[^>]*>(.*?)</tr>', html, re.DOTALL)
        
        for row in rows:
            event = _extract_investing_event(row)
            if event:
                events.append(event)
        
    except Exception as e:
        logging.warning("Failed to parse Investing.com calendar HTML: %s", e)
    
    return events


def _extract_investing_event(row_html: str) -> Optional[Dict[str, Any]]:
    """从Investing.com的单个HTML行提取事件信息"""
    import re
    
    event: Dict[str, Any] = {"source": "Investing.com", "country": "United States"}
    
    # 提取事件名称 - 查找event链接中的文本
    # 格式: <a href="/economic-calendar/xxx">Event Name</a>
    event_link_match = re.search(r'href="/economic-calendar/[^"]*"[^>]*>([^<]+)</a>', row_html)
    if event_link_match:
        event['event'] = event_link_match.group(1).strip()
    else:
        return None  # 没有事件名称则跳过
    
    # 提取时间
    time_match = re.search(r'class="[^"]*time[^"]*"[^>]*>([^<]+)</td>', row_html)
    if time_match:
        event['time'] = time_match.group(1).strip()
    
    # 提取日期时间属性
    datetime_match = re.search(r'data-event-datetime="([^"]+)"', row_html)
    if datetime_match:
        event['datetime'] = datetime_match.group(1)
    
    # 提取重要性（星星数量 - 计算grayFullBullishIcon的数量）
    bull_count = len(re.findall(r'grayFullBullishIcon', row_html))
    event['importance'] = bull_count if bull_count > 0 else 1
    
    # 提取actual值
    actual_match = re.search(r'class="[^"]*\bact\b[^"]*"[^>]*>([^<]*)</td>', row_html, re.IGNORECASE)
    if actual_match:
        val = actual_match.group(1).strip()
        event['actual'] = val if val and val != '&nbsp;' else None
    
    # 提取forecast值
    forecast_match = re.search(r'class="[^"]*\bfore\b[^"]*"[^>]*>([^<]*)</td>', row_html, re.IGNORECASE)
    if forecast_match:
        val = forecast_match.group(1).strip()
        event['forecast'] = val if val and val != '&nbsp;' else None
    
    # 提取previous值 - 可能在<span>中
    prev_match = re.search(r'class="[^"]*\bprev\b[^"]*"[^>]*>(?:<span[^>]*>)?([^<]*)(?:</span>)?</td>', row_html, re.IGNORECASE)
    if prev_match:
        val = prev_match.group(1).strip()
        event['previous'] = val if val and val != '&nbsp;' else None
    
    # 提取重要性描述
    importance_match = re.search(r'title="((?:Low|Moderate|High)[^"]*Volatility[^"]*)"', row_html)
    if importance_match:
        event['importance_desc'] = importance_match.group(1)
    
    return event if event.get('event') else None


# -------- CBOE PCR helpers --------
def _pick_col(df: pd.DataFrame, keywords: List[str], default_idx: int = 0) -> Optional[str]:
    """根据关键词查找列名，找不到时返回 default_idx 对应的列。"""
    cols = list(df.columns)
    for k in keywords:
        for c in cols:
            if k in str(c).lower():
                return c
    return cols[default_idx] if cols else None


def _read_cboe_csv_table(raw_csv: str, header_token: str = "DATE", max_scan_rows: int = 30) -> pd.DataFrame:
    """
    CBOE 的部分 CSV 前几行是免责声明/元数据，真正表头通常在包含 DATE 的那一行。
    这里自动扫描并抽取表格部分，避免把免责声明当成 header 导致列名/取值错位。
    """
    try:
        df_raw = pd.read_csv(io.StringIO(raw_csv), header=None)
    except Exception:
        # 兜底：让 pandas 自己解析（可能失败，但不要在这里吞异常）
        return pd.read_csv(io.StringIO(raw_csv))

    header_idx: Optional[int] = None
    scan_n = min(len(df_raw), max_scan_rows)
    for idx in range(scan_n):
        row_vals = [str(v).strip().upper() for v in df_raw.iloc[idx].tolist()]
        if header_token.upper() in row_vals:
            header_idx = idx
            break

    if header_idx is None:
        df = pd.read_csv(io.StringIO(raw_csv))
        df.columns = [str(c).strip() for c in df.columns]
        return df.dropna(how="all")

    columns: List[str] = []
    for i, v in enumerate(df_raw.iloc[header_idx].tolist()):
        name = str(v).strip()
        if not name or name.lower() == "nan":
            name = f"col_{i}"
        columns.append(name)

    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = columns
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def _read_csv_url(url: str, timeout: int = 15) -> pd.DataFrame:
    headers = {"User-Agent": "finance-middleware/1.0", "Accept": "text/csv,*/*"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def _download_text(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "finance-middleware/1.0", "Accept": "text/csv,*/*"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _find_header_line_index(lines: List[str], header_token: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if header_token in line:
            return i
    return None


def _parse_pc_csv(text: str) -> Tuple[float, str]:
    lines = text.splitlines()
    # 1) Date-based tables (totalpc/equitypc)
    idx = _find_header_line_index(lines, "Date,")
    if idx is not None:
        csv_text = "\n".join(lines[idx:])
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        date_col = "Date"
        ratio_col = None
        for c in df.columns:
            if "P/C" in c or "Put/Call" in c:
                ratio_col = c
                break
        if ratio_col is None:
            put_col = next((c for c in df.columns if "Put" in c and "Volume" in c), None)
            call_col = next((c for c in df.columns if "Call" in c and "Volume" in c), None)
            if put_col and call_col:
                df["__ratio__"] = pd.to_numeric(df[put_col], errors="coerce") / pd.to_numeric(
                    df[call_col], errors="coerce"
                )
                ratio_col = "__ratio__"
            else:
                raise ValueError("Cannot find ratio column in Date-based CSV.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[date_col, ratio_col]).sort_values(date_col)
        last = df.iloc[-1]
        return float(last[ratio_col]), last[date_col].date().isoformat()

    # 2) indexpcarchive: header "Trade_date"
    idx = _find_header_line_index(lines, "Trade_date")
    if idx is not None:
        csv_text = "\n".join(lines[idx:])
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        date_col = "Trade_date"
        ratio_col = "P/C Ratio" if "P/C Ratio" in df.columns else None
        if ratio_col is None:
            put_col = "Put" if "Put" in df.columns else None
            call_col = "Call" if "Call" in df.columns else None
            if put_col and call_col:
                df["__ratio__"] = pd.to_numeric(df[put_col], errors="coerce") / pd.to_numeric(
                    df[call_col], errors="coerce"
                )
                ratio_col = "__ratio__"
            else:
                raise ValueError("Cannot find ratio column in Trade_date-based CSV.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[date_col, ratio_col]).sort_values(date_col)
        last = df.iloc[-1]
        return float(last[ratio_col]), last[date_col].date().isoformat()

    raise ValueError("Unknown CBOE CSV format (no Date,/Trade_date header found).")


def get_cboe_put_call_ratio(timeout: int = 15) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    返回 total/equity/index put/call ratio 最新值，按顺序尝试三个官方 CSV，兼容说明行/不同表头。
    """
    max_stale_days = 7
    url_candidates = [
        ("cboe-totalpc", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"),
        ("cboe-equitypc", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"),
        ("cboe-indexpcarchive", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv"),
    ]
    last_error = None
    for source, url in url_candidates:
        try:
            text = _download_text(url, timeout=timeout)
            value, date_str = _parse_pc_csv(text)
            try:
                latest_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                if latest_date < (dt.date.today() - dt.timedelta(days=max_stale_days)):
                    last_error = f"{url} stale ({date_str})"
                    continue
            except Exception:
                pass
            return {"value": value, "date": date_str, "source": source, "url": url}, None
        except Exception as e:
            last_error = f"{url} failed: {e}"
            continue
    return None, last_error or "CBOE put/call fetch failed."


def fetch_put_call_from_cboe_equitypc(
    url: str = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    max_age_days: int = 10,
) -> Dict[str, Any]:
    """
    首选：读取 CBOE 官方 equitypc.csv，返回最新值与 5/20 日均值。
    - 自动跳过免责声明/元数据行，避免 ratio/date 列误判
    - 强制 pd.to_datetime，彻底消除 numpy.int64.date() 报错链路
    - 如果数据过旧（> max_age_days），直接抛错触发上层兜底
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    df = _read_cboe_csv_table(r.text, header_token="DATE", max_scan_rows=30)
    df.columns = [str(c).strip() for c in df.columns]

    # 自动识别日期列
    date_col: Optional[str] = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "dates", "trade date", "tradedate"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0] if len(df.columns) else None
    if not date_col:
        raise ValueError("equitypc.csv: date column not found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # 自动识别 ratio 列（注意避开 date 列，避免把 DATE 当 ratio 转成纳秒整数）
    ratio_col: Optional[str] = None
    for c in df.columns:
        if c == date_col:
            continue
        if "ratio" in c.lower():
            ratio_col = c
            break

    # 如果没有 ratio 列，就用 PUT/CALL 自己算
    if ratio_col is None:
        put_col = next((c for c in df.columns if c.lower().strip() == "put"), None)
        call_col = next((c for c in df.columns if c.lower().strip() == "call"), None)
        if put_col and call_col:
            df["pc_ratio_calc"] = pd.to_numeric(df[put_col], errors="coerce") / pd.to_numeric(
                df[call_col], errors="coerce"
            )
            ratio_col = "pc_ratio_calc"
        else:
            raise ValueError(f"equitypc.csv: ratio/put/call column not found. columns={df.columns.tolist()}")

    df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
    df = df.dropna(subset=[ratio_col])

    if df.empty:
        raise ValueError("equitypc.csv parsed empty (maybe HTML/blocked or schema changed)")

    latest = df.iloc[-1]
    latest_dt = latest[date_col]

    tail5 = df.tail(5)
    tail20 = df.tail(20)

    def _mean(series: pd.Series) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if not s.empty else None

    latest_ratio = float(latest[ratio_col]) if pd.notna(latest[ratio_col]) else None
    latest_date = latest_dt.strftime("%Y-%m-%d") if hasattr(latest_dt, "strftime") else None

    if latest_ratio is None or latest_date is None:
        raise ValueError("equitypc.csv: failed to parse valid ratio/date")

    # 数据新鲜度校验：过旧直接判失败，交给上层兜底
    try:
        age_days = (datetime.utcnow().date() - latest_dt.date()).days  # type: ignore[union-attr]
    except Exception:
        age_days = 9999
    if age_days > max_age_days:
        raise ValueError(f"equitypc.csv stale: latest={latest_date}, age_days={age_days}")

    return {
        "put_call_ratio": latest_ratio,
        "put_call_ratio_5d": _mean(tail5[ratio_col]),
        "put_call_ratio_20d": _mean(tail20[ratio_col]),
        "put_call_source": "cboe-equitypc",
        "put_call_date": latest_date,
        "put_call_note": None,
        "put_call_url": url,
    }


def fetch_put_call_from_cboe_daily_options_json(lookback_days: int = 45) -> Dict[str, Any]:
    """
    兜底：抓 Cboe daily 页面背后的 JSON 数据文件（比直接扒 HTML 稳定）。
    文件形如：
      https://cdn.cboe.com/data/us/options/market_statistics/daily/YYYY-MM-DD_daily_options
    这里会回溯 lookback_days，找到最近 20 个"有 EQUITY PUT/CALL RATIO 的交易日"，并计算 5d/20d 均值。
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    base = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{dt}_daily_options"

    samples: List[Tuple[str, float]] = []
    for i in range(lookback_days):
        dt_str = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = base.format(dt=dt_str)
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue

        ratio_val: Optional[float] = None
        ratios = data.get("ratios") if isinstance(data, dict) else None
        if isinstance(ratios, list):
            for item in ratios:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip().upper()
                if "EQUITY PUT/CALL RATIO" in name:
                    try:
                        ratio_val = float(item.get("value"))
                    except Exception:
                        ratio_val = None
                    break

        if ratio_val is None:
            continue

        samples.append((dt_str, ratio_val))
        if len(samples) >= 20:
            break

    if not samples:
        raise ValueError(f"cboe daily_options: EQUITY PUT/CALL RATIO not found within last {lookback_days} days")

    latest_date, latest_ratio = samples[0]
    values = [v for _, v in samples]

    def _avg(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    out = {
        "put_call_ratio": float(latest_ratio),
        "put_call_ratio_5d": _avg(values[:5]) if len(values) >= 5 else None,
        "put_call_ratio_20d": _avg(values[:20]) if len(values) >= 20 else None,
        "put_call_source": "cboe-daily-options-json",
        "put_call_date": latest_date,
        "put_call_note": None if len(values) >= 20 else f"only collected {len(values)} trading days",
        "put_call_url": base.format(dt=latest_date),
    }
    return out


def get_put_call_from_cboe(kind: str = "total", max_age_days: int = 10) -> Dict[str, Any]:
    """
    直接从 CBOE 官方 CSV 读取 put/call，返回最新值与 5/20 日均值，自动校验数据新鲜度。
    作为最后兜底使用。
    """
    url = CBOE_PCR_URLS.get(kind, CBOE_PCR_URLS["total"])
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        df = _read_cboe_csv_table(r.text, header_token="DATE", max_scan_rows=30)
        if df.empty:
            return {
                "put_call_ratio": None,
                "put_call_date": None,
                "put_call_ratio_5d": None,
                "put_call_ratio_20d": None,
                "put_call_source": f"cboe-{kind}",
                "put_call_note": "CBOE CSV empty",
                "put_call_url": url,
            }

        date_col = _pick_col(df, keywords=["date"], default_idx=0)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if df.empty:
            return {
                "put_call_ratio": None,
                "put_call_date": None,
                "put_call_ratio_5d": None,
                "put_call_ratio_20d": None,
                "put_call_source": f"cboe-{kind}",
                "put_call_note": "CBOE CSV date parse failed",
                "put_call_url": url,
            }

        ratio_col = _pick_col(df, keywords=["p/c", "put/call", "ratio"], default_idx=1)
        if ratio_col == date_col:
            # 避免把 DATE 列当成 ratio 列，导致 date 列被数值化后出现 numpy.int64.date() 报错
            ratio_col = next(
                (c for c in df.columns if c != date_col and ("ratio" in str(c).lower() or "p/c" in str(c).lower())),
                None,
            )
            if ratio_col is None and len(df.columns) > 1:
                ratio_col = df.columns[1]
            if ratio_col is None:
                return {
                    "put_call_ratio": None,
                    "put_call_date": None,
                    "put_call_ratio_5d": None,
                    "put_call_ratio_20d": None,
                    "put_call_source": f"cboe-{kind}",
                    "put_call_note": "CBOE CSV ratio column missing",
                    "put_call_url": url,
                }
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[ratio_col])
        if df.empty:
            return {
                "put_call_ratio": None,
                "put_call_date": None,
                "put_call_ratio_5d": None,
                "put_call_ratio_20d": None,
                "put_call_source": f"cboe-{kind}",
                "put_call_note": "CBOE CSV ratio parse failed",
                "put_call_url": url,
            }

        latest = df.iloc[-1]
        latest_ts = pd.to_datetime(latest[date_col], errors="coerce")
        latest_ratio = float(latest[ratio_col])

        tail = df.tail(30)
        ratio_5d = float(tail[ratio_col].tail(5).mean()) if len(tail) >= 5 else None
        ratio_20d = float(tail[ratio_col].tail(20).mean()) if len(tail) >= 20 else None

        latest_date = latest_ts.strftime("%Y-%m-%d") if pd.notna(latest_ts) else None
        age_days = (datetime.now(timezone.utc).date() - latest_ts.date()).days if pd.notna(latest_ts) else 9999
        note = None
        if age_days > max_age_days:
            note = f"CBOE PCR feed stale: latest={latest_date}, age_days={age_days}"

        return {
            "put_call_ratio": latest_ratio,
            "put_call_date": latest_date,
            "put_call_ratio_5d": ratio_5d,
            "put_call_ratio_20d": ratio_20d,
            "put_call_source": f"cboe-{kind}",
            "put_call_note": note,
            "put_call_url": url,
        }

    except Exception as e:
        return {
            "put_call_ratio": None,
            "put_call_date": None,
            "put_call_ratio_5d": None,
            "put_call_ratio_20d": None,
            "put_call_source": f"cboe-{kind}",
            "put_call_note": f"fetch failed: {e}",
            "put_call_url": url,
        }


def _parse_date(date_str: Optional[str]) -> date:
    """Parse YYYY-MM-DD, default今天；若为周末自动回退到上一个工作日。"""
    if not date_str:
        parsed = datetime.utcnow().date()
    else:
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date, expected YYYY-MM-DD") from exc
    while parsed.weekday() >= 5:  # 周末退回最近交易日
        parsed -= timedelta(days=1)
    return parsed


def _safe_asof(series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    if series is None or series.empty:
        return None
    try:
        value = series.asof(target)
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Optional[float], signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _fmt_num(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _to_billions(value: Optional[float]) -> Optional[float]:
    """FRED 货币量单位通常是百万，转为十亿美元以便阅读。"""
    if value is None:
        return None
    return value / 1000.0


def _download_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, threads=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception:
        return pd.DataFrame()


def _first_available_price(
    tickers: List[str], target: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[Optional[float], Optional[str]]:
    """依次尝试多个 ticker，返回第一个成功的收盘价与 ticker 名。"""
    for ticker in tickers:
        prices = _download_prices([ticker], start, end)
        if prices.empty or ticker not in prices:
            continue
        value = _safe_asof(prices[ticker], target)
        if value is not None:
            return value, ticker
    return None, None


def _extract_put_call_ratio(df: pd.DataFrame) -> Optional[float]:
    """更健壮地从 CBOE CSV 中提取 P/C，比对多种列名并兼容缺失。"""
    if df.empty:
        return None
    candidates = [col for col in df.columns if "P/C" in col.upper() or "PUT" in col.upper()]
    for col in candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    # 兜底：取最后一个数值列的最新值
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in reversed(numeric_cols):
        series = df[col].dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return None


# -----------------------------------------------------------------------------
# 市场广度数据获取 - WSJ Market Breadth (动态页面，需要JavaScript渲染)
# -----------------------------------------------------------------------------
def fetch_market_breadth(timeout: int = 15) -> Dict[str, Any]:
    """
    尝试从WSJ网页抓取NYSE/NASDAQ的涨跌家数、新高新低等市场广度数据。
    
    注意: WSJ页面使用JavaScript动态渲染，静态请求可能无法获取完整数据。
    当无法获取时，系统会使用RSP-SPX差异作为广度代理指标。
    
    数据来源: https://www.wsj.com/market-data/stocks/us
    
    Returns:
        Dict包含:
        - nyse_advancing: NYSE上涨家数
        - nyse_declining: NYSE下跌家数
        - nasdaq_advancing: NASDAQ上涨家数
        - nasdaq_declining: NASDAQ下跌家数
        - nyse_new_highs: NYSE新高数量
        - nyse_new_lows: NYSE新低数量
        - nasdaq_new_highs: NASDAQ新高数量
        - nasdaq_new_lows: NASDAQ新低数量
        - advance_decline_ratio: 综合涨跌比
        - new_high_low_ratio: 新高新低比
        - breadth_source: 数据来源
        - breadth_note: 备注信息
    """
    import re
    
    result: Dict[str, Any] = {
        "nyse_advancing": None,
        "nyse_declining": None,
        "nasdaq_advancing": None,
        "nasdaq_declining": None,
        "nyse_new_highs": None,
        "nyse_new_lows": None,
        "nasdaq_new_highs": None,
        "nasdaq_new_lows": None,
        "advance_decline_ratio": None,
        "new_high_low_ratio": None,
        "breadth_source": None,
        "breadth_note": None,
    }
    
    try:
        # WSJ页面是动态渲染的，尝试获取可能的静态数据
        url = "https://www.wsj.com/market-data/stocks/us"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        html = r.text
        
        def parse_num(s: str) -> Optional[int]:
            if not s:
                return None
            try:
                return int(s.replace(",", "").strip())
            except ValueError:
                return None
        
        # 尝试解析动态渲染前可能存在的数据
        # 注意：这些正则可能需要根据WSJ页面结构变化而更新
        
        # 尝试从JSON数据中提取（如果存在）
        json_patterns = [
            r'"issuesAdvancing"\s*:\s*(\d+)',
            r'"issuesDeclining"\s*:\s*(\d+)',
            r'"newHighs"\s*:\s*(\d+)',
            r'"newLows"\s*:\s*(\d+)',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                if 'advancing' in pattern.lower():
                    result["nyse_advancing"] = parse_num(match.group(1))
                elif 'declining' in pattern.lower():
                    result["nyse_declining"] = parse_num(match.group(1))
                elif 'highs' in pattern.lower():
                    result["nyse_new_highs"] = parse_num(match.group(1))
                elif 'lows' in pattern.lower():
                    result["nyse_new_lows"] = parse_num(match.group(1))
        
        # 计算综合指标
        total_adv = (result["nyse_advancing"] or 0) + (result["nasdaq_advancing"] or 0)
        total_dec = (result["nyse_declining"] or 0) + (result["nasdaq_declining"] or 0)
        if total_adv > 0 and total_dec > 0:
            result["advance_decline_ratio"] = round(total_adv / total_dec, 3)
        
        total_nh = (result["nyse_new_highs"] or 0) + (result["nasdaq_new_highs"] or 0)
        total_nl = (result["nyse_new_lows"] or 0) + (result["nasdaq_new_lows"] or 0)
        if total_nh >= 0 and total_nl > 0:
            result["new_high_low_ratio"] = round(total_nh / total_nl, 3)
        elif total_nh > 0 and total_nl == 0:
            result["new_high_low_ratio"] = float("inf")
        
        # 检查是否获取到有效数据
        if result["nyse_advancing"] is None and result["nasdaq_advancing"] is None:
            result["breadth_source"] = "RSP-SPX proxy"
            result["breadth_note"] = "WSJ requires JavaScript; using RSP-SPX spread as breadth proxy"
        else:
            result["breadth_source"] = "WSJ Market Data"
        
    except requests.exceptions.Timeout:
        result["breadth_source"] = "RSP-SPX proxy"
        result["breadth_note"] = "WSJ timeout; using RSP-SPX spread as breadth proxy"
    except requests.exceptions.RequestException as e:
        result["breadth_source"] = "RSP-SPX proxy"
        result["breadth_note"] = f"WSJ unavailable; using RSP-SPX spread as breadth proxy"
    except Exception as e:
        logging.warning("WSJ market breadth error: %s", e)
        result["breadth_source"] = "RSP-SPX proxy"
        result["breadth_note"] = f"WSJ error; using RSP-SPX spread as breadth proxy"
    
    return result


# -----------------------------------------------------------------------------
# SPY期权数据获取 - 使用yfinance
# -----------------------------------------------------------------------------
def fetch_spy_options_metrics(timeout: int = 30) -> Dict[str, Any]:
    """
    使用yfinance获取SPY期权链数据，计算期权市场情绪指标。
    
    数据来源: Yahoo Finance Options Chain
    
    Returns:
        Dict包含:
        - options_pcr_volume: 期权成交量看跌/看涨比率
        - options_pcr_oi: 期权未平仓合约看跌/看涨比率
        - total_call_volume: 看涨期权总成交量
        - total_put_volume: 看跌期权总成交量
        - total_call_oi: 看涨期权总未平仓合约
        - total_put_oi: 看跌期权总未平仓合约
        - atm_iv_call: 平价看涨期权隐含波动率
        - atm_iv_put: 平价看跌期权隐含波动率
        - iv_skew: 隐含波动率偏斜 (put IV - call IV)
        - near_expiry: 最近到期日
        - options_source: 数据来源
        - options_note: 备注信息
    """
    result: Dict[str, Any] = {
        "options_pcr_volume": None,
        "options_pcr_oi": None,
        "total_call_volume": None,
        "total_put_volume": None,
        "total_call_oi": None,
        "total_put_oi": None,
        "atm_iv_call": None,
        "atm_iv_put": None,
        "iv_skew": None,
        "near_expiry": None,
        "options_source": None,
        "options_note": None,
    }
    
    try:
        spy = yf.Ticker("SPY")
        
        # 获取当前价格
        hist = spy.history(period="1d")
        if hist.empty:
            result["options_note"] = "Failed to get SPY current price"
            return result
        current_price = hist["Close"].iloc[-1]
        
        # 获取可用的到期日
        expirations = spy.options
        if not expirations:
            result["options_note"] = "No SPY options expirations available"
            return result
        
        # 选择最近的2-3个到期日来计算综合指标
        near_expirations = list(expirations[:3])
        result["near_expiry"] = near_expirations[0] if near_expirations else None
        
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        atm_calls = []
        atm_puts = []
        
        for exp in near_expirations:
            try:
                opt_chain = spy.option_chain(exp)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                # 累加成交量和未平仓合约
                total_call_vol += calls["volume"].sum() if "volume" in calls else 0
                total_put_vol += puts["volume"].sum() if "volume" in puts else 0
                total_call_oi += calls["openInterest"].sum() if "openInterest" in calls else 0
                total_put_oi += puts["openInterest"].sum() if "openInterest" in puts else 0
                
                # 找到最接近当前价格的行权价（ATM）
                if not calls.empty and "strike" in calls and "impliedVolatility" in calls:
                    calls_atm = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
                    if not calls_atm.empty and not pd.isna(calls_atm["impliedVolatility"].iloc[0]):
                        atm_calls.append(calls_atm["impliedVolatility"].iloc[0])
                
                if not puts.empty and "strike" in puts and "impliedVolatility" in puts:
                    puts_atm = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]
                    if not puts_atm.empty and not pd.isna(puts_atm["impliedVolatility"].iloc[0]):
                        atm_puts.append(puts_atm["impliedVolatility"].iloc[0])
                        
            except Exception as e:
                logging.warning("Failed to fetch SPY options for %s: %s", exp, e)
                continue
        
        # 计算指标
        result["total_call_volume"] = int(total_call_vol) if total_call_vol > 0 else None
        result["total_put_volume"] = int(total_put_vol) if total_put_vol > 0 else None
        result["total_call_oi"] = int(total_call_oi) if total_call_oi > 0 else None
        result["total_put_oi"] = int(total_put_oi) if total_put_oi > 0 else None
        
        # PCR (Put/Call Ratio)
        if total_call_vol > 0:
            result["options_pcr_volume"] = round(total_put_vol / total_call_vol, 3)
        if total_call_oi > 0:
            result["options_pcr_oi"] = round(total_put_oi / total_call_oi, 3)
        
        # ATM隐含波动率
        if atm_calls:
            result["atm_iv_call"] = round(sum(atm_calls) / len(atm_calls), 4)
        if atm_puts:
            result["atm_iv_put"] = round(sum(atm_puts) / len(atm_puts), 4)
        
        # IV偏斜 (Put IV - Call IV)
        if result["atm_iv_put"] is not None and result["atm_iv_call"] is not None:
            result["iv_skew"] = round(result["atm_iv_put"] - result["atm_iv_call"], 4)
        
        result["options_source"] = "Yahoo Finance SPY Options"
        
    except Exception as e:
        logging.warning("SPY options metrics error: %s", e)
        result["options_note"] = f"SPY options error: {str(e)}"
    
    return result


def _fetch_fear_greed() -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """优先使用 fear-greed-index 包，失败则调用 CNN API。"""
    try:
        from fear_greed_index.CNNFearAndGreedIndex import CNNFearAndGreedIndex

        fgi = CNNFearAndGreedIndex()
        data = fgi.get()
        if data:
            return int(data.get("score")), data.get("rating"), "fear-greed-index"
    except Exception as exc:
        logging.warning("fear-greed-index 包不可用，尝试直接请求 CNN: %s", exc)
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=10,
            headers=DEFAULT_HEADERS,
        )
        resp.raise_for_status()
        payload = resp.json()
        return int(payload["fear_and_greed"]["score"]), payload["fear_and_greed"].get("rating"), "cnn-api"
    except Exception as exc:
        logging.warning("CNN Fear & Greed 接口不可用: %s", exc)
    # 尝试历史 CSV 兜底
    try:
        hist_df = pd.read_csv(
            "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/master/fear-greed-historical.csv"
        )
        if not hist_df.empty:
            last_row = hist_df.iloc[-1]
            return int(last_row.get("fear_greed_index")), str(last_row.get("label")), "csv-fallback"
    except Exception as exc:
        logging.warning("FGI 历史 CSV 兜底失败: %s", exc)
    return None, None, None


def fetch_fundamentals(date_str: str) -> Dict[str, Any]:
    """获取利率、DXY、经济日历与新闻。"""
    target = pd.to_datetime(date_str)
    start = target - timedelta(days=10)

    rates = {
        "dgs2": None,
        "dgs10": None,
        "fedfunds": None,
        "term_spread": None,
        "ffr_minus_2y": None,
        "fedfunds_date": None,
        "ffr_minus_2y_note": None,
        # 新增：实际利率与通胀预期
        "real10y": None,
        "breakeven10y": None,
        # 新增：信用利差OAS
        "ig_oas": None,
        "hy_oas": None,
    }
    if fred_client:
        try:
            dgs2 = fred_client.get_series("DGS2", start, target)
            dgs10 = fred_client.get_series("DGS10", start, target)
            rates["dgs2"] = _safe_asof(dgs2, target)
            rates["dgs10"] = _safe_asof(dgs10, target)
            if rates["dgs10"] is not None and rates["dgs2"] is not None:
                rates["term_spread"] = rates["dgs10"] - rates["dgs2"]
        except Exception as exc:
            logging.warning("Failed to fetch FRED rates: %s", exc)
        
        # 获取10年期实际利率 (TIPS收益率) - FRED: DFII10
        try:
            real10y_series = fred_client.get_series("DFII10", start, target)
            rates["real10y"] = _safe_asof(real10y_series, target)
        except Exception as exc:
            logging.warning("Failed to fetch FRED DFII10 (real10y): %s", exc)
        
        # 获取10年期盈亏平衡通胀率 - FRED: T10YIE
        try:
            breakeven_series = fred_client.get_series("T10YIE", start, target)
            rates["breakeven10y"] = _safe_asof(breakeven_series, target)
        except Exception as exc:
            logging.warning("Failed to fetch FRED T10YIE (breakeven10y): %s", exc)
        
        # 获取投资级债券OAS利差 - FRED: BAMLC0A0CM (ICE BofA US Corporate Index OAS)
        try:
            ig_oas_series = fred_client.get_series("BAMLC0A0CM", start, target)
            rates["ig_oas"] = _safe_asof(ig_oas_series, target)
        except Exception as exc:
            logging.warning("Failed to fetch FRED BAMLC0A0CM (ig_oas): %s", exc)
        
        # 获取高收益债券OAS利差 - FRED: BAMLH0A0HYM2 (ICE BofA US High Yield Index OAS)
        try:
            hy_oas_series = fred_client.get_series("BAMLH0A0HYM2", start, target)
            rates["hy_oas"] = _safe_asof(hy_oas_series, target)
        except Exception as exc:
            logging.warning("Failed to fetch FRED BAMLH0A0HYM2 (hy_oas): %s", exc)

    # Fed Funds：优先 EFFR，再退 FEDFUNDS，取最近一次有效值
    effr, _ = fred_latest_observation("EFFR")
    ff_obs, ff_note = (effr, None) if effr else fred_latest_observation("FEDFUNDS")
    if ff_obs:
        rates["fedfunds"] = ff_obs["value"]
        rates["fedfunds_date"] = ff_obs["date"]
    else:
        rates["fedfunds"] = None
        rates["fedfunds_date"] = None
    # ffr_minus_2y：用最新 FedFunds/EFFR 减去最新 DGS2
    dgs2_latest, _ = fred_latest_observation("DGS2")
    if ff_obs and dgs2_latest:
        rates["ffr_minus_2y"] = ff_obs["value"] - dgs2_latest["value"]
        rates["ffr_minus_2y_note"] = f"FF({ff_obs['date']}) - DGS2({dgs2_latest['date']})"
    # asof 日期（使用最新 observation）
    dgs2_obs, _ = fred_latest_observation("DGS2")
    dgs10_obs, _ = fred_latest_observation("DGS10")
    rates["dgs2_date"] = dgs2_obs["date"] if dgs2_obs else None
    rates["dgs10_date"] = dgs10_obs["date"] if dgs10_obs else None

    dxy_value, dxy_ticker = _first_available_price(
        ["DX-Y.NYB", "DXY", "USDOLLAR"], target, start, target + timedelta(days=1)
    )

    # 优先使用 Investing.com 经济日历
    events, events_note = get_investing_calendar_events(
        country="united states", lookback_days=1, lookforward_days=7
    )
    
    # 如果 Investing.com 不可用，尝试 TradingEconomics 作为备选
    if not events:
        te_events, te_note = get_te_calendar_events(
            countries=("united states",), lookback_days=7, lookforward_days=7, importance_min=2
        )
        if te_events:
            events = te_events
            events_note = te_note
        else:
            # 最终回退到 FRED release dates
            fred_events = build_events_fred()
            if fred_events:
                events = fred_events
                if events_note:
                    events_note += " | Fallback to FRED releases."
                else:
                    events_note = "Fallback to FRED releases."

    news: List[Dict[str, Any]] = []
    try:
        feed = feedparser.parse(NEWS_RSS)
        news = [
            {
                "title": entry.title,
                "link": entry.link,
                "published": getattr(entry, "published", None),
                "source": getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else "Google News",
            }
            for entry in feed.entries[:5]
        ]
    except Exception as exc:
        logging.warning("Failed to parse news feed: %s", exc)

    return {
        **rates,
        "dxy": dxy_value,
        "dxy_ticker": dxy_ticker,
        "events": events,
        "events_note": events_note,
        "news": news,
        "news_note": "Google News RSS 仅覆盖最近几天，如为历史日期请注意时效性。",
    }


def fetch_liquidity(date_str: str) -> Dict[str, Any]:
    """净流动性（十亿美元）及信用利差。"""
    target = pd.to_datetime(date_str)
    start = target - timedelta(days=40)

    walcl = rrp = tga = None
    net_liquidity = net_change_4w = None
    if fred_client:
        try:
            walcl_series = fred_client.get_series("WALCL", start, target)
            rrp_series = fred_client.get_series("RRPONTSYD", start, target)
            tga_series = fred_client.get_series("WTREGEN", start, target)
            walcl = _to_billions(_safe_asof(walcl_series, target))
            rrp = _to_billions(_safe_asof(rrp_series, target))
            tga = _to_billions(_safe_asof(tga_series, target))
            if walcl is not None and rrp is not None and tga is not None:
                net_liquidity = walcl - rrp - tga
                prior_date = target - timedelta(days=28)
                prior_val = _to_billions(_safe_asof(walcl_series, prior_date))
                prior_rrp = _to_billions(_safe_asof(rrp_series, prior_date))
                prior_tga = _to_billions(_safe_asof(tga_series, prior_date))
                if prior_val is not None and prior_rrp is not None and prior_tga is not None:
                    net_change_4w = net_liquidity - (prior_val - prior_rrp - prior_tga)
        except Exception as exc:
            logging.warning("Failed to fetch liquidity series: %s", exc)

    prices = _download_prices(["HYG", "IEI"], start, target + timedelta(days=1))
    credit_ratio = credit_change_20d = None
    if not prices.empty and {"HYG", "IEI"} <= set(prices.columns):
        ratio_series = prices["HYG"] / prices["IEI"]
        credit_ratio = _safe_asof(ratio_series, target)
        past_date = target - timedelta(days=20)
        past_ratio = _safe_asof(ratio_series, past_date)
        if credit_ratio is not None and past_ratio is not None:
            credit_change_20d = credit_ratio - past_ratio

    return {
        "walcl": walcl,
        "rrp": rrp,
        "tga": tga,
        "net_liquidity": net_liquidity,
        "net_change_4w": net_change_4w,
        "credit_ratio": credit_ratio,
        "credit_change_20d": credit_change_20d,
    }


def fetch_sentiment(date_str: str) -> Dict[str, Any]:
    """情绪相关指标：Fear & Greed、VIX 期限结构、PCR、风险偏好价差、市场广度。"""
    target = pd.to_datetime(date_str)

    fgi_score, fgi_rating, fgi_source = _fetch_fear_greed()

    vix = vix3m = term_structure = vix_source = None
    vix_data = _download_prices(["^VIX", "^VIX3M"], target - timedelta(days=30), target + timedelta(days=1))
    if not vix_data.empty:
        vix = _safe_asof(vix_data.get("^VIX"), target)
        vix3m = _safe_asof(vix_data.get("^VIX3M"), target)
        vix_source = "^VIX3M"
    if vix is not None and vix3m is None:
        alt_data = _download_prices(["VIXY"], target - timedelta(days=30), target + timedelta(days=1))
        if not alt_data.empty and "VIXY" in alt_data:
            vix3m = _safe_asof(alt_data["VIXY"], target)
            vix_source = "VIXY"
    if vix is not None and vix3m is not None:
        if vix3m > vix:
            term_structure = "contango"
        elif vix3m < vix:
            term_structure = "backwardation"
        else:
            term_structure = "flat"

    # Put/Call：优先 equitypc.csv，失败/过旧则用 daily_options JSON 兜底；再不行就退回 CBOE(total/equity) 逻辑
    try:
        pcr = fetch_put_call_from_cboe_equitypc()
    except Exception as e_equitypc:
        try:
            pcr = fetch_put_call_from_cboe_daily_options_json()
        except Exception as e_daily:
            pcr = get_put_call_from_cboe("total")
            if pcr.get("put_call_ratio") is None:
                pcr = get_put_call_from_cboe("equity")
            base_note = pcr.get("put_call_note")
            extra = f"equitypc failed: {e_equitypc}; daily_options failed: {e_daily}"
            pcr["put_call_note"] = f"{base_note}; {extra}" if base_note else extra

    spreads = {}
    spread_pairs = [("SPY", "XLU", "spy_xlu"), ("HYG", "IEF", "hyg_ief"), ("BTC-USD", "GC=F", "btc_gold")]
    tickers = list({a for pair in spread_pairs for a in pair[:2]})
    prices = _download_prices(tickers, target - timedelta(days=15), target + timedelta(days=1))
    returns = prices.pct_change() if not prices.empty else pd.DataFrame()
    for lhs, rhs, name in spread_pairs:
        if not returns.empty and lhs in returns and rhs in returns:
            spreads[name] = _safe_asof(returns[lhs] - returns[rhs], target)

    # 获取市场广度数据（涨跌家数、新高新低）
    breadth = fetch_market_breadth()

    return {
        "fgi_score": fgi_score,
        "fgi_rating": fgi_rating,
        "fgi_source": fgi_source,
        "vix": vix,
        "vix3m": vix3m,
        "vix_term_source": vix_source,
        "term_structure": term_structure,
        "put_call_ratio": pcr.get("put_call_ratio"),
        "put_call_ratio_5d": pcr.get("put_call_ratio_5d"),
        "put_call_ratio_20d": pcr.get("put_call_ratio_20d"),
        "put_call_source": pcr.get("put_call_source"),
        "put_call_date": pcr.get("put_call_date"),
        "put_call_note": pcr.get("put_call_note"),
        "put_call_url": pcr.get("put_call_url"),
        # 市场广度数据
        "nyse_advancing": breadth.get("nyse_advancing"),
        "nyse_declining": breadth.get("nyse_declining"),
        "nasdaq_advancing": breadth.get("nasdaq_advancing"),
        "nasdaq_declining": breadth.get("nasdaq_declining"),
        "nyse_new_highs": breadth.get("nyse_new_highs"),
        "nyse_new_lows": breadth.get("nyse_new_lows"),
        "nasdaq_new_highs": breadth.get("nasdaq_new_highs"),
        "nasdaq_new_lows": breadth.get("nasdaq_new_lows"),
        "advance_decline_ratio": breadth.get("advance_decline_ratio"),
        "new_high_low_ratio": breadth.get("new_high_low_ratio"),
        "breadth_source": breadth.get("breadth_source"),
        "breadth_note": breadth.get("breadth_note"),
        **spreads,
    }


def _calc_indicators(series: pd.Series, target: pd.Timestamp) -> Dict[str, Any]:
    """计算单资产的价格结构指标。"""
    s = series.dropna()
    if s.empty:
        return {}
    ma20 = s.rolling(20).mean().asof(target)
    ma50 = s.rolling(50).mean().asof(target)
    ma200 = s.rolling(200).mean().asof(target)
    atr14 = s.diff().abs().rolling(14).mean().asof(target)
    close = s.asof(target)
    if pd.isna(close):
        return {}
    trend_label = None
    if ma50 and ma200:
        if close > ma50 > ma200:
            trend_label = "uptrend"
        elif close < ma50 < ma200:
            trend_label = "downtrend"
        else:
            trend_label = "range"
    return {
        "close": float(close),
        "distance_ma20_pct": float((close - ma20) / ma20) if ma20 else None,
        "distance_ma50_pct": float((close - ma50) / ma50) if ma50 else None,
        "distance_ma200_pct": float((close - ma200) / ma200) if ma200 else None,
        "atr_pct": float(atr14 / close) if atr14 else None,
        "boll_width": float(s.rolling(20).std().asof(target) * 2 / ma20) if ma20 else None,
        "trend_label": trend_label,
    }


def fetch_technicals(date_str: str) -> Dict[str, Any]:
    """技术面指标：主流指数/商品/加密的均线距离、ATR、布林带宽度、期权指标等。"""
    target = pd.to_datetime(date_str)
    assets = {
        "SPX": "^GSPC",
        "NDX": "^NDX",
        "RSP": "RSP",
        "IWM": "IWM",
        "XLK": "XLK",
        "XLP": "XLP",
        "XLU": "XLU",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "CRUDE": "CL=F",
        "BTC": "BTC-USD",
    }
    hist = _download_prices(list(assets.values()), target - timedelta(days=750), target + timedelta(days=1))
    technicals: Dict[str, Any] = {}
    for name, ticker in assets.items():
        if hist.empty or ticker not in hist:
            continue
        technicals[name] = _calc_indicators(hist[ticker], target)

    breadth_diff = style_ratio = None
    if not hist.empty and {"RSP", "^GSPC"} <= set(hist.columns):
        returns = hist.pct_change()
        breadth_diff = _safe_asof(returns["RSP"] - returns["^GSPC"], target)
    if not hist.empty and {"XLK", "XLP"} <= set(hist.columns):
        style_ratio = _safe_asof(hist["XLK"] / hist["XLP"], target)

    # 获取SPY期权数据作为技术面补充
    spy_options = fetch_spy_options_metrics()

    return {
        "assets": technicals,
        "breadth_diff": breadth_diff,
        "style_ratio": style_ratio,
        # SPY期权指标
        "options_pcr_volume": spy_options.get("options_pcr_volume"),
        "options_pcr_oi": spy_options.get("options_pcr_oi"),
        "total_call_volume": spy_options.get("total_call_volume"),
        "total_put_volume": spy_options.get("total_put_volume"),
        "total_call_oi": spy_options.get("total_call_oi"),
        "total_put_oi": spy_options.get("total_put_oi"),
        "atm_iv_call": spy_options.get("atm_iv_call"),
        "atm_iv_put": spy_options.get("atm_iv_put"),
        "iv_skew": spy_options.get("iv_skew"),
        "near_expiry": spy_options.get("near_expiry"),
        "options_source": spy_options.get("options_source"),
        "options_note": spy_options.get("options_note"),
    }


def assign_labels(modules: Dict[str, Any]) -> Dict[str, str]:
    """根据阈值打标签，供 LLM 生成综合观点。"""
    labels: Dict[str, str] = {}
    fundamentals = modules.get("fundamentals", {})
    liquidity = modules.get("liquidity", {})
    sentiment = modules.get("sentiment", {})
    technicals = modules.get("technicals", {}).get("assets", {})

    term_spread = fundamentals.get("term_spread")
    if term_spread is None:
        labels["macro_regime"] = "Unknown"
    elif term_spread < -0.5:
        labels["macro_regime"] = "Hawkish"
    elif term_spread > 0.0:
        labels["macro_regime"] = "Dovish"
    else:
        labels["macro_regime"] = "Neutral"

    net_change = liquidity.get("net_change_4w")
    if net_change is None:
        labels["liquidity_regime"] = "Unknown"
    elif net_change > 500:
        labels["liquidity_regime"] = "Easing"
    elif net_change < -500:
        labels["liquidity_regime"] = "Tightening"
    else:
        labels["liquidity_regime"] = "Neutral"

    fgi_score = sentiment.get("fgi_score")
    vix = sentiment.get("vix") or 0
    ad_ratio = sentiment.get("advance_decline_ratio")  # 涨跌家数比
    nh_nl_ratio = sentiment.get("new_high_low_ratio")  # 新高新低比
    
    # 计算广度得分作为辅助判断
    breadth_bullish = 0
    if ad_ratio is not None:
        if ad_ratio > 1.5:
            breadth_bullish += 1
        elif ad_ratio < 0.67:
            breadth_bullish -= 1
    if nh_nl_ratio is not None:
        if nh_nl_ratio > 2:
            breadth_bullish += 1
        elif nh_nl_ratio < 0.5:
            breadth_bullish -= 1
    
    if fgi_score is None:
        labels["sentiment_regime"] = "Unknown"
    elif fgi_score >= 60 and vix < 20:
        labels["sentiment_regime"] = "Greed"
    elif fgi_score <= 40 or vix > 25:
        labels["sentiment_regime"] = "Fear"
    elif breadth_bullish >= 2:
        labels["sentiment_regime"] = "Greed"  # 广度强劲支持贪婪
    elif breadth_bullish <= -2:
        labels["sentiment_regime"] = "Fear"  # 广度疲弱支持恐惧
    else:
        labels["sentiment_regime"] = "Neutral"

    spx = technicals.get("SPX", {})
    trend = spx.get("trend_label")
    boll = spx.get("boll_width")
    
    # 获取技术面模块根级数据（期权数据）
    tech_module = modules.get("technicals", {})
    iv_skew = tech_module.get("iv_skew")  # 期权隐含波动率偏斜
    options_pcr_oi = tech_module.get("options_pcr_oi")  # 期权未平仓PCR
    
    if trend is None:
        labels["technical_regime"] = "Unknown"
    elif trend in {"uptrend", "downtrend"} and boll and boll > 0.04:
        labels["technical_regime"] = "Trending"
    else:
        labels["technical_regime"] = "Range"
    
    # 添加期权情绪作为辅助标签（可选）
    if options_pcr_oi is not None:
        if options_pcr_oi > 1.2:
            labels["options_sentiment"] = "Bearish"
        elif options_pcr_oi < 0.8:
            labels["options_sentiment"] = "Bullish"
        else:
            labels["options_sentiment"] = "Neutral"
    
    if iv_skew is not None:
        if iv_skew > 0.05:
            labels["iv_skew_signal"] = "Put_Heavy"  # 看跌期权IV偏高，可能是对冲需求
        elif iv_skew < -0.02:
            labels["iv_skew_signal"] = "Call_Heavy"  # 看涨期权IV偏高
        else:
            labels["iv_skew_signal"] = "Balanced"

    return labels


def compute_signals(modules: Dict[str, Any], labels: Dict[str, str]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, List[str]]]:
    """
    Compute lightweight signal summaries for each module alongside heuristic
    labels and data quality annotations.  These summaries decouple the LLM
    analysis from raw data structures and make it easier to reason about
    missing fields.

    Parameters
    ----------
    modules : Dict[str, Any]
        The four top-level modules returned by fetch_fundamentals,
        fetch_liquidity, fetch_sentiment and fetch_technicals.
    labels : Dict[str, str]
        Heuristic regime labels (macro/liquidity/sentiment/technical) as
        produced by assign_labels().

    Returns
    -------
    signals : dict
        Mapping from module name to a list of signal dictionaries.  Each
        signal has a `name`, a `value` and a `quality` field ("good" or
        "missing").
    heuristics : dict
        Mapping from module name to the heuristic regime label (e.g. "Dovish").
    data_quality : dict
        Mapping from module name to a list of keys that were missing in the
        original module data.  This allows the LLM to call out gaps.
    """
    signals: Dict[str, List[Dict[str, Any]]] = {}
    heuristics: Dict[str, str] = {}
    data_quality: Dict[str, List[str]] = {}

    # Fundamentals signals
    fund = modules.get("fundamentals", {}) or {}
    fund_keys = [
        "dgs2",
        "dgs10",
        "term_spread",
        "ffr_minus_2y",
        "real10y",
        "breakeven10y",
        "ig_oas",
        "hy_oas",
        "dxy",
    ]
    fund_signals: List[Dict[str, Any]] = []
    fund_missing: List[str] = []
    for k in fund_keys:
        val = fund.get(k)
        quality = "good" if val is not None else "missing"
        fund_signals.append({"name": k, "value": val, "quality": quality})
        if val is None:
            fund_missing.append(k)
    signals["fundamentals"] = fund_signals
    data_quality["fundamentals"] = fund_missing
    heuristics["fundamentals"] = labels.get("macro_regime", "Unknown")

    # Liquidity signals
    liq = modules.get("liquidity", {}) or {}
    liq_keys = [
        "walcl",
        "rrp",
        "tga",
        "net_liquidity",
        "net_change_4w",
        "credit_ratio",
        "credit_change_20d",
    ]
    liq_signals: List[Dict[str, Any]] = []
    liq_missing: List[str] = []
    for k in liq_keys:
        val = liq.get(k)
        quality = "good" if val is not None else "missing"
        liq_signals.append({"name": k, "value": val, "quality": quality})
        if val is None:
            liq_missing.append(k)
    signals["liquidity"] = liq_signals
    data_quality["liquidity"] = liq_missing
    heuristics["liquidity"] = labels.get("liquidity_regime", "Unknown")

    # Sentiment signals
    sent = modules.get("sentiment", {}) or {}
    sent_keys = [
        "fgi_score",
        "vix",
        "vix3m",
        "term_structure",
        "put_call_ratio",
        "spy_xlu",
        "hyg_ief",
        "btc_gold",
        # 市场广度数据
        "advance_decline_ratio",
        "new_high_low_ratio",
    ]
    sent_signals: List[Dict[str, Any]] = []
    sent_missing: List[str] = []
    for k in sent_keys:
        val = sent.get(k)
        quality = "good" if val is not None else "missing"
        sent_signals.append({"name": k, "value": val, "quality": quality})
        if val is None:
            sent_missing.append(k)
    # 添加详细的涨跌家数数据作为补充信号
    breadth_detail = {
        "nyse_adv": sent.get("nyse_advancing"),
        "nyse_dec": sent.get("nyse_declining"),
        "nasdaq_adv": sent.get("nasdaq_advancing"),
        "nasdaq_dec": sent.get("nasdaq_declining"),
        "nyse_new_highs": sent.get("nyse_new_highs"),
        "nyse_new_lows": sent.get("nyse_new_lows"),
        "nasdaq_new_highs": sent.get("nasdaq_new_highs"),
        "nasdaq_new_lows": sent.get("nasdaq_new_lows"),
    }
    if any(v is not None for v in breadth_detail.values()):
        sent_signals.append({"name": "market_breadth_detail", "value": breadth_detail, "quality": "good"})
    signals["sentiment"] = sent_signals
    data_quality["sentiment"] = sent_missing
    heuristics["sentiment"] = labels.get("sentiment_regime", "Unknown")

    # Technical signals
    tech = modules.get("technicals", {}) or {}
    assets = tech.get("assets", {}) or {}
    tech_signals: List[Dict[str, Any]] = []
    tech_missing: List[str] = []
    # We summarise each asset's trend_label and key distance metrics; missing if any critical piece absent
    for asset_name, metrics in assets.items():
        if not metrics:
            tech_signals.append({"name": asset_name, "value": None, "quality": "missing"})
            tech_missing.append(asset_name)
            continue
        # summarise as a dictionary including close, MA distances and trend
        tech_signals.append(
            {
                "name": asset_name,
                "value": {
                    "close": metrics.get("close"),
                    "distance_ma20_pct": metrics.get("distance_ma20_pct"),
                    "distance_ma50_pct": metrics.get("distance_ma50_pct"),
                    "distance_ma200_pct": metrics.get("distance_ma200_pct"),
                    "atr_pct": metrics.get("atr_pct"),
                    "boll_width": metrics.get("boll_width"),
                    "trend_label": metrics.get("trend_label"),
                },
                "quality": "good",
            }
        )
    # If breadth or style ratio missing, note it in missing list
    if tech.get("breadth_diff") is None:
        tech_missing.append("breadth_diff")
    if tech.get("style_ratio") is None:
        tech_missing.append("style_ratio")
    
    # 添加SPY期权数据作为技术面补充
    options_keys = [
        "options_pcr_volume",
        "options_pcr_oi",
        "atm_iv_call",
        "atm_iv_put",
        "iv_skew",
    ]
    for k in options_keys:
        val = tech.get(k)
        quality = "good" if val is not None else "missing"
        tech_signals.append({"name": k, "value": val, "quality": quality})
        if val is None:
            tech_missing.append(k)
    
    # 添加详细的期权数据
    options_detail = {
        "total_call_volume": tech.get("total_call_volume"),
        "total_put_volume": tech.get("total_put_volume"),
        "total_call_oi": tech.get("total_call_oi"),
        "total_put_oi": tech.get("total_put_oi"),
        "near_expiry": tech.get("near_expiry"),
    }
    if any(v is not None for v in options_detail.values()):
        tech_signals.append({"name": "spy_options_detail", "value": options_detail, "quality": "good"})
    
    signals["technicals"] = tech_signals
    data_quality["technicals"] = tech_missing
    heuristics["technicals"] = labels.get("technical_regime", "Unknown")

    return signals, heuristics, data_quality


def require_auth(
    session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
) -> None:
    """Dependency to enforce authentication on protected endpoints."""
    if not (APP_PASSWORD or "").strip():
        return
    token: Optional[str] = session
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
    if not token and x_session_token:
        token = x_session_token.strip()
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="Authentication required")
    return


@app.post("/auth/login")
async def login(
    response: Response,
    password: Optional[str] = Body(None, embed=True),
    password_query: Optional[str] = Query(None, alias="password"),
) -> Dict[str, str]:
    """Simple password-based login.  Expects a JSON body with `password`.

    On success a secure random session token is issued and set as a HTTPOnly
    cookie.  If APP_PASSWORD is not set on the server, login will always
    succeed without checking the password (not recommended for production).
    """
    pwd = (password if password is not None else password_query or "").strip()
    expected = (APP_PASSWORD or "").strip()
    if expected and pwd != expected:
        raise HTTPException(status_code=401, detail="Invalid password")
    token = secrets.token_hex(16)
    _sessions[token] = {"created": time.time()}
    response.set_cookie(key="session", value=token, httponly=True, secure=False, samesite="lax")
    # 同时返回 token，方便非浏览器客户端用 Authorization 头调用受保护接口
    return {"message": "login successful", "token": token}


@app.post("/auth/logout")
async def logout(
    response: Response,
    session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
) -> Dict[str, str]:
    """Invalidate the current session cookie and remove it from the session store."""
    token: Optional[str] = session
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
    if not token and x_session_token:
        token = x_session_token.strip()
    if token and token in _sessions:
        _sessions.pop(token, None)
    response.delete_cookie("session")
    return {"message": "logged out"}


@app.post("/v3/snapshot/run")
async def run_snapshot(
    date: Optional[str] = Query(None, description="YYYY-MM-DD, default today"),
    auth: Any = Depends(require_auth),
) -> Dict[str, Any]:
    """Generate a market snapshot for the given date and persist it.

    This endpoint triggers data collection for fundamentals, liquidity,
    sentiment and technicals for the specified date.  It also computes
    heuristic labels and lightweight signal summaries.  The resulting
    snapshot is stored on disk (DATA_DIR) and can be retrieved via `/v3/snapshot/{id}`.
    """
    target_date = _parse_date(date)
    date_str = target_date.isoformat()
    try:
        fundamentals = fetch_fundamentals(date_str)
        liquidity = fetch_liquidity(date_str)
        sentiment = fetch_sentiment(date_str)
        technicals = fetch_technicals(date_str)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to build snapshot")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    modules = {
        "fundamentals": fundamentals,
        "liquidity": liquidity,
        "sentiment": sentiment,
        "technicals": technicals,
    }
    labels = assign_labels(modules)
    signals, heuristics_by_module, data_quality = compute_signals(modules, labels)
    snapshot_id = secrets.token_hex(8)
    snapshot = {
        "id": snapshot_id,
        "date": date_str,
        "created_ts": time.time(),
        "modules": modules,
        "labels": labels,
        "signals": signals,
        "heuristics": heuristics_by_module,
        "data_quality": data_quality,
    }
    _save_snapshot_to_disk(snapshot)
    _snapshot_cache[snapshot_id] = snapshot  # memory cache
    return snapshot


@app.get("/v3/snapshot/{snapshot_id}")
async def get_snapshot(snapshot_id: str, auth: Any = Depends(require_auth)) -> Dict[str, Any]:
    """Return a previously generated snapshot by id."""
    snapshot = _get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


class ModuleAnalysisRequest(BaseModel):
    snapshot_id: str
    module: str
    call_llm: Optional[bool] = True  # 是否调用 LLM，默认调用
    provider: Optional[str] = None
    model: Optional[str] = None


@app.post("/v3/analysis/module")
async def analyse_module(req: ModuleAnalysisRequest, auth: Any = Depends(require_auth)) -> Dict[str, Any]:
    """对单个模块进行 LLM 分析。

    该端点会：
    1. 获取指定模块的信号、标签和数据质量信息
    2. 构建分析 prompt
    3. 调用 Gemini API 生成分析结果（如果 call_llm=True）
    
    参数：
    - snapshot_id: 快照 ID（由 /v3/snapshot/run 生成）
    - module: 模块名称（fundamentals/liquidity/sentiment/technicals）
    - call_llm: 是否调用 LLM（默认 True）
    """
    snapshot = _get_snapshot(req.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    module_name = req.module.lower()
    if module_name not in {"fundamentals", "liquidity", "sentiment", "technicals"}:
        raise HTTPException(status_code=400, detail="Invalid module name")
    signals = snapshot["signals"].get(module_name, [])
    heuristic_label = snapshot["heuristics"].get(module_name)
    missing = snapshot["data_quality"].get(module_name, [])

    as_of_date = str(snapshot.get("date") or "")
    system_prompt = GLOBAL_SYSTEM_PROMPT
    user_prompt = build_module_user_prompt(
        module_name=module_name,
        as_of_date=as_of_date,
        signals=signals,
        heuristic_label=heuristic_label,
        missing=missing,
        supplemental=None,
    )
    
    response: Dict[str, Any] = {
        "module": module_name,
        "signals": signals,
        "heuristic_label": heuristic_label,
        "data_quality": missing,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    
    # 调用 Gemini API
    if req.call_llm:
        analysis_text, error = call_gemini(system_prompt, user_prompt, model=req.model)
        if analysis_text:
            response["analysis"] = analysis_text
            response["llm_provider"] = "gemini"
            response["llm_model"] = _normalize_gemini_model_name(req.model)
            # 保存模块短评，供按钮2综合分析/Chat 注入使用
            try:
                snapshot.setdefault("llm_module_reports", {})[module_name] = {
                    "text": analysis_text,
                    "llm_model": response["llm_model"],
                    "created_ts": time.time(),
                }
            except Exception:
                pass
            _save_snapshot_to_disk(snapshot)
            _log_llm_event(
                kind="module",
                snapshot_id=req.snapshot_id,
                module_name=module_name,
                record={
                    "model": response["llm_model"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "analysis": analysis_text,
                },
            )
        else:
            response["analysis"] = None
            response["llm_error"] = error
            _log_llm_event(
                kind="module",
                snapshot_id=req.snapshot_id,
                module_name=module_name,
                record={
                    "model": _normalize_gemini_model_name(req.model),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": error,
                },
            )
    
    return response


class OverallAnalysisRequest(BaseModel):
    snapshot_id: str
    call_llm: Optional[bool] = True  # 是否调用 LLM，默认调用
    include_module_summaries: Optional[bool] = True
    provider: Optional[str] = None
    model: Optional[str] = None


@app.post("/v3/analysis/overall")
async def analyse_overall(req: OverallAnalysisRequest, auth: Any = Depends(require_auth)) -> Dict[str, Any]:
    """对所有模块进行综合 LLM 分析。

    该端点会：
    1. 获取快照中的所有模块数据
    2. 使用 build_llm_prompts() 构建综合分析 prompt
    3. 调用 Gemini API 生成完整的市场分析报告（如果 call_llm=True）
    
    参数：
    - snapshot_id: 快照 ID（由 /v3/snapshot/run 生成）
    - call_llm: 是否调用 LLM（默认 True）
    - include_module_summaries: 是否包含模块级别的信号/启发式标签/数据质量信息
    """
    snapshot = _get_snapshot(req.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    modules = snapshot["modules"]
    labels = snapshot["labels"]

    # 可选：拼接按钮1模块短评输出（若已生成）
    module_reports = snapshot.get("llm_module_reports") or {}
    report_blocks: List[str] = []
    for m in ["fundamentals", "liquidity", "sentiment", "technicals"]:
        item = module_reports.get(m) if isinstance(module_reports, dict) else None
        txt = (item.get("text") if isinstance(item, dict) else None) if item else None
        if txt:
            report_blocks.append(f"[{m}]\n{txt}")
    module_reports_text = "\n\n".join(report_blocks) if report_blocks else ""

    prompt_context_text = build_prompt_context(modules, labels)
    # 额外把 data_quality（缺失项）显式写进输入，便于模型做“可靠性边界”
    try:
        dq = snapshot.get("data_quality") or {}
        if isinstance(dq, dict) and dq:
            dq_lines = []
            for k in ["fundamentals", "liquidity", "sentiment", "technicals"]:
                miss = dq.get(k) or []
                miss_text = ", ".join(miss) if isinstance(miss, list) and miss else "无"
                dq_lines.append(f"- {k}: {miss_text}")
            prompt_context_text = prompt_context_text + "\n\n=== DATA_QUALITY_MISSING ===\n" + "\n".join(dq_lines)
    except Exception:
        pass

    as_of_date = str(snapshot.get("date") or "")
    system_prompt = GLOBAL_SYSTEM_PROMPT
    user_prompt = build_overall_user_prompt(
        as_of_date=as_of_date,
        prompt_context_text=prompt_context_text,
        labels=labels,
        module_reports_or_empty=module_reports_text,
        supplemental_overall=None,
    )
    
    response: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "labels": labels,
    }
    
    if req.include_module_summaries:
        response["heuristics"] = snapshot["heuristics"]
        response["data_quality"] = snapshot["data_quality"]
        response["signals"] = snapshot["signals"]
    
    # 调用 Gemini API
    if req.call_llm:
        analysis_text, error = call_gemini(
            system_prompt,
            user_prompt,
            temperature=1.0,
            max_tokens=10240,  # 总体分析需要更长的输出
            model=req.model,
        )
        if analysis_text:
            response["analysis"] = analysis_text
            response["llm_provider"] = "gemini"
            response["llm_model"] = _normalize_gemini_model_name(req.model)
            # 保存综合专业版输出，供 Chat 注入使用
            try:
                snapshot["last_overall_report"] = analysis_text
                snapshot["last_overall_report_model"] = response["llm_model"]
                snapshot["last_overall_report_ts"] = time.time()
            except Exception:
                pass
            _save_snapshot_to_disk(snapshot)
            _log_llm_event(
                kind="overall",
                snapshot_id=req.snapshot_id,
                record={
                    "model": response["llm_model"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "analysis": analysis_text,
                },
            )
        else:
            response["analysis"] = None
            response["llm_error"] = error
            _log_llm_event(
                kind="overall",
                snapshot_id=req.snapshot_id,
                record={
                    "model": _normalize_gemini_model_name(req.model),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": error,
                },
            )
    
    return response


class ChatRequest(BaseModel):
    snapshot_id: str
    messages: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    provider: Optional[str] = None
    model: Optional[str] = None


@app.post("/v3/chat")
async def chat(req: ChatRequest, auth: Any = Depends(require_auth)) -> Dict[str, Any]:
    """基于市场快照的多轮对话。

    该端点会：
    1. 获取快照数据构建上下文
    2. 使用 Gemini API 进行多轮对话
    
    参数：
    - snapshot_id: 快照 ID（由 /v3/snapshot/run 生成）
    - messages: 对话历史，格式为 [{"role": "user"|"assistant", "content": "..."}]
    
    返回：
    - system_prompt: 系统提示词（包含市场快照上下文）
    - reply: LLM 回复内容
    - llm_provider/llm_model: LLM 提供商和模型信息
    """
    snapshot = _get_snapshot(req.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    # 构建包含市场快照的系统提示词
    prompt_context = build_prompt_context(snapshot["modules"], snapshot["labels"])
    last_report = snapshot.get("last_overall_report")
    # 避免把超长报告全部塞进 system prompt 导致成本/延迟暴涨：做一个安全截断
    last_report_text = None
    if isinstance(last_report, str) and last_report.strip():
        max_chars = int(os.getenv("CHAT_LAST_REPORT_MAX_CHARS", "7000"))
        txt = last_report.strip()
        last_report_text = txt if len(txt) <= max_chars else (txt[:max_chars] + "\n\n（已截断：完整报告请重新调用 /v3/analysis/overall 获取）")
    system_prompt = build_chat_system_prompt(
        as_of_date=str(snapshot.get("date") or ""),
        prompt_context_text=prompt_context,
        last_overall_report_text_or_summary=last_report_text,
    )
    
    response: Dict[str, Any] = {
        "system_prompt": system_prompt,
    }
    
    # 调用 Gemini Chat API
    if req.messages:
        reply_text, error = call_gemini_chat(
            system_prompt,
            req.messages,
            temperature=1.0,
            max_tokens=10240,
            model=req.model,
        )
        if reply_text:
            response["reply"] = reply_text
            response["llm_provider"] = "gemini"
            response["llm_model"] = _normalize_gemini_model_name(req.model)
            _log_llm_event(
                kind="chat",
                snapshot_id=req.snapshot_id,
                record={
                    "model": response["llm_model"],
                    "system_prompt": system_prompt,
                    "messages": req.messages,
                    "reply": reply_text,
                },
            )
        else:
            response["reply"] = None
            response["llm_error"] = error
            _log_llm_event(
                kind="chat",
                snapshot_id=req.snapshot_id,
                record={
                    "model": _normalize_gemini_model_name(req.model),
                    "system_prompt": system_prompt,
                    "messages": req.messages,
                    "error": error,
                },
            )
    else:
        response["reply"] = "请发送您的问题。"
    
    return response


def build_llm_prompts(date_str: str, modules: Dict[str, Any], labels: Dict[str, str]) -> Dict[str, str]:
    """兼容旧调用：返回当前“综合专业版”提示词（按钮2）。

    注意：新的按钮2 prompt 已替换为结构化的一致性评分 + 跨资产链条 + 多资产细分框架。
    """
    prompt_context = build_prompt_context(modules, labels)
    system_prompt = GLOBAL_SYSTEM_PROMPT
    user_prompt = build_overall_user_prompt(
        as_of_date=date_str,
        prompt_context_text=prompt_context,
        labels=labels,
        module_reports_or_empty="",
        supplemental_overall=None,
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def build_prompt_context(modules: Dict[str, Any], labels: Dict[str, str]) -> str:
    """根据四大模块构建面向 LLM 的提示文本。"""
    fundamentals = modules["fundamentals"]
    liquidity = modules["liquidity"]
    sentiment = modules["sentiment"]
    technicals = modules["technicals"]

    news_lines = [f"- {n['title']} (source: {n.get('source')})" for n in fundamentals.get("news", [])]
    event_lines = [
        f"- {e.get('category')}: {e.get('actual')} (fcst {e.get('forecast')}, prev {e.get('previous')})"
        for e in fundamentals.get("events", [])
    ]
    assets = technicals.get("assets", {})
    asset_lines = []
    for name, data in assets.items():
        if not data:
            continue
        close_val = data.get("close")
        atr_text = _fmt_pct(data.get("atr_pct"), signed=False)
        asset_lines.append(
            f"- {name}: {_fmt_num(close_val, 2)} "
            f"(MA20 {_fmt_pct(data.get('distance_ma20_pct'))} / "
            f"MA50 {_fmt_pct(data.get('distance_ma50_pct'))} / "
            f"MA200 {_fmt_pct(data.get('distance_ma200_pct'))}, "
            f"ATR% {atr_text}, "
            f"trend {data.get('trend_label')})"
        )

    prompt = [
        "=== FUNDAMENTALS ===",
        f"- 2Y: {_fmt_num(fundamentals.get('dgs2'), 2, '%')} | 10Y: {_fmt_num(fundamentals.get('dgs10'), 2, '%')} "
        f"| Term spread: {_fmt_num(fundamentals.get('term_spread'), 2, '%')}",
        f"- Fed Funds: {_fmt_num(fundamentals.get('fedfunds'), 2, '%')} | FFR-2Y: {_fmt_num(fundamentals.get('ffr_minus_2y'), 2, '%')}",
        f"- DXY: {_fmt_num(fundamentals.get('dxy'))} (ticker {fundamentals.get('dxy_ticker') or 'DX-Y.NYB/DXY'})",
        f"- Key events:\n{chr(10).join(event_lines) if event_lines else '  (none, maybe credentials missing)'}",
        f"- Top news:\n{chr(10).join(news_lines) if news_lines else '  (none)'}",
        "",
        "=== LIQUIDITY ===",
        f"- Net liquidity: {_fmt_num(liquidity.get('net_liquidity'), 2, 'B')} (Δ4w {_fmt_num(liquidity.get('net_change_4w'), 2, 'B')})",
        f"- Components: WALCL {_fmt_num(liquidity.get('walcl'), 2, 'B')} | RRP {_fmt_num(liquidity.get('rrp'), 2, 'B')} | "
        f"TGA {_fmt_num(liquidity.get('tga'), 2, 'B')}",
        f"- Credit ratio (HYG/IEI): {_fmt_num(liquidity.get('credit_ratio'))} (Δ20d {_fmt_num(liquidity.get('credit_change_20d'))})",
        "",
        "=== SENTIMENT ===",
        f"- Fear & Greed: {sentiment.get('fgi_score')} ({sentiment.get('fgi_rating')}, source={sentiment.get('fgi_source')})",
        f"- VIX: {_fmt_num(sentiment.get('vix'))} | VIX3M: {_fmt_num(sentiment.get('vix3m'))} (source={sentiment.get('vix_term_source')}) "
        f"| Term structure: {sentiment.get('term_structure')}",
        f"- Put/Call: {_fmt_num(sentiment.get('put_call_ratio'))}",
        f"- Risk spreads: SPY-XLU {_fmt_pct(sentiment.get('spy_xlu'))} | HYG-IEF {_fmt_pct(sentiment.get('hyg_ief'))} | "
        f"BTC-Gold {_fmt_pct(sentiment.get('btc_gold'))}",
        "",
        "=== TECHNICALS ===",
        *asset_lines,
        f"- Breadth diff (RSP-SPX): {_fmt_pct(technicals.get('breadth_diff'))}",
        f"- Style ratio (XLK/XLP): {_fmt_num(technicals.get('style_ratio'))}",
        "",
        "=== LABELS ===",
        f"- Macro: {labels.get('macro_regime')} | Liquidity: {labels.get('liquidity_regime')} | "
        f"Sentiment: {labels.get('sentiment_regime')} | Technical: {labels.get('technical_regime')}",
    ]
    return "\n".join(prompt)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/context")
async def get_context(date: Optional[str] = Query(None, description="YYYY-MM-DD, default today")) -> Dict[str, Any]:
    target_date = _parse_date(date)
    date_str = target_date.isoformat()
    try:
        fundamentals = fetch_fundamentals(date_str)
        liquidity = fetch_liquidity(date_str)
        sentiment = fetch_sentiment(date_str)
        technicals = fetch_technicals(date_str)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to build context")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    modules = {
        "fundamentals": fundamentals,
        "liquidity": liquidity,
        "sentiment": sentiment,
        "technicals": technicals,
    }
    labels = assign_labels(modules)
    prompt_context = build_prompt_context(modules, labels)
    return {**modules, "date": date_str, "labels": labels, "prompt_context": prompt_context}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
