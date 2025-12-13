# Fintech_project
项目说明和实施手册（全流程文档）

说明

本文档面向开发者、量化研究者和交易者，描述如何构建一个面向交易决策支持的系统。系统通过中间件从多个公开数据源采集市场和宏观数据，经清洗、标准化和标签化后向大型语言模型 (LLM) 提供结构化输入，LLM 则生成每个模块和整体层面的分析报告。系统最终通过 Dify 工作流触发，在 Render 平台运行 Python FastAPI 服务，并将结果推送到前端或邮件端口。



### 一、项目目标

交易者在日度甚至日内决策时，希望快速回答四个核心问题：

1. 叙事背景 —— 今天的重要宏观事件是什么？结果与市场预期有何差异？利率和美元走势暗示了怎样的宏观叙事。 

2. 资金环境 —— 市场的上涨/下跌是否由真实资金支持？流动性是在回流还是抽离，信用市场是否支撑股市？ 

3. 市场情绪 —— 市场处于贪婪还是恐慌？期权和资产表现是否给出一致信号？是否存在明显错位或极端？

4. 价格结构 —— 当前价格处于趋势还是震荡？波动是否异常？市场广度与风格轮动如何？

为了服务上述问题，系统将数据采集和分析划分为四大模块：Fundamentals（宏观与政策）、Liquidity（流动性与成本）、Sentiment（情绪与风险偏好）以及 Technicals（价格结构与噪声）。每个模块负责收集特定数据、计算衍生指标、生成分类标签，并以统一的 JSON 结构返回供 LLM 使用。



### 二、总体流程

1. 配置并部署 Python 环境（建议 Python 3.10+），准备必要依赖（FastAPI、pandas、yfinance、fredapi、tradingeconomics、feedparser、fear‑greed‑index 等）。在 .env 文件中保存 FRED API Key 以及 Trading Economics 帐号密钥等必要凭证。 

2. 在 Render 平台创建 Web Service（Python 3），设置构建命令 pip install -r requirements.txt，启动命令 uvicorn app:app --host 0.0.0.0 --port $PORT。项目的主脚本 app.py（后端服务）暴露两个端点：/health 用于健康检查，/v1/context?date=YYYY-MM-DD 用于按日期返回结构化数据和提示文本。

&nbsp;3. Dify 创建工作流并配置定时触发器 (Cron)。工作流第一步使用 HTTP Request 节点调用 Render 服务的 /v1/context。如果 date 参数为空则自动获取当前交易日。 

4. FastAPI 服务按四个模块顺序采集数据、处理并构建统一 JSON。生成两个输出：raw\_json（包含所有原始/衍生数据）和 prompt\_context（一个面向 LLM 的多行文本，该文本摘要每个模块的关键数字和表格）。 

5. Dify 将 prompt\_context 传递给 LLM 节点。LLM 需根据预设的模板生成五部分内容：四个模块的总结 (fundamentals\_summary、liquidity\_summary、sentiment\_summary、technicals\_summary)，以及一个 cross\_asset\_view 和 risk\_stance，用于整合四模块信息给出总体观点。LLM 的输出一般采用结构化 JSON 便于前端渲染。 

6. 工作流最后使用通知节点（如 Email、SMS 或前端 API），将 LLM 结果和原始数据一起推送到用户。



### 三、环境与依赖

必需依赖：fastapi、uvicorn、pandas、numpy、requests、yfinance、fredapi、feedparser、fear-greed-index、tradingeconomics (若使用官方经济日历 API)、python-dotenv。建议使用 virtualenv 或 conda 管理环境，并在项目根目录下提供 requirements.txt 供安装。

API 凭证：

FRED API：前往圣路易斯联储网站注册后可免费获取。请将密钥写入 .env 文件，代码读取 FRED\_API\_KEY 变量。

Trading Economics：免费版需要申请 client key 和 secret。若不想注册，可选择备用爬虫（见下文）。

新闻接口（pygooglenews 或 GNews）：无需凭证，默认即可以。



**时间与时区：**模块输入接受 date 参数。例如 2025-12-11 代表以 2025 年 12 月 11 日的视角获取数据。若未传入 date，则自动使用当前交易日。请注意时区统一，例如美国东部时间收盘后更新数据，确保日期对应正确。所有时间序列需调用 pandas 的 asof 方法取最近可用值。



### 四、数据模块和数据源

本节逐个数据源说明获取方法、字段映射、速率限制、异常处理和备选方案。所有采集代码均应在 FastAPI 服务内实现，相互独立但共享统一的输出结构。



##### **1. Fundamentals — 宏观与政策**

###### **数据选择：**

\- 2 年和 10 年美国国债收益率、联邦基金利率：使用 FRED API 查询系列 DGS2（2 年期国债）、DGS10（10 年期国债）和 FEDFUNDS（联邦基金利率）。这些系列提供日度数据并无需付费。   



\- 美元指数 (DXY)：使用 yfinance 读取 DX-Y.NYB Ticker 的收盘价。yfinance 是基于 Yahoo Finance 的第三方库，安装使用简单但数据非官方，偶有缺口。  



 - 经济日历：优选 Trading Economics API 查询指定日期的宏观事件，其响应包含 Category、Previous、Forecast、Actual、Importance、Date 等字段。API 需要 client key 和 secret，免费版有速率限制（每分钟约 20 次）。若无法申请帐号，可改用爬虫抓取 Investing.com 网站的经济日历，但需处理网页结构变化。   



\- 宏观新闻：使用 Google News RSS 通过 feedparser 或 GNews 包搜索主题新闻。查询字符串可以包含关键字组，如“Federal Reserve OR FOMC OR rates”等。RSS 接口仅返回最近几天新闻，无法获取深度历史。



###### **请求方法与字段映射：**



**- FRED：**

from fredapi import Fred     fred = Fred(api\_key=os.getenv('FRED\_API\_KEY'))     series = fred.get\_series('DGS2', start\_date, end\_date)     



函数 get\_series 接受系列 ID 和日期区间，返回 pandas 序列。根据需要调用 series.asof(target\_date) 获取最后一个有效值。映射字段可定义为 dgs2, dgs10, fedfunds, term\_spread（10Y-2Y 差）和 ffr\_minus\_2y（FFR-2Y 差）。   



**- yfinance：**     

import yfinance as yf     dxy\_data = yf.download('DX-Y.NYB', start=start\_date, end=end\_date, progress=False)     dxy\_value = float(dxy\_data\['Close'].asof(target\_date))     



返回 DataFrame 需取 Close 列。注意 yfinance 调用可能因网络或 Yahoo Finance 限制失败，需要重试；若获取失败，可选择使用 ICE 或 CME 的 DXY 合约数据 (付费)。   



**- Trading Economics：**     

from tradingeconomics import login, getCalendar     login(client\_key, client\_secret)     events = getCalendar(start=target\_date, end=target\_date, country=\['United States'])     



返回列表中每条包含 Category 类别、Date 时间戳、Previous 前值、Forecast 预期和 Actual 公布值等字段。字段需映射到前端表格。若返回为空，则说明当天没有重大事件。   



**- 新闻：**     

from pygooglenews import GoogleNews     gn = GoogleNews(lang='en', country='US')     search = gn.search('Federal Reserve OR FOMC', when='3d')     news\_items = \[{'title': e.title, 'link': e.link, 'published': e.published} for e in search\['entries']\[:5]]     



可定义不同主题的关键词映射到 policy, equity, crypto, commodities 四类。若需要替代，可以改用 gnews 包，其接口返回 JSON 且支持更多筛选。



###### **限额与异常处理：**

\- FRED 日调用次数限制较高，单用户每天可调用几十万次，但建议缓存同一天结果避免重复请求。当 API 连接失败时返回 None 并记录异常。   - Trading Economics 免费版有速率限制（一般为每分钟 20 次），请求过多会返回 429 错误；应增加延迟或降级到备用爬虫。也可只在每天首次调用时获取日历并缓存。   - 新闻接口存在网络连接错误、响应为空等情况，需加入重试逻辑，若持续失败则返回空数组并在提示中说明数据缺失。



###### **替代方案：**

\- 若无法使用 FRED，可用国债 ETF 收益率作为短端/中端利率的 proxy。例如使用 yfinance 下载 SHY (1–3Y 国债 ETF) 和 IEF (7–10Y 国债 ETF) 收益率计算利差。但这种替代不如原系列精确。   



\- 若不想注册 Trading Economics，可爬取 Investing.com 经济日历 (HTML)，例如通过 BeautifulSoup 解析页面表格；但页面结构变化频繁，需要维护。   



\- 新闻模块可添加其他 RSS 源，如 Yahoo Finance、Bloomberg 或 The Guardian；然而大多数大型媒体需要付费授权，建议仅示例使用 Google 新闻。



##### **2. Liquidity — 流动性与资金成本**

###### **数据选择：**

**- 净流动性：**计算公式为 Net Liquidity = WALCL – RRPONTSYD – WTREGEN。其中 WALCL 表示联储总资产，RRPONTSYD 为隔夜逆回购余额，WTREGEN 为财政部一般账户余额。数据来自 FRED。   

**- 信用利差：**使用高收益债 ETF HYG 与中期国债 ETF IEI 或 IEF 收盘价之比作为信用利差 proxy。由 yfinance 提供。   

**- 期限利差：**使用 10Y-2Y 利差（已在 Fundamentals 模块计算）或额外计算 20 日变动用于趋势判断。



###### **请求方法与字段映射：**

**- FRED 组合：**

walcl = fred.get\_series('WALCL', start\_date, end\_date)     rrp = fred.get\_series('RRPONTSYD', start\_date, end\_date)     tga = fred.get\_series('WTREGEN', start\_date, end\_date)     net\_now = walcl.asof(target\_date) - rrp.asof(target\_date) - tga.asof(target\_date)     



除了当前值，可回溯 28 天前的值计算净流动性近四周变化 (net\_change\_4w)。字段映射包括 walcl, rrp, tga, net\_liquidity, net\_change\_4w。   



**- 信用比率：**     

import yfinance as yf     prices = yf.download(\['HYG', 'IEI'], start=start\_date, end=end\_date, progress=False)\['Close']     ratio\_now = float((prices\['HYG'] / prices\['IEI']).asof(target\_date))     ratio\_past = float((prices\['HYG'] / prices\['IEI']).asof(target\_date - pd.Timedelta(days=20)))     credit\_change\_20d = ratio\_now - ratio\_past     



字段包括 credit\_ratio 和 credit\_change\_20d。



###### **限额与异常处理：**

\- FRED 这三组系列的更新频率一般为周度或日度，若 asof 返回 NaN，应往前寻找最后一周的可用值。无法获取时标记 is\_imputed。   

\- yfinance 在周末或节假日不提供数据；若目标日期恰逢周末，应向前寻找最近交易日。若下载失败，可尝试更换 ETF，例如使用 LQD（投资级公司债）与 IEI 之比作为替代。   

\- 计算净流动性时不同系列具有不同公布延迟，若组合后出现负值或突变需在日志中记录，并在输出备注中说明；LLM 层可通过文本提示处理异常。

###### 

###### **替代方案：**

\- 如果 WALCL 等 FRED 指标不再公开，可考虑使用新闻报导的资产负债表变化以及资金市场互换利率等指标，但这些数据往往需付费。   

\- 信用利差亦可用 CDS 指数或 BAA-AAA 利差，但免费接口难以获取。



##### **3. Sentiment — 情绪与风险偏好**

###### **数据选择：**

\- CNN Fear \& Greed 指数：安装第三方包 fear-greed-index 调用 CNN 网站；该包返回 score（0 到 100 整数）和 rating（“fear”、“greed” 等）。另有 fear-greed-data 项目提供历史 CSV。   

\- 波动率期限结构：使用 yfinance 下载 ^VIX 和 ^VIX3M 收盘价，计算当日是否处于 contango（正常：VIX < VIX3M）或 backwardation（倒挂）。某些环境下 ^VIX3M 数据缺失，可改用相关期货或 ETF 近似，如 VIXY 或 UXM1 合约。   

\- CBOE Put/Call Ratio：CBOE 提供公开 CSV totalpc.csv，字段包括日期、CALLS、PUTS、P/C。通过 pandas 读取最后一行即可获取当日 PCR 值。   

\- Risk-on/off 比率：使用 yfinance 获取 SPY、XLU、HYG、IEF、BTC-USD、GLD 等多资产收盘价，计算日收益率差 (SPY – XLU，HYG – IEF，BTC – Gold)。

###### 

###### **请求方法与字段映射：**

**- Fear \& Greed：**     

from fear\_greed\_index.CNNFearAndGreedIndex import CNNFearAndGreedIndex     fgi = CNNFearAndGreedIndex()     current = fgi.get() # {'score': 40, 'rating': 'fear'}     



如需历史数据，可直接 pandas 读取 CSV：https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/master/fear-greed-historical.csv。映射字段包括 fgi\_score 和 fgi\_rating。   



**- VIX \& VIX3M：**     

vix = yf.download('^VIX', period='2mo')\['Close']     try:         vix3m = yf.download('^VIX3M', period='2mo')\['Close']     except Exception:         vix3m = None     vix\_today = float(vix.iloc\[-1]); vix3m\_today = float(vix3m.iloc\[-1]) if vix3m is not None else None     term\_structure = 'backwardation' if vix3m\_today and vix\_today > vix3m\_today else 'contango'     



字段包括 vix, vix3m, term\_structure。若 vix3m 缺失返回 None 并记录原因。   



**- Put/Call Ratio：**     

import pandas as pd     pc\_df = pd.read\_csv('https://cdn.cboe.com/resources/options/volume\_and\_call\_put\_ratios/totalpc.csv')     pc\_ratio = float(pc\_df\['P/C'].iloc\[-1])     



CBOE 文件可能因为网络或 CDN 更换而不可用，需准备备用地址或缓存历史。字段定义为 put\_call\_ratio。   



**- Risk-on/off：**     

prices = yf.download(\['SPY', 'XLU', 'HYG', 'IEF', 'BTC-USD', 'GLD'], period='35d')\['Close']     returns = prices.pct\_change()     spreads = {         'spy\_xlu': float(returns\['SPY'].iloc\[-1] - returns\['XLU'].iloc\[-1]),         'hyg\_ief': float(returns\['HYG'].iloc\[-1] - returns\['IEF'].iloc\[-1]),         'btc\_gold': float(returns\['BTC-USD'].iloc\[-1] - returns\['GLD'].iloc\[-1])     }     



字段包括 spy\_xlu, hyg\_ief, btc\_gold。



###### **限额与异常处理：**

\- Fear \& Greed 包会爬取 CNN 网站，若 CNN 更改页面结构将导致包不可用；应监控抛出异常并记录。备用方案是使用历史 CSV 数据并用最近收盘的 VIX、信贷利差等自定义算法估算情绪。   

\- ^VIX3M 在某些地区无法访问，下载会抛异常；可改用 ETF VIXY 或期货合约 UXM1 来近似。需在输出中增加 data\_missing 标识。   - CBOE CSV 不包含跨周期平均值，需要自定义滚动均值；同时可能出现空行或格式变动，应加入数据清洗逻辑。如果 CSV 无法下载可尝试由其他开源项目提供的镜像。   

\- yfinance 拉取多资产时可能因为某支资产停牌或网络错误导致空值；用 returns.asof() 处理缺失或改用更稳的 API（如 alpaca 或 IEX Cloud，需注册）。



###### **替代方案：**

\- 情绪可以引用 AAII Investors Sentiment Survey 或 NAAIM Exposure Index，但这类数据通常需要订阅。   

\- Optionmetrics 等机构提供的 Put/Call 数据更全面，但费用较高。若专业化程度不要求极高，可维持 CBOE CSV。



##### **4. Technicals — 价格结构与噪声**

###### **数据选择：**

\- 核心指数/ETF：^GSPC（标普 500）、^NDX（纳指）、RSP（标普等权重 ETF）、IWM（罗素 2000）、XLK（科技）、XLP（消费必需品）、XLU（公用事业）、DXY（美元指数）、GC=F（黄金期货）、CL=F（原油期货）、BTC-USD（比特币）。这些符号可全部由 yfinance 获取日度收盘价。   

\- 技术指标：20/50/200 日移动平均线 (MA20/50/200) 用于识别短/中/长期趋势；ATR14（平均真实波幅）用于测量市场噪声；布林带宽度（20 日标准差的两倍除以均线）用于判断波幅收缩或扩张。   

\- 广度与风格指标：RSP 收益率减去 SPY 收益率反映市场广度；XLK 与 XLP 价格比表示成长与防御风格轮动。



##### **请求方法与字段映射：**

**- 数据下载：**     

assets = \['^GSPC', '^NDX', 'RSP', 'IWM', 'XLK', 'XLP', 'XLU', 'DXY', 'GC=F', 'CL=F', 'BTC-USD']     hist = yf.download(assets, period='2y')\['Close']     

\#为计算长期均线，需至少 2 年数据。下载后，对于每个资产，使用以下函数计算指标：     

def calc\_indicators(series: pd.Series, target\_date: str):         s = series.dropna(); t = pd.to\_datetime(target\_date)         ma20 = s.rolling(20).mean().asof(t)         ma50 = s.rolling(50).mean().asof(t)         ma200 = s.rolling(200).mean().asof(t)         atr14 = s.diff().abs().rolling(14).mean().asof(t)         boll\_width = (s.rolling(20).std() \* 2 / s.rolling(20).mean()).asof(t)         close = s.asof(t)         return {             'close': float(close),             'distance\_ma20\_pct': float((close - ma20) / ma20),             'distance\_ma50\_pct': float((close - ma50) / ma50),             'distance\_ma200\_pct': float((close - ma200) / ma200),             'atr\_pct': float(atr14 / close),             'boll\_width': float(boll\_width),             'trend\_label': 'uptrend' if close > ma50 > ma200 else 'downtrend' if close < ma50 < ma200 else 'range'         }     

\#各字段分别对应收盘价、与移动均线的距离（%）、ATR 占比、布林带宽度和趋势标签。   



**- 广度和风格：**     

returns = hist.pct\_change()     breadth\_diff = float(returns\['RSP'].asof(target\_date) - returns\['^GSPC'].asof(target\_date))     style\_ratio = float(hist\['XLK'].asof(target\_date) / hist\['XLP'].asof(target\_date))     

\#广度差 (breadth\_diff) 和风格比率 (style\_ratio) 字段用于判断宽基指数是否由少数权重股驱动以及成长与防御轮动。

###### 

###### **限额与异常处理：**

\- yfinance 拉取大量资产时可能会超过请求限制，建议使用分批下载，或设置 threads=False。若某个资产返回为空，尝试重试或在指标中标记缺失。   

\- 价格空值或停牌日应通过前后值填充；移动均线需足够长度，如不足 200 个数据点则返回 None 并添加 trend\_label='unknown'。



###### **替代方案：**

\- 若使用 yfinance 不稳定，可改用 Alphavantage、IEX Cloud 或 Polygon.io，均提供官方 API，但需注册免费密钥。字段映射需相应调整。   

\- 技术指标也可通过 TA-Lib 计算，但该库编译复杂，不建议在初版中引入。



### 五、数据处理与标签化

在采集各模块数据后，需要统一处理流程，保证不同数据源之间时间一致、单位统一且缺失值合理填充。

1. 时间对齐：所有序列均以 target\_date 为基准，通过 asof 方法取最近可用值。若 target\_date 为周末或节假日，自动使用前一个交易日。 

2. 单位统一：收益率保留百分号前两位；大额值（如 WALCL）以十亿美元为单位输出；比率类直接使用浮点数；时间字段统一 ISO8601 格式。 

3. 缺失值填充：如某值缺失则尝试向前填充最近值，并设置 is\_imputed=True；如果无法填充则返回 None 并在 final 输出时注明。 

4. 标签判定：根据阈值判断宏观叙事和风险 regime。例如：当 2s10s 利差为负且绝对值大于 0.5 bps 时，宏观标签为 “Hawkish”，反之为 “Dovish”；当净流动性四周变化超过 +500 亿美元为 “Easing”，低于 –500 亿美元为 “Tightening”，介于其中为 “Neutral”；情绪标签根据 FGI 区间或 PCR、VIX 倒挂判定；技术标签根据趋势标签和波幅大小判定 “Trending” 或 “Range”。标签值将存储在输出的 labels 字段中。



### 六、统一输出结构



FastAPI 的 /v1/context 返回一个 JSON 对象，其中包含所有模块数据、标签和面向 LLM 的 prompt 文本：



{

&nbsp; "date": "2025-12-11",

&nbsp; "fundamentals": { ... },

&nbsp; "liquidity": { ... },

&nbsp; "sentiment": { ... },

&nbsp; "technicals": { ... },

&nbsp; "labels": {

&nbsp;    "macro\_regime": "Dovish",

&nbsp;    "liquidity\_regime": "Neutral",

&nbsp;    "sentiment\_regime": "Fear",

&nbsp;    "technical\_regime": "Range"

&nbsp; },

&nbsp; "prompt\_context": "=== MARKET DATA (Technical \& Price) ===\\n...\\n=== LIQUIDITY \& MACRO ===\\n...\\n=== SENTIMENT ===\\n...\\n=== TECHNICALS ===\\n..."

}



其中 prompt\_context 是一段多行文本，按如下模板生成：



=== FUNDAMENTALS ===

\- 2Y: 3.57% | 10Y: 4.17% | Term spread: 0.60% | Fed Funds: 5.25% | FFR-2Y: 1.68%

\- DXY: 105.20

\- Key events: \[CPI (actual 3.1%, forecast 3.3%), ...]

\- Top news:

&nbsp; \* Fed hints at pause (source: Bloomberg)

&nbsp; \* ...



=== LIQUIDITY ===

\- Net liquidity: -590B USD (Δ4w -50B)

\- Components: WALCL 8.45T, RRP 1.33T, TGA 0.45T

\- Credit ratio (HYG/IEI): 1.12 (Δ20d +0.03)



=== SENTIMENT ===

\- Fear \& Greed: 40 (Fear)

\- VIX: 18.5 | VIX3M: 19.7 → term structure: contango

\- Put/Call: 0.93 (10d avg: 0.81)

\- Risk spreads: SPY–XLU +0.5%, HYG–IEF –0.1%, BTC–Gold +1.8%



=== TECHNICALS ===

\- SPX: 4200 (MA20+3.0%, MA50+5.5%, MA200+10.0%, ATR% 1.0%, trend uptrend)

\- ... (other assets)

\- Breadth diff (RSP–SPY): –0.4%

\- Style ratio (XLK/XLP): 2.1

以上示例仅供参考，具体格式可根据前端需求调整。



### 七、Dify 集成与 LLM 生成

1. 创建工作流：在 Dify 中创建 Workflow，选择 schedule trigger 设置每天某个时间自动运行。也可通过 Gmail Trigger 监听指定邮件内容触发。 

2. HTTP 请求节点：配置请求方式为 GET，URL 指向 Render 服务的 /v1/context（例如 https://your‑service.onrender.com/v1/context?date={{ date }}）。设置超时和重试次数；将返回 JSON 保存为变量。例如 context\_data。 

3. LLM 节点：编写 Prompt，包含 {{ context\_data.prompt\_context }} 以及要求其输出固定 JSON。分模块部分使用 fundamentals\_summary, liquidity\_summary, sentiment\_summary, technicals\_summary，整合部分使用 cross\_asset\_view 和 risk\_stance。例如：“请根据提供的上下文数据撰写各模块摘要和综合观点，并输出 JSON 格式。”。 

4. 输出节点：选择 Email、Notion、Slack 等插件，将 context\_data 和 LLM 输出合并发送给用户。邮件正文可包含两部分：上半部分展示原始数据表格和指标，下半部分展示 LLM 的文字分析。



### 八、Render 部署简要步骤

1. 本地测试：在本地运行 FastAPI 服务，确保 uvicorn app:app --reload 可以访问 /v1/context 并返回正确 JSON。利用单元测试检查各模块函数的边界情况（如无数据或异常网络）。 

2. 代码仓库：将项目代码推送到 GitHub 或 GitLab，并确保包含 requirements.txt、app.py、Dockerfile 或 Render 所需的环境配置文件。建议在根目录写入 .env.example 供其他人参考。 

3. Render 创建服务：登录 Render，点击 “New Web Service”，选择与代码仓库关联。设置运行环境为 Python；在 Build Command 中填入 pip install -r requirements.txt；在 Start Command 中填入 uvicorn app:app --host 0.0.0.0 --port $PORT。配置环境变量，如 FRED\_API\_KEY, TE\_CLIENT\_KEY, TE\_CLIENT\_SECRET，确保与 .env 匹配。 

4. 部署验证：等待 Render 部署完成后访问 https://your‑service.onrender.com/health，若返回状态正常，则可用。随后在 Dify 中配置 HTTP 请求节点使用该地址。注意 Render 免费版会在长时间无请求时休眠，可以通过计划任务定期唤醒。



### 九、错误处理与维护建议

1. 错误捕获：每个请求应捕获网络异常 (requests.exceptions.RequestException)、HTTP 状态异常（如 4xx/5xx）、解析异常 (KeyError, ValueError)。应记录错误并继续执行，返回 None 或空列表而非终止进程。 

2. 速率限制与重试：对 Trading Economics 和 yfinance 添加简单的指数退避重试机制（sleep 增长），并设定最大重试次数。如果在获取新闻时超过查询配额，应返回空列表并在提示上下文中标明数据缺失。 

3. 缓存与持久化：可使用本地缓存（如 SQLite 或 Redis）存储每日数据，避免同一天重复请求。也可定期将数据写入对象存储以备日后回溯分析。 

4. 更新与替换：密切关注数据源 API 的变更。如果 FRED 移除某个系列或更改返回格式，需及时更新代码。对于未稳定的数据源（如 fear-greed-index 包），可监测 GitHub 更新或改用其他情绪指标。 

5. 扩展与版本控制：项目应使用版本控制管理；在开发新功能（例如添加 FedWatch 概率或经济惊喜指数）时，应在分支中进行，并更新本文档。输出结构如有变化（增加字段或调整标签阈值），应在文档中注明并通知前端开发者和 LLM Prompt 设计者。



### 十、结语

本文档详细说明了构建日度市场监控与交易辅助系统的步骤和原理。通过公开数据源采集利率、流动性、情绪和技术指标，并结合分类标签，系统为大型语言模型提供结构化输入，使其能够生成各模块总结和综合观点。项目的流水线由 Dify 工作流触发，FastAPI 服务部署在 Render，既保证了数据的实时性和稳定性，也方便后续扩展。遵循此说明，开发者可以快速搭建原型并根据实际需求定制指标、数据源和输出形式。



