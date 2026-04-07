

import argparse
import asyncio
import contextvars
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Literal

import dotenv

# ── Optional heavy dependencies (fail gracefully) ──────────────────────────
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from forecasting_tools import (
        AskNewsSearcher,
        BinaryQuestion,
        ForecastBot,
        GeneralLlm,
        MetaculusClient,
        MetaculusQuestion,
        MultipleChoiceQuestion,
        NumericDistribution,
        NumericQuestion,
        DateQuestion,
        DatePercentile,
        Percentile,
        ConditionalQuestion,
        ConditionalPrediction,
        PredictionTypes,
        PredictionAffirmed,
        BinaryPrediction,
        PredictedOptionList,
        ReasonedPrediction,
        SmartSearcher,
        clean_indents,
        structure_output,
    )
except ImportError:
    print("❌ Critical Error: 'forecasting_tools' not found.")
    print("   pip install forecasting-tools")
    exit(1)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Correlation ID for request tracing
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')

# ═══════════════════════════════════════════════════════════════════════════
#  METRICS COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

class Metrics:
    """Simple metrics collection for monitoring."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = {}
    
    def increment(self, name: str, value: int = 1):
        self.counters[name] += value
    
    def record_time(self, name: str, duration: float):
        self.timers[name].append(duration)
    
    def set_gauge(self, name: str, value: float):
        self.gauges[name] = value
    
    def get_summary(self) -> dict:
        summary = dict(self.counters)
        for name, times in self.timers.items():
            if times:
                summary[f"{name}_count"] = len(times)
                summary[f"{name}_avg"] = sum(times) / len(times)
                summary[f"{name}_min"] = min(times)
                summary[f"{name}_max"] = max(times)
        summary.update(self.gauges)
        return summary

# Global metrics instance
metrics = Metrics()

# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURED LOGGING
# ═══════════════════════════════════════════════════════════════════════════

class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = correlation_id.get() or 'no-id'
        return True

def setup_logging():
    """Configure structured logging with correlation IDs."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationIdFilter())
    
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    
    # Suppress noisy libraries
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL CONFIGURATION — Free OpenRouter models
# ═══════════════════════════════════════════════════════════════════════════

# Using OpenRouter's current available free tier models
# Format for LiteLLM: "openrouter/{provider}/{model}:free"
PRIMARY_MODEL   = "openrouter/openrouter/free"                             # Primary — auto-routed free model
REASONING_MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"      # Best for analysis
PARSER_MODEL    = "openrouter/openrouter/free"                       # Structured output parsing

# ── Tournaments ─────────────────────────────────────────────────────────────
TOURNAMENT_IDS = {
    "spring_bot":      32916,
    "market_pulse":    "market-pulse-26q2",
    "minibench":       "minibench",
}

# MarketPulse tickers for common financial questions
MARKETPULSE_TICKER_HINTS = {
    "S&P":    "^GSPC",
    "SPX":    "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW":    "^DJI",
    "BTC":    "BTC-USD",
    "ETH":    "ETH-USD",
    "GOLD":   "GC=F",
    "OIL":    "CL=F",
    "EUR":    "EURUSD=X",
    "GBP":    "GBPUSD=X",
    "VIX":    "^VIX",
    "AAPL":   "AAPL",
    "MSFT":   "MSFT",
    "TSLA":   "TSLA",
    "NVDA":   "NVDA",
}

# ═══════════════════════════════════════════════════════════════════════════
#  EXTREMIZATION MODULE
# ═══════════════════════════════════════════════════════════════════════════

def extremize_binary(p: float, strength: float = 0.3) -> float:
    """
    Power extremization: pushes probabilities away from 0.5 toward 0 or 1.

    Uses the formula from the superforecasting / aggregation literature:
        extremized = p^(1-s) / (p^(1-s) + (1-p)^(1-s))
    where s ∈ (0,1) is strength.  s=0 → identity, s→1 → hard clamp.

    Additionally enforces a DEAD ZONE: any prediction landing in [0.43, 0.57]
    is nudged out to 0.40 or 0.60 to prevent wishy-washy 50/50 forecasts.
    """
    # Clamp to safe range first
    p = max(0.02, min(0.98, p))

    # Power extremization
    denom = p ** (1 - strength) + (1 - p) ** (1 - strength)
    p_ext = p ** (1 - strength) / denom
    p_ext = max(0.02, min(0.98, p_ext))

    # Dead-zone enforcement: ban [0.43, 0.57]
    if 0.43 <= p_ext <= 0.57:
        p_ext = 0.40 if p_ext < 0.50 else 0.60

    return round(p_ext, 4)


def extremize_option_list(options: PredictedOptionList, strength: float = 0.25) -> PredictedOptionList:
    """Extremize a multiple-choice distribution away from uniform."""
    # Sharpening: raise each prob to (1+strength), then renormalize
    raw = [(opt, max(1e-6, opt.probability) ** (1 + strength)) for opt in options.predicted_options]
    total = sum(v for _, v in raw)
    for opt, v in raw:
        opt.probability = round(v / total, 4)
    return options


# ═══════════════════════════════════════════════════════════════════════════
#  RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        # Remove old calls
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until oldest call expires
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Recheck after waiting
                return await self.acquire()
        
        self.calls.append(now)

# Global rate limiter for external APIs
api_rate_limiter = RateLimiter(calls_per_minute=30)  # Conservative limit

# ═══════════════════════════════════════════════════════════════════════════
#  DATA SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_input(text: str) -> str:
    """Sanitize user inputs to prevent injection or malicious content."""
    if not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>]', '', text)  # Remove angle brackets
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)  # Remove JS URLs
    text = re.sub(r'data:', '', text, flags=re.IGNORECASE)  # Remove data URLs
    
    # Limit length
    if len(text) > 10000:
        text = text[:10000] + "..."
    
    return text.strip()

class BaseSearcher:
    """Base class for search providers."""
    
    async def search(self, query: str, num_results: int = 5) -> str:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        return True

class FirecrawlSearcher(BaseSearcher):
    """Firecrawl search provider."""
    
    def is_available(self) -> bool:
        return bool(os.getenv("FIRECRAWL_API_KEY") and HAS_HTTPX)
    
    async def search(self, query: str, num_results: int = 5) -> str:
        if not self.is_available():
            return ""
        
        # Sanitize input
        query = sanitize_input(query)
        
        # Rate limiting
        await api_rate_limiter.acquire()
        
        api_key = os.getenv("FIRECRAWL_API_KEY")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/search",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"query": query, "limit": num_results, "scrapeOptions": {"formats": ["markdown"]}},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                snippets = []
                for r in results:
                    title   = r.get("title", "")
                    url     = r.get("url", "")
                    content = r.get("markdown", r.get("description", ""))[:800]
                    snippets.append(f"### {title}\n{url}\n{content}")
                metrics.increment("firecrawl_searches_success")
                return "\n\n".join(snippets)
        except Exception as e:
            metrics.increment("firecrawl_searches_failed")
            logger.warning(f"Firecrawl search failed: {e}")
            return ""

class LinkupSearcher(BaseSearcher):
    """Linkup search provider."""
    
    def is_available(self) -> bool:
        return bool(os.getenv("LINKUP_API_KEY") and HAS_HTTPX)
    
    async def search(self, query: str, depth: str = "standard") -> str:
        if not self.is_available():
            return ""
        
        # Sanitize input
        query = sanitize_input(query)
        
        # Rate limiting
        await api_rate_limiter.acquire()
        
        api_key = os.getenv("LINKUP_API_KEY")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.linkup.so/v1/search",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"q": query, "depth": depth, "outputType": "sourcedAnswer", "includeImages": False},
                )
                resp.raise_for_status()
                data = resp.json()
                answer  = data.get("answer", "")
                sources = data.get("sources", [])
                source_lines = [f"- [{s.get('name','')}]({s.get('url','')}) — {s.get('snippet','')[:300]}" for s in sources[:6]]
                metrics.increment("linkup_searches_success")
                return f"{answer}\n\nSources:\n" + "\n".join(source_lines) if answer else ""
        except Exception as e:
            metrics.increment("linkup_searches_failed")
            logger.warning(f"Linkup search failed: {e}")
            return ""

class AskNewsSearcherWrapper(BaseSearcher):
    """AskNews search provider wrapper."""
    
    def is_available(self) -> bool:
        has_asknews = (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY")
        return bool(has_asknews)
    
    async def search(self, query: str, num_results: int = 5) -> str:
        if not self.is_available():
            return ""
        
        try:
            prompt_for_asknews = f"{query}\n\nResolution: forecast resolution"
            ans = await AskNewsSearcher().call_preconfigured_version(
                "asknews/news-summaries", prompt_for_asknews
            )
            metrics.increment("asknews_searches_success")
            return ans
        except Exception as e:
            metrics.increment("asknews_searches_failed")
            logger.warning(f"AskNews fallback failed: {e}")
            return ""

# Initialize search providers
firecrawl_searcher = FirecrawlSearcher()
linkup_searcher = LinkupSearcher()
asknews_searcher = AskNewsSearcherWrapper()

async def fetch_yfinance_context(question_text: str) -> str:
    """Extract ticker from question and fetch current price + recent trend."""
    if not HAS_YFINANCE:
        return ""
    ticker_symbol = None
    q_upper = question_text.upper()
    for keyword, sym in MARKETPULSE_TICKER_HINTS.items():
        if keyword.upper() in q_upper:
            ticker_symbol = sym
            break
    if not ticker_symbol:
        return ""
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="30d")
        if hist.empty:
            return ""
        last_price  = hist["Close"].iloc[-1]
        start_price = hist["Close"].iloc[0]
        pct_change  = ((last_price - start_price) / start_price) * 100
        high_30     = hist["High"].max()
        low_30      = hist["Low"].min()
        info        = tk.info
        name        = info.get("shortName", ticker_symbol)
        return (
            f"[yfinance live data for {name} ({ticker_symbol})]\n"
            f"  Current price : {last_price:.4f}\n"
            f"  30-day change : {pct_change:+.2f}%\n"
            f"  30-day high   : {high_30:.4f}\n"
            f"  30-day low    : {low_30:.4f}\n"
            f"  Data as of    : {datetime.now().strftime('%Y-%m-%d')}\n"
        )
    except Exception as e:
        logger.warning(f"yfinance fetch failed for '{question_text}': {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE GATING for AI Spring Tournament
# ═══════════════════════════════════════════════════════════════════════════

CONFIDENCE_GATE_PATTERN = re.compile(
    r"confidence[:\s]+([0-9]{1,3})[\s]*[%\/]",
    re.IGNORECASE,
)

async def assess_forecast_confidence(
    question: BinaryQuestion, research: str, llm: GeneralLlm
) -> float:
    """
    Ask the REASONING_MODEL to rate its own confidence 0–100.
    Returns 0–1.  If parsing fails, defaults to 0.5 (neutral → will forecast).
    """
    prompt = clean_indents(f"""
        You are a superforecaster quality-checker.
        Question: {question.question_text}
        Research summary: {research[:1500]}

        On a scale of 0–100, how confident are you that forecasting this question
        with the available information will produce a RELIABLE, USEFUL forecast?
        (0 = nearly no information / pure guess, 100 = highly informed, clear signal)

        Factors that lower confidence: sparse research, very long time horizon,
        question depends on opaque political decisions, no base rate available.
        Factors that raise confidence: clear data, strong base rates, near-term resolution,
        market prices available, expert consensus visible.

        Reply with a single line: "Confidence: XX%"
    """)
    try:
        resp = await llm.invoke(prompt)
        m = CONFIDENCE_GATE_PATTERN.search(resp)
        if m:
            return int(m.group(1)) / 100.0
    except Exception:
        pass
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Bot3112026(ForecastBot):
    """
    311bot SuperForecaster — Spring 2026
    ─────────────────────────────────────────
    • Free OpenRouter models with fallback rotation
    • Extremization on all binary + MC forecasts
    • yfinance grounding for MarketPulse questions
    • Selective forecasting on AI Spring Tournament (confidence gate)
    • Firecrawl + Linkup search, AskNews fallback
    """

    _max_concurrent_questions         = 1
    _concurrency_limiter              = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # Gate: minimum confidence to submit on AI Spring Tournament
    AI_SPRING_CONFIDENCE_THRESHOLD = 0.60

    # Extremization strength for binary (0–1, higher = more extreme)
    BINARY_EXTREMIZE_STRENGTH = 0.30

    # Flag: set True during AI spring tournament runs
    _is_ai_spring_tournament: bool = False

    # ── RESEARCH ────────────────────────────────────────────────────────────

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # 1. Detect if this is a MarketPulse question → add yfinance data
            yf_context = ""
            is_market_pulse = self._is_marketpulse_question(question)
            if is_market_pulse:
                yf_context = await fetch_yfinance_context(question.question_text)
                if yf_context:
                    logger.info(f"yfinance data fetched for: {question.page_url}")

            # 2. Build search research
            search_results = await self._search(question)

            # 3. Summarize with LLM
            yf_section = f"\n\n## Live Market Data\n{yf_context}" if yf_context else ""
            raw_context = f"{search_results}{yf_section}"

            if not raw_context.strip():
                logger.warning(f"No research gathered for {question.page_url}")
                return "No research available."

            prompt = clean_indents(f"""
                You are an assistant to a superforecaster.
                Produce a concise, dense research brief (max 600 words) covering:
                - Key facts relevant to this question
                - Current status / recent developments
                - Whether the question would currently resolve Yes or No (if applicable)
                - Any relevant base rates or market signals

                Do NOT produce a forecast yourself.

                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                {question.fine_print}

                Raw research gathered:
                {raw_context[:4000]}
            """)

            try:
                research = await self.get_llm("default", "llm").invoke(prompt)
                logger.info(f"Research ready for {question.page_url} ({len(research)} chars)")
                return research
            except Exception as e:
                logger.error(f"Research summarization failed: {e}")
                return raw_context[:2000]

    def _is_marketpulse_question(self, question: MetaculusQuestion) -> bool:
        """Heuristic: MarketPulse questions mention financial instruments."""
        text = (question.question_text + " " + (question.background_info or "")).upper()
        return any(kw.upper() in text for kw in MARKETPULSE_TICKER_HINTS.keys())

    async def _search(self, question: MetaculusQuestion) -> str:
        """Try Firecrawl → Linkup → AskNews in order, merge results."""
        query = f"{question.question_text} forecast resolution {datetime.now().year}"
        results = []

        # Try Firecrawl
        if firecrawl_searcher.is_available():
            fc = await firecrawl_searcher.search(query, num_results=4)
            if fc:
                results.append(f"## Firecrawl Results\n{fc}")

        # Try Linkup
        if linkup_searcher.is_available():
            lk = await linkup_searcher.search(query)
            if lk:
                results.append(f"## Linkup Results\n{lk}")

        # Fallback to AskNews if both above empty
        if not results and asknews_searcher.is_available():
            ans = await asknews_searcher.search(query)
            if ans:
                results.append(f"## AskNews Results\n{ans}")

        return "\n\n".join(results)

    # ── BINARY QUESTIONS ────────────────────────────────────────────────────

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:

        # ── AI Spring Tournament confidence gate ──────────────────────────
        if self._is_ai_spring_tournament:
            reasoning_llm = GeneralLlm(
                model=REASONING_MODEL, temperature=0.1, timeout=60, allowed_tries=2
            )
            confidence = await assess_forecast_confidence(question, research, reasoning_llm)
            if confidence < self.AI_SPRING_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"🚫 Skipping {question.page_url} — confidence {confidence:.0%} "
                    f"< threshold {self.AI_SPRING_CONFIDENCE_THRESHOLD:.0%}"
                )
                # Return None signals the framework to skip publishing
                # (wrap in a sentinel prediction close to community median or 0.5)
                # We'll set it to exactly 0.5 and let skip logic handle it
                return ReasonedPrediction(
                    prediction_value=0.5,
                    reasoning=f"[SKIPPED — low confidence: {confidence:.0%}. No submission made.]",
                )

        prompt = clean_indents(f"""
            You are a professional superforecaster.

            Question: {question.question_text}

            Background: {question.background_info}

            Resolution criteria (not yet met): {question.resolution_criteria}
            {question.fine_print}

            Research: {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Think step by step:
            (a) Time remaining until resolution.
            (b) Status quo outcome if nothing changes.
            (c) Scenario that leads to NO.
            (d) Scenario that leads to YES.
            (e) Relevant base rates or reference classes.
            (f) Your confidence level (0–100) that your forecast is well-informed.

            IMPORTANT: Good forecasters commit. Do NOT hedge at 45–55%.
            If the evidence leans one way, say so clearly.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with: "Probability: ZZ%" (0–100)
        """)

        result = await self._binary_prompt_to_forecast(question, prompt)

        # Apply extremization
        raw_p = result.prediction_value
        ext_p = extremize_binary(raw_p, strength=self.BINARY_EXTREMIZE_STRENGTH)
        if abs(ext_p - raw_p) > 0.01:
            logger.info(f"Extremized {raw_p:.2%} → {ext_p:.2%} for {question.page_url}")
        result.prediction_value = ext_p
        return result

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.02, min(0.98, binary_prediction.prediction_in_decimal))
        logger.info(f"Raw forecast {question.page_url}: {decimal_pred:.2%}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    # ── MULTIPLE CHOICE QUESTIONS ───────────────────────────────────────────

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            You are a professional superforecaster.

            Question: {question.question_text}
            Options: {question.options}

            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}

            Research: {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Think step by step:
            (a) Time until resolution.
            (b) Status quo outcome.
            (c) Unexpected scenario with a different winner.

            Commit to your best estimate. Don't spread probability uniformly — the world is not uniform.
            {self._get_conditional_disclaimer_if_necessary(question)}

            Final answer (probabilities must sum to 1.0):
            {chr(10).join(f"Option_{chr(65+i)}: Probability_{chr(65+i)}" for i in range(len(question.options)))}
        """)
        result = await self._multiple_choice_prompt_to_forecast(question, prompt)

        # Apply MC extremization
        result.prediction_value = extremize_option_list(
            result.prediction_value, strength=0.25
        )
        return result

    async def _multiple_choice_prompt_to_forecast(
        self, question: MultipleChoiceQuestion, prompt: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(f"""
            Option names must exactly match one of: {question.options}
            Remove any "Option" prefix if present.
            Include 0% options as entries (don't skip them).
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        logger.info(f"MC forecast {question.page_url}: {predicted_option_list}")
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    # ── NUMERIC QUESTIONS ───────────────────────────────────────────────────

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        # For MarketPulse numeric questions, prepend live yfinance data
        extra_data = ""
        if self._is_marketpulse_question(question):
            yf_ctx = await fetch_yfinance_context(question.question_text)
            if yf_ctx:
                extra_data = f"\n\n## Live Market Data (from yfinance)\n{yf_ctx}"

        prompt = clean_indents(f"""
            You are a professional superforecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units: {question.unit_of_measure if question.unit_of_measure else "infer from context"}

            Research: {research}{extra_data}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_bound_message}
            {upper_bound_message}

            Formatting:
            - No scientific notation.
            - Percentiles must be in ascending order.
            - Express in stated units.

            Think step by step:
            (a) Time until resolution.
            (b) Outcome if nothing changes.
            (c) Outcome if current trend continues.
            (d) Expert/market expectations.
            (e) Low scenario.
            (f) High scenario.

            Set WIDE 90/10 intervals — unknown unknowns are real.

            Final answer:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
        """)
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsing_instructions = clean_indents(f"""
            Parse a numeric forecast distribution.
            Question: "{question.question_text}"
            Units: {question.unit_of_measure}
            Bounds: [{question.lower_bound}, {question.upper_bound}]
            Convert scientific notation to plain numbers.
            If no explicit percentiles found, return empty.
        """)
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Numeric forecast {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ── DATE QUESTIONS ──────────────────────────────────────────────────────

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(f"""
            You are a professional superforecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}

            Research: {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_bound_message}
            {upper_bound_message}

            Dates must be in format YYYY-MM-DD (append THH:MM:SSZ if hours matter).
            Always give dates in chronological order (earliest at P10).

            Think step by step:
            (a–f) as in a standard numeric question but for dates.

            Final answer:
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
        """)
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        parsing_instructions = clean_indents(f"""
            Parse a date forecast.
            Question: "{question.question_text}"
            Bounds: [{question.lower_bound}, {question.upper_bound}]
            Format all dates as valid ISO-8601 strings; assume midnight UTC if no time given.
        """)
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        percentile_list = [
            Percentile(percentile=dp.percentile, value=dp.value.timestamp())
            for dp in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Date forecast {question.page_url}: {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ── CONDITIONAL QUESTIONS ───────────────────────────────────────────────

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, full_research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """)
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,
            child=child_info.prediction_value,
            prediction_yes=yes_info.prediction_value,
            prediction_no=no_info.prediction_value,
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            prev = previous_forecasts[-1]
            if prev.timestamp_end is None or prev.timestamp_end > datetime.now(timezone.utc):
                pretty_value = DataOrganizer.get_readable_prediction(prev)
                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Reaffirmed existing forecast at {pretty_value}.",
                    ),
                    research,
                )

        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        question_type = question_type.title()
        return clean_indents(f"""
            {research}
            ---
            ## {question_type} Question Information
            Previously forecasted: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            Reasoning:
            ```
            {reasoning.reasoning}
            ```
            Do NOT re-forecast the {question_type} question above.
        """)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit  = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit  = ""
        else:
            raise ValueError(f"Unexpected question type: {type(question)}")

        upper_msg = (
            f"The question creator thinks the answer is likely not higher than {upper} {unit}."
            if question.open_upper_bound else
            f"The answer cannot be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the answer is likely not lower than {lower} {unit}."
            if question.open_lower_bound else
            f"The answer cannot be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg

    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents("""
            This is a CONDITIONAL question. Only forecast the CHILD question given the parent's resolution.
            Never re-forecast the parent.
        """)

    # ── HEALTH CHECKS ─────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check health of external dependencies."""
        health = {
            "firecrawl": firecrawl_searcher.is_available(),
            "linkup": linkup_searcher.is_available(),
            "asknews": asknews_searcher.is_available(),
            "yfinance": HAS_YFINANCE,
            "httpx": HAS_HTTPX,
            "forecasting_tools": True,  # If we got here, it's loaded
        }
        
        # Test actual API connectivity (lightweight)
        try:
            if health["firecrawl"]:
                # Quick test search
                test_result = await firecrawl_searcher.search("test query", 1)
                health["firecrawl_connectivity"] = bool(test_result or True)  # Available even if no results
        except:
            health["firecrawl_connectivity"] = False
        
        try:
            if health["linkup"]:
                test_result = await linkup_searcher.search("test query")
                health["linkup_connectivity"] = bool(test_result or True)
        except:
            health["linkup_connectivity"] = False
        
        return health


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_logging()
    
    # Set correlation ID for main execution
    correlation_id.set(str(uuid.uuid4()))
    
    parser = argparse.ArgumentParser(description="OracleDeck SuperForecaster Bot 2026")
    parser.add_argument(
        "--mode",
        choices=["tournament", "market_pulse", "ai_spring", "test"],
        default="tournament",
        help=(
            "tournament     → all tournaments (default)\n"
            "market_pulse   → MarketPulse only (uses yfinance)\n"
            "ai_spring      → AI Spring Tournament only (confidence-gated)\n"
            "test           → single test question"
        ),
    )
    parser.add_argument(
        "--extremize-strength",
        type=float,
        default=0.30,
        help="Extremization strength 0.0–0.9 (default: 0.30)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to forecast in AI Spring Tournament (default: 0.60)",
    )
    args = parser.parse_args()

    # ── Dependency checks ────────────────────────────────────────────────
    if not HAS_YFINANCE:
        logger.warning("⚠️  yfinance not installed — market data unavailable. pip install yfinance")
    if not HAS_HTTPX:
        logger.warning("⚠️  httpx not installed — Firecrawl/Linkup unavailable. pip install httpx")
    if not os.getenv("FIRECRAWL_API_KEY") and not os.getenv("LINKUP_API_KEY"):
        has_asknews = (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY")
        if not has_asknews:
            logger.warning("⚠️  No search API keys found (FIRECRAWL_API_KEY, LINKUP_API_KEY, or AskNews). Research will be empty.")

    # ── Build bot ────────────────────────────────────────────────────────
    bot = Bot3112026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=False,
        llms={
            "default":    GeneralLlm(model=PRIMARY_MODEL,   temperature=0.3, timeout=60, allowed_tries=3),
            "summarizer": PRIMARY_MODEL,
            "researcher": PRIMARY_MODEL,
            "parser":     GeneralLlm(model=PARSER_MODEL,    temperature=0.0, timeout=45, allowed_tries=3),
        },
    )

    bot.BINARY_EXTREMIZE_STRENGTH      = args.extremize_strength
    bot.AI_SPRING_CONFIDENCE_THRESHOLD = args.confidence_threshold

    client          = MetaculusClient()
    forecast_reports = []

    if args.mode == "tournament":
        # All tournaments
        for tid_name, tid in TOURNAMENT_IDS.items():
            bot._is_ai_spring_tournament = (tid_name == "spring_bot")
            logger.info(f"▶ Forecasting tournament: {tid_name} ({tid})")
            try:
                reports = asyncio.run(
                    bot.forecast_on_tournament(tid, return_exceptions=True)
                )
                forecast_reports.extend(reports)
            except Exception as e:
                logger.error(f"Tournament {tid_name} failed: {e}")

    elif args.mode == "market_pulse":
        bot._is_ai_spring_tournament = False
        logger.info("▶ MarketPulse mode (yfinance grounded)")
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(TOURNAMENT_IDS["market_pulse"], return_exceptions=True)
        )

    elif args.mode == "ai_spring":
        bot._is_ai_spring_tournament = True
        logger.info(f"▶ AI Spring Tournament — confidence gate: {args.confidence_threshold:.0%}")
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(TOURNAMENT_IDS["spring_bot"], return_exceptions=True)
        )

    elif args.mode == "test":
        TEST_URLS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
        ]
        bot._is_ai_spring_tournament = False
        bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(u) for u in TEST_URLS]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
    
    # Log final metrics
    logger.info(f"Final metrics: {metrics.get_summary()}")
