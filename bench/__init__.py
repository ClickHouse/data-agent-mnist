"""
Candidate runners and blind judge for the synthetic text2sql benchmark.

ch_query and system_prompt are passed as arguments to keep DB/schema concerns in
the pipeline stages and infrastructure concerns here. The package is the API:
callers `import bench` and use the names below.

  clients      auth helpers and one client per provider endpoint
  concurrency  retry with backoff, thread-pool map
  usage        TokenUsage, the per-run token accumulator
  runners      tool schemas, the agentic loops, run_candidate
  librechat    the LibreChat product-surface runner
  completion   one-shot completion routed to a judge's provider
  scoring      result-set comparison, column linker, majority-vote ground truth
  judge        judge prompt, judge_score, judge_panel
"""
from registry import (  # noqa: F401
    ADAPTIVE_THINKING_ONLY, ALL_CANDIDATES, ANNOTATORS, ANTHROPIC_CANDIDATES,
    BEDROCK_REASONING, CANDIDATES, EFFORT_CAPABLE, ENDPOINTS, FIREWORKS_CANDIDATES,
    GATEWAY_CANDIDATES, GEMINI_CANDIDATES, GEMINI_GLOBAL, JUDGE_MODEL,
    LIBRECHAT_CANDIDATES, LINKER,
    MODELS,
    JUDGE_MODEL_IDS, JUDGE_PROVIDER, JUDGE_SEATS, MANTLE_CANDIDATES,
    MANTLE_RESPONSES_CANDIDATES, OPENAI_CANDIDATES, OPENAI_RESPONSES_ONLY,
    RETIRED_CANDIDATES,
    SCORE_ABS_TOL, SCORE_REL_TOL, SCORE_ROUND_DECIMALS,
)

from .clients import (  # noqa: F401
    AWS_REGION, _AWSv4Auth, _GCPAuth,
    anthropic_client, anthropic_native, bedrock, fireworks_client, gateway_client,
    gemini_client, gemini_global_client, mantle_client, mantle_openai_client,
    openai_client,
)
from .concurrency import map_concurrent, retry  # noqa: F401
from .usage import TokenUsage  # noqa: F401
from .runners import (  # noqa: F401
    ERR_MAX_OUTPUT_TOKENS, MAX_TURNS,
    TOOLS_BEDROCK, TOOLS_MESSAGES, TOOLS_OPENAI, TOOLS_RESPONSES,
    emits_inline_reasoning, run_candidate, run_candidate_bedrock,
    run_candidate_messages_api, run_candidate_openai_compat, run_candidate_responses_api,
)
from .librechat import (  # noqa: F401
    ERR_LIBRECHAT_NO_TRACE, _LangfuseReader, _LibreChatSession, _parse_sse_event,
    _reconstruct_librechat_run, run_candidate_librechat,
)
from .completion import _judge_complete, judge_token_budget  # noqa: F401
from .scoring import (  # noqa: F401
    AGREEMENT_TOL, SCORING, ComparisonPolicy, _COL_LINK_CACHE, _link_columns,
    _parse_result, _results_match, annotators_agree, bt_sigma_aggregate,
    is_exploratory, majority_vote_gt,
)
from .judge import (  # noqa: F401
    _has_results, judge_panel, judge_score, select_panel,
)

# Shuffle seed for the eval loop's question order.
EVAL_SEED = 42
