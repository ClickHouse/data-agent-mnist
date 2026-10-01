"""Per-run token accounting, normalised across providers."""

class TokenUsage:
    """Per-run token accumulator, summed across the agentic turns.

    Usage can only be captured at call time, so a run without it can never be
    costed retroactively: re-tokenizing the transcript structurally UNDER-counts
    reasoning models, because hidden thinking is billed as output and never
    appears in the transcript. That is exactly the set of models whose
    cost we most need.

    Providers name these differently AND disagree on what nests inside what, so
    the accumulator normalises to one schema in which prompt / cache-read /
    cache-write are DISJOINT. Every `add_*` is defensive: a provider that omits
    `usage`, or a gateway that drops the details sub-objects, must degrade to
    zeros rather than raise. Telemetry is never allowed to kill a run that has
    already spent real money on tool calls.
    """

    __slots__ = ("prompt", "completion", "reasoning", "cache_read", "cache_write",
                 "calls", "missing", "per_turn")

    def __init__(self):
        self.prompt = self.completion = self.reasoning = 0
        self.cache_read = self.cache_write = 0
        self.calls = self.missing = 0
        # One entry per turn, which the sums cannot reconstruct in either direction.
        #
        # A sum answers "what did the run cost". It cannot answer how close a turn
        # came to the per-call output cap, and that cap is what truncates a run
        # (ERR_MAX_OUTPUT_TOKENS) and therefore scores it as a failure. From
        # completion_tokens / api_calls you get the mean turn and never the largest,
        # and the gap between them is unbounded: measured on the board, that
        # inference leaves the peak somewhere in an interval 23 to 91 points wide.
        #
        # Storing the vector rather than a single max because the statistic worth
        # having is not decided yet. Floor, median, p90 and peak all fall out of it,
        # and so does input growth per turn, which is what the prompt-caching work
        # needed and could not see: it could measure that 42.9% of input was served
        # from cache overall, but not which turn the cache started hitting.
        #
        # Cheap: about 10 to 60 entries per run, a few hundred KB across a full
        # board against a results file already in the tens of MB.
        self.per_turn = []

    @staticmethod
    def _int(obj, *names):
        """First present, integer-valued attribute/key among `names`, else 0."""
        for n in names:
            v = getattr(obj, n, None)
            if v is None and isinstance(obj, dict):
                v = obj.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    @staticmethod
    def _sub(obj, name):
        v = getattr(obj, name, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(name)
        return v

    def _record(self, u, prompt, completion, reasoning=0, c_read=0, c_write=0):
        self.calls += 1
        # `u is None` is not the only miss: a provider can return an empty or
        # unrecognised usage object, and adding zeros from it would look like an
        # exact zero rather than a gap. Any real call bills input, so extracting
        # nothing on both axes means we failed to read it.
        if u is None or (prompt == 0 and completion == 0):
            self.missing += 1
            return
        self.prompt += prompt
        self.completion += completion
        self.reasoning += reasoning
        self.cache_read += c_read
        self.cache_write += c_write
        # Recorded raw, in the provider's own accounting, with nothing derived.
        # `reasoning` is NOT added to `completion`: every provider that reports it
        # counts it inside its completion figure, so adding them would double-count
        # exactly the models the field exists to illuminate, and would overstate
        # how close a turn came to the output cap. Kept as its own element so a
        # reader can still see the split per turn.
        self.per_turn.append([prompt, completion, reasoning, c_read, c_write])

    def add_bedrock(self, resp):
        """Converse: cache figures sit beside inputTokens, not inside it."""
        try:
            u = (resp or {}).get("usage")
            self._record(u,
                         self._int(u, "inputTokens"), self._int(u, "outputTokens"),
                         0,
                         self._int(u, "cacheReadInputTokens"),
                         self._int(u, "cacheWriteInputTokens"))
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_openai(self, resp):
        """chat.completions: prompt_tokens is INCLUSIVE of cached_tokens.

        Unlike the Anthropic/Bedrock convention, where cache reads are reported
        alongside a cache-free input count. Subtract so `self.prompt` means the
        same thing on every path and the total does not double-count.
        """
        try:
            u = getattr(resp, "usage", None)
            cd = self._sub(u, "completion_tokens_details")
            pd = self._sub(u, "prompt_tokens_details")
            cached = self._int(pd, "cached_tokens")
            self._record(u,
                         max(self._int(u, "prompt_tokens", "input_tokens") - cached, 0),
                         self._int(u, "completion_tokens", "output_tokens"),
                         self._int(cd, "reasoning_tokens"),
                         cached)
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_responses(self, resp):
        """Responses API: same inclusive-input convention as chat.completions.

        Both `cached_tokens` and `cache_write_tokens` live under
        `input_tokens_details`, i.e. they are components OF `input_tokens`, so
        BOTH come out of the residual prompt figure or the buckets stop being
        disjoint and the total double-counts writes.
        """
        try:
            u = getattr(resp, "usage", None)
            od = self._sub(u, "output_tokens_details")
            idt = self._sub(u, "input_tokens_details")
            cached = self._int(idt, "cached_tokens")
            written = self._int(idt, "cache_write_tokens")
            self._record(u,
                         max(self._int(u, "input_tokens") - cached - written, 0),
                         self._int(u, "output_tokens"),
                         self._int(od, "reasoning_tokens"),
                         cached, written)
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_anthropic(self, resp):
        """Messages API: thinking is billed inside output_tokens, not broken out."""
        try:
            u = getattr(resp, "usage", None)
            self._record(u,
                         self._int(u, "input_tokens"), self._int(u, "output_tokens"),
                         0,
                         self._int(u, "cache_read_input_tokens"),
                         self._int(u, "cache_creation_input_tokens"))
        except Exception:
            self.calls += 1
            self.missing += 1

    def add_langfuse(self, usage_details, *, prompt_is_inclusive: bool = False):
        """One Langfuse observation's usageDetails -> one turn.

        The LibreChat runner does not hold a provider response; it reads usage back
        off the agent's trace, in whatever keys LibreChat's integration wrote. So
        the disjointness the other add_* methods inherit from a provider's own
        accounting has to be re-established here from key names, and anything not
        recognised is left at zero rather than guessed.

        `input`'s convention is the one thing a dict cannot settle: OpenAI-style
        counts cache reads inside it, Anthropic-style reports them beside it. We do
        not guess. Pass prompt_is_inclusive=True once a real trace shows the inclusive
        form, and the cache figures come out of the prompt so the buckets stay
        disjoint; left False, prompt is recorded as given and a nonzero cache_read is
        the signal that the assumption still needs checking against a trace.
        """
        ud = usage_details or {}
        prompt     = self._int(ud, "input", "prompt_tokens", "input_tokens")
        completion = self._int(ud, "output", "completion_tokens", "output_tokens")
        c_read = self._int(ud, "cache_read_input_tokens", "cacheReadInputTokens", "cached_tokens")
        c_write = self._int(ud, "cache_creation_input_tokens", "cacheWriteInputTokens",
                            "cache_write_tokens")
        if prompt_is_inclusive:
            prompt = max(prompt - c_read - c_write, 0)
        self._record(ud or None, prompt, completion, 0, c_read, c_write)

    def as_dict(self) -> dict:
        # The three input components are disjoint by construction (see
        # add_openai / add_responses), so they are safe to sum. Cache reads and
        # writes are both billed, reads at a discount and writes at a premium, so
        # a "leanness" total that drops them flatters cache-heavy providers.
        #
        # `reasoning_tokens` is deliberately NOT in the total: every provider that
        # reports it counts it inside its completion figure, so adding it again
        # would double-count exactly the models it is meant to illuminate. It is a
        # breakdown of completion_tokens, not a fifth bucket.
        return {
            "prompt_tokens": self.prompt, "completion_tokens": self.completion,
            "reasoning_tokens": self.reasoning,
            "cache_read_tokens": self.cache_read, "cache_write_tokens": self.cache_write,
            "total_tokens": (self.prompt + self.completion
                             + self.cache_read + self.cache_write),
            "api_calls": self.calls,
            # One [prompt, completion, reasoning, cache_read, cache_write] per turn
            # that reported usage, in call order. Every per-turn statistic derives
            # from this and none of them from the sums: peak output against the cap,
            # median turn, and input growth across the transcript. Turns that
            # reported no usage are absent, so len() can be below api_calls; that
            # gap is `calls_missing_usage`.
            "per_turn": self.per_turn,
            # >0 means some turns reported no usage, so the totals are a LOWER
            # bound and must not be presented as exact.
            "calls_missing_usage": self.missing,
        }
