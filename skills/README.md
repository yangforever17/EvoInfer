# Skills

The knowledge the optimization agents read before each episode: Markdown files with `name` / `description`
frontmatter, one directory per skill. `orchestrator.py` copies the chosen version into each agent's worktree
(`.evo/skills/` for Codex, `.claude/skills/` for Claude Code, `.dsh/skills/` for DeepSeek Harness). The copy is agent-local and never part of the
evaluated engine.

- `v0/`: the hand-written skills the paper's runs started from. `evo-workflow` is the entry point (the rules and the
  measure-then-change loop); the others are `gpu-profiling`, `cuda-graphs`, `moe-decode` and `speculative-decoding`.
- `CURRENT`: the default version (`--skills` or `$EVO_SKILLS` override it).

New versions come from `meta.py evolve --base v0 --new v1` (the meta-agent, see docs/architecture.md) or by hand.
Compare a new version against the old one with paired arms before making it the default (`experiments/e4_skills.sh`).
