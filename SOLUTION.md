# Lead Generation Solution

## 1. Objective
This solution identifies high-value lead candidates from a graph built from Excel node/edge sheets, guided by topology rules in topology.md.

Key goals:
- High recall: avoid missing true candidates.
- High precision: reduce noisy candidates in top ranks.
- Explainability: each final lead includes a reason tied to rules and graph evidence.

## 2. High-Level Architecture
The pipeline in main.py is a hybrid GraphRAG retrieval + reranking + LLM scoring flow:

1. Data load and graph build
- Read Node and Edge sheets from COI_Template.xlsx.
- Normalize strings (trim spaces and remove invisible unicode characters) to avoid entity mismatch.
- Build a directed graph with node attributes and edge metadata.

2. Representation learning (embeddings)
- Node-context embedding:
  - Node attributes
  - Outgoing edge types
  - Incoming edge types
  - Neighbor sample
- Edge-context embedding:
  - Source node summary
  - Edge type
  - Target node summary

3. Hybrid retrieval
- Deterministic rule evidence:
  - Match likely edge types from topology text.
  - Add source nodes connected by matched edge types with deterministic bonus.
- Semantic retrieval:
  - Run topology query and goal query against edge embeddings.
  - Run topology query and goal query against node embeddings.
- Score fusion:
  - Combine node and edge channel scores with configurable weights.
  - Add bonus for matched rule edge types.
  - Keep top candidate pool for downstream ranking.

4. Reranking (optional but enabled by default)
- Use BAAI/bge-reranker-v2-m3 on top-N retrieved candidates.
- Blend reranker score with retrieval score to refine order.

5. LLM scoring and explanation
- Batch candidates to avoid context length issues.
- Ask LLM to output strict JSON with score + reason.
- Filter by score threshold.
- Deduplicate by node and keep best score.

6. Optional topology update
- Can append newly discovered rules if auto_write_rules is enabled.
- Recommended default for safety: disabled.

## 3. Default Models
Current defaults in main.py:
- Embedding model: BAAI/bge-m3
- Reranker model: BAAI/bge-reranker-v2-m3
- LLM model: deepseek-chat

Safety fallbacks:
- If embedding model load fails, fallback to all-MiniLM-L6-v2.
- If reranker model load fails, reranker is disabled and retrieval still runs.

## 4. Main Parameters
Core quality/performance controls:

- retrieval_k
  - Top-K per semantic search channel.
  - Higher value increases recall, may reduce speed/precision.

- candidate_pool_size
  - Number of candidates kept before LLM scoring.
  - Larger pool increases recall and cost.

- topology_query_weight / goal_query_weight
  - Balance between strict topology intent and business goal intent.

- edge_score_weight / node_score_weight
  - Balance between edge channel and node channel in retrieval fusion.

- rule_type_bonus
  - Extra score for candidates linked to edge types matching topology rules.

- deterministic_match_bonus
  - Extra deterministic boost for rule-consistent graph matches.

- rerank_top_n / reranker_weight
  - Controls second-stage reranking strength.

- score_threshold
  - Final cutoff on LLM score.

- llm_batch_size
  - Controls request size to avoid context overflow.

## 5. Why This Works Better
Compared with naive node-only semantic retrieval:
- It uses structural graph context, not just flat node text.
- It retrieves from both node and edge spaces.
- It injects deterministic topology evidence before semantic ranking.
- It adds reranker-based precision refinement before expensive LLM scoring.

This combination generally improves both recall and precision for topology-driven lead discovery.

## 6. Run Instructions
1. Ensure environment variables are available:
- DEEPSEEK_API_KEY in .secrets/deepseek.env or environment.

2. Ensure dependencies are installed from requirements.txt.

3. Run:
- python main.py

4. Output:
- Top high-priority targets with score and reason.

## 7. Validation Scenario
For topology focused on sold business behavior, expected names include:
- Addison Rassell
- Basil Lee
- Christopher Fairmont
- Fredrick Chang

Validation command pattern:
- Instantiate GraphRAGLeadGenerator with topology.md and target goal.
- Call generate_targets().
- Check expected names are present in returned list and preferably top ranks.

## 8. Recommended Tuning Strategy
If recall is too low:
- Increase retrieval_k.
- Increase candidate_pool_size.
- Increase topology_query_weight.
- Increase deterministic_match_bonus.

If precision is too low:
- Increase reranker_weight.
- Increase score_threshold.
- Reduce candidate_pool_size.
- Increase edge_score_weight when rules are edge-type-centric.

If speed/cost is too high:
- Reduce candidate_pool_size.
- Reduce llm_batch_size only if needed for stability.
- Disable reranker for quick iteration.

## 9. Operational Notes
- Keep auto_write_rules disabled by default in production-like runs.
- Use debug=True during tuning to inspect matched edge types, candidate counts, and rerank effects.
- Persist benchmark metrics (Recall@K, Precision@K, MRR) when evaluating model/weight changes.

## 10. Current File of Record
The complete implementation is in:
- main.py
