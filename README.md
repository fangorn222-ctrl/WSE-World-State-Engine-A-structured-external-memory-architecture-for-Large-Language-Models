# WSE (World State Engine)

**A structured external memory architecture for Large Language Models**

WSE replaces brute-force long-context scaling with a three-component system: a parallel SLM extracts facts from the dialogue stream into a structured property graph, a deterministic query engine retrieves them by exact addressing, and the main LLM formulates queries and interprets results. Retrieval cost drops by 4–5 orders of magnitude compared to attention-based search; accuracy becomes independent of session length.

## The Problem

Expanding LLM context windows to millions of tokens is computationally expensive and unreliable. Attention cost scales linearly with context length (processing each additional token through a dense-equivalent model incurs roughly 2P FLOP in parameter-matrix compute alone, before sequence-length-dependent attention overhead), while retrieval accuracy degrades due to the well-documented "lost in the middle" phenomenon. Paying more for worse results is not a scalable solution.

## The Idea

Instead of forcing the main LLM to search through raw text, WSE externalizes facts into an indexed store and retrieves them deterministically — the way libraries and search engines solved the same problem for human memory.

Three components, strict separation of concerns:

| Component | Role | Nature |
|---|---|---|
| **Extractor** (SLM) | Parses dialogue, extracts and normalizes facts, writes to store | Small language model (~4–8B+), runs in parallel |
| **Entity Store** | Holds structured facts as a property graph with delta chains, relations, and provenance links | Data structure in RAM, session-scoped |
| **Query Engine** | Retrieves facts into main LLM's context by executing deterministic query plans | Lightweight non-ML program |

The main LLM owns the ontology (type system), formulates structured queries, and interprets results. It does not store facts and does not search — it thinks.

## Key Properties

- **Event sourcing**: changes stored as delta chains, not overwritten snapshots — full history, temporal queries, correction support
- **Deterministic retrieval**: set intersection and graph traversal, not probabilistic similarity search — correct query guarantees correct result
- **Provenance tracking**: every fact carries typed source references — reliability assessment by construction
- **Compact injection**: only bare values enter the main LLM's context (2–12 tokens per fact) — minimal context window consumption
- **Scale-independent accuracy**: 100 facts or 100,000 facts — same query, same accuracy
- **Domain-agnostic**: identical architecture for creative writing, medical AI, legal analysis, or any domain requiring long-session fact tracking

## Read the Full Proposal

→ **[WSE Architecture Proposal](wse_architecture_v2.md)**

The document covers: problem statement, detailed architecture, data model (five layers of an entity record), query pipeline, context window strategy, redefinition of cold retrieval, accuracy model, positioning against existing approaches (RAG, GraphRAG, MemGPT, long-context scaling), application examples (worldbuilding, medical, legal), proposed benchmark, design principles, limitations and failure modes, and open research questions.

## Status

This is an architectural proposal at the concept stage. No implementation exists. The document describes the design, its rationale, and identifies open research questions — particularly around minimum extractor size, optimal graph structure, and ontology evolution. Contributions, critique, and discussion are welcome.

## Open Research Questions

1. **Extractor size vs. capability** — minimum viable model size that balances extraction quality against computational cost
2. **Optimal data representation** — property graph vs. flat tags vs. hypergraph vs. hybrid; trade-offs between filtering precision, extensibility, and compactness
3. **Ontology evolution** — schema migration when the type system changes mid-session
4. **Query plan complexity ceiling** — practical limits of LLM-formulated query plans for deeply interconnected graphs

## Related Work

- **RAG** (Retrieval-Augmented Generation) — WSE replaces probabilistic chunk retrieval with deterministic fact addressing
- **GraphRAG** (Microsoft Research, 2024) — WSE shares the knowledge-graph principle but uses real-time SLM extraction, LLM-owned evolving ontology, and event sourcing for temporal reasoning
- **MemGPT** — WSE separates fact extraction (dedicated SLM) from reasoning (main LLM), avoiding the overhead of self-managed memory

## Development and Attribution

WSE was conceived and developed by **Alexey Ketslakh** ([fangorn222@gmail.com](link)) through a series of collaborative sessions with AI systems used as intellectual sparring partners:

- **Core concept** — Alexey: the principle of SLM-based parallel fact extraction into a structured external store with deterministic retrieval; entity-centric data model with temporal layers; LLM-owned ontology; the reframing of cold retrieval from primary context source to provenance-addressed archive; event sourcing for delta chains; provenance as a first-class citizen; the computational cost argument (attention-based search vs. indexed access)
- **ChatGPT** (OpenAI, v5.3) — contributed the two-pass extractor architecture (extract → verify) and the multi-stage ingestion pipeline design (novelty checking, plausibility validation, contradiction detection layers)
- **Claude** (Anthropic, Opus 4.6) — contributed the insight that fact retrieval into the main model's context should be performed by a deterministic query engine rather than an LLM; terminology refinement (entity record, delta chains, property graph, ingestion pipeline, projections as a category); structuring and co-authoring the architecture document

The architecture, design decisions, and all domain-specific applications are Alexey's work. AI contributions were in the role of peer reviewers and co-developers on specific subsystems. 
