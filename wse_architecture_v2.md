# WSE (World State Engine) — Architecture Proposal

## 1. Problem Statement

Large Language Models handle long context poorly. As context grows, two effects compound:

1. **Computational cost scales linearly with context length.** For a model with P active parameters, each token of context costs approximately 2P FLOP in attention computation. For a 22B-parameter model, that is ~44 GFLOP per token. For a 70B model, ~140 GFLOP. Searching for a single fact across 10⁸ tokens of context costs 4.4–14 ExaFLOP — regardless of whether the fact is found.

2. **Retrieval accuracy degrades with context length.** The "lost in the middle" phenomenon is well-documented: models systematically lose track of information positioned away from the beginning and end of context. This degradation is not a training failure — it is an architectural property of attention over long sequences.

The current industry response — expanding context windows to 1M–10M tokens — addresses this by brute force: paying linearly more compute for sublinearly degrading accuracy.

WSE proposes a structural alternative: **extract facts from the dialogue into an external structured store, and retrieve them via deterministic queries**. This reduces retrieval cost by 4–5 orders of magnitude (from GFLOP-per-token attention to MFLOP-range set operations) while making retrieval accuracy independent of session length.

### The Library Analogy

The problem is not unique to LLMs. A human who memorizes a large text verbatim faces the same limitation: they can recite it, but cannot reliably answer targeted factual queries about it. Asked a specific question, they will either scan linearly through their recall (slow, error-prone) or confabulate a plausible answer (fast, unreliable). This is the human equivalent of "lost in the middle."

No civilization solves this by training better memorizers. Instead, humanity built **writing → catalogs → indexed search**. A person with a search engine outperforms an eidetic memory champion on factual queries about texts the champion has read — because indexed retrieval is O(log n) or O(1), while linear scan is O(n) with degrading accuracy.

WSE applies this same principle to LLMs: **provide the model with a structured external index rather than expanding its internal context window**.

---

## 2. Architecture

WSE consists of three components. They do not communicate directly — all interaction passes through a shared external data store.

### 2.1 Extractor (SLM)

A specialized small language model (estimated minimum: 4–8B+ parameters), running **in parallel** with the main LLM. It receives copies of both user messages and main model responses as input.

The Extractor is not a conversational model. It is trained specifically for structured fact extraction from natural language. It uses language understanding capabilities for tasks that require interpretation, not just pattern matching:

- **Semantic normalization**: recognizing that "every other resident perished" denotes a 50% population decrease, not a literal enumeration
- **Entity resolution**: mapping variant spellings, transliterations, abbreviations, and aliases to a canonical identifier ("Isaac Newton" = "И. Ньютон" = "Исаак Ньютон")
- **Unit normalization**: converting between measurement systems and reconciling approximations ("about 30 years old" is consistent with "age: 31 years 2 months")
- **Plausibility checking**: flagging physically or logically impossible values ("a person weighing 76.2 grams") unless the domain ontology explicitly permits them
- **Fact vs. interpretation separation**: recording "population decreased by 50%" (fact) but not "the king was probably angry about it" (interpretation)

The Extractor writes to the Entity Store following a six-stage **ingestion pipeline**:

1. **Extraction** — identify facts, events, and projections in the incoming text
2. **Normalization** — convert to canonical forms (units, names, references)
3. **Deduplication** — determine whether this is a new fact or a re-mention of an existing one
4. **Validation** — check plausibility; check for contradictions against existing records
5. **Linking** — attach provenance references to source text; connect to existing entity records via relations
6. **Tagging** — assign properties and relation types according to the current ontology

The Extractor operates with a slight delay (~1–2 seconds behind the conversation flow). This does not affect system quality because the most recent 2–3 exchanges remain in the main model's KV-cache and do not require WSE retrieval.

**Important constraint**: when the Extractor detects a contradiction between incoming information and existing records in the Entity Store, it does not resolve the conflict autonomously. It flags the contradiction and surfaces it to the user (via the main LLM) as the ultimate arbiter.

The exact architecture, training methodology, and minimum viable size of the Extractor remain open research questions.

### 2.2 Entity Store

An external structured data store held in server RAM. Size: a few MiB per session; tens of thousands of concurrent sessions fit in a single server's memory.

**Each session maintains its own independent Entity Store. This is session-scoped working memory, not cross-session persistent memory.**

The Entity Store is organized as a **property graph**: nodes are entity records, edges are typed and directed relations between them.

#### Three categories of extractable information:

- **Events** — things that happened and changed the state of the world. Always bound to a point in time. ("the city was renamed", "the dosage was increased")
- **Properties** — static or slowly-changing characteristics. Not bound to a specific event. ("height: 182 cm", "jurisdiction: Delaware")
- **Projections** — unresolved states: plans, hypotheses, expectations, pending decisions. ("a merger is being considered", "three candidates are under review"). Projections may later be confirmed (→ become Events), retracted, or remain unresolved.

#### Five layers of an Entity Record:

**Layer 1 — Properties**: static or slowly-changing attributes of the entity.
```
entity: "E-1"
  attribute_a: "value"
  attribute_b: "value"
  attribute_c: "value" (updated at T⁴)
```

**Layer 2 — Temporal Layer (Delta Chains)**: changes over time, stored as a chain of operations rather than overwritten values. This follows the **event sourcing** pattern — the history of changes is itself the data.
```
attribute_x chain:
  T⁰: initial_value
  T¹: ×0.5 (event_A)        — provenance: [source_ref_1, source_ref_2]
  T²: +950 (event_B)        — provenance: [correction of earlier "+780", source_ref_3]
  T³: ×2 (event_C)          — provenance: [source_ref_4]
  T⁴: −200 (event_D)        — provenance: [source_ref_5]
```
Delta chains allow any temporal query to be answered from a single entity record: "what was the value after event A?", "when was the largest change?", "what caused the decrease?", "what is the current value?" — all derivable from the same chain.

**Critically**: the main LLM computes final values from delta chains — not the Extractor and not the Query Engine. Interpreting whether "doubled" means ×2 or +100%, resolving corrections ("I changed my mind, it was 950 not 780"), and handling ambiguous phrasing requires **language understanding**, not arithmetic. This is the main LLM's responsibility.

**Layer 3 — Relations**: typed, directed connections to other entity records, with their own temporal dimension.
```
relation_type_A: Entity_P (T⁰–T³) → Entity_Q (T⁴–present)
relation_type_B: [Entity_R, Entity_S]
```

**Layer 4 — Lifecycle Status**: current state of the record or individual facts within it:
- **active** — current, confirmed
- **superseded** — replaced by a newer version (old version preserved in the delta chain)
- **retracted** — explicitly cancelled by the user or author
- **speculative** — unresolved projection, plan, or hypothesis

**Layer 5 — Provenance Links**: each fact may carry **multiple** source references of different types:
- **user_statement** — direct assertion or decision by the user
- **in_context_document** — referenced within the working content (a quoted record, a cited source)
- **unverified_claim** — mentioned without confirmation (hearsay, rumor, unverified report)
- **model_output** — established by the main LLM during generation
- **correction** — explicitly overrides a previous value

When the Extractor encounters a re-mention of an existing fact from a new source, it does not discard it — it adds a new provenance link to the existing record. Multiple provenance links allow the main LLM to assess reliability: a fact supported by a user decision + a cited document + an independent mention is better established than one backed only by an unverified claim.

**A note on the same fact carrying different provenance types**: a single factual record (e.g., "attribute X decreased by 50%") may accumulate provenance links over the course of a session — first from a user statement, later from a document referenced in the narrative, later still from an indirect mention. These are not separate facts but separate **evidence paths** to the same record.

### 2.3 Query Engine (Deterministic)

A lightweight, non-ML program that retrieves data from the Entity Store into the main LLM's context. It performs deterministic operations: filtering by property values, traversing relations, computing set intersections, evaluating temporal predicates.

**Workflow:**

1. The main LLM formulates a **query plan** — a structured sequence of retrieval steps specifying entity types, properties, relations, filters, and the desired output format.
2. The Query Engine executes the plan deterministically — analogous to a game engine computing pathfinding or a database executing a query plan: no probabilistic reasoning, no semantic similarity matching.
3. The Query Engine returns a **Memory Pack** — a compact payload injected into the main LLM's context. The Memory Pack contains **only values and minimal identifiers** needed for the main LLM to process the result. Storage metadata, addressing structure, and retrieval tags are stripped.

**Context cost**: a fact stored internally with full metadata (entity ID, property name, value, provenance array, tags, temporal markers — potentially 100–300 bytes in JSON) enters the main LLM's context as just the value (2–4 tokens for a simple property) or value with minimal disambiguation (8–12 tokens when multiple entities are involved). Typical queries require 8–32 facts, consuming less than 512 tokens — a negligible fraction of even a modest context window.

**Computational cost**: set intersection, graph traversal, and filtering operations cost on the order of megaFLOPs. For comparison, generating a single token with a 22B-parameter model costs ~44 GFLOP. The entire query pipeline costs less than one generated token. This is a 4–5 order of magnitude reduction compared to attention-based search through a long context.

### 2.4 Ontology Management

The type system governing the Entity Store — entity types, property types, relation types, and their usage rules — is called the **ontology**. It is owned and managed exclusively by the main LLM.

- **Initialization**: at session start, the main LLM defines a base ontology appropriate to the domain.
- **Evolution**: during the session, the main LLM may introduce new types as needed, subject to a strict rule: **no new type may be created if an existing type already covers the same semantic space**. This prevents ontology bloat and tag fragmentation.
- **Storage**: the current ontology (types, definitions, and usage rules for the Extractor) occupies a dedicated section of the main LLM's context window, budgeted at ≤8,192 tokens — sufficient for hundreds of types with descriptions and rules.
- **Communication**: the Extractor follows the ontology rules when tagging and categorizing extracted facts. The Query Engine uses the ontology to validate and execute query plans.

---

## 3. Redefining Cold Retrieval

WSE fundamentally changes the role of stored conversation history (cold retrieval):

**Without WSE** (standard RAG): stored history is the *primary source of context*. Retrieval uses semantic similarity (embedding distance) to find relevant text chunks — a probabilistic process with inherent information loss. Retrieval accuracy depends on embedding quality and degrades with corpus size.

**With WSE**: stored history becomes an **archive of source texts** addressed by deterministic provenance links from entity records. Each fact in the Entity Store carries direct pointers to the text passages that established it. Retrieval accuracy is **100% by construction** — the system fetches the exact referenced passage, not the "most similar" one.

Semantic embedding search is not eliminated but demoted: it may serve as a fallback for exploratory queries where the user's question does not map to any indexed entity, but it is no longer the primary retrieval mechanism.

---

## 4. Context Window Strategy

WSE does not propose eliminating or minimizing the context window. It proposes maintaining a reasonably large window (128K–384K tokens) and allocating it efficiently:

| Section | Purpose | Typical budget |
|---|---|---|
| Input | Current user message | variable |
| Recent context | Last several exchanges in full | bulk of the window |
| WSE ontology + Memory Pack | Current type system + retrieved facts | ≤8,192 tokens |
| Reasoning / tool use | Chain-of-thought, tool calls | variable |
| Output | Model response | variable |

The design principle: **the context window holds what requires sequential attention (recent dialogue, reasoning chains); the Entity Store holds what requires indexed access (accumulated facts, history, relations)**. These are complementary, not competing, storage mechanisms.

128K tokens with WSE-backed retrieval is expected to outperform 10M tokens of brute-force context — at a fraction of the compute, with higher accuracy, and with accuracy that does not degrade as the session grows.

---

## 5. Accuracy Model

System accuracy = min(Extractor accuracy, Query formulation accuracy).

**Extractor accuracy** depends on model size, training data, and the ingestion pipeline design. This is a standard ML optimization problem with established methodology.

**Query formulation accuracy** depends on the main LLM's ability to decompose a natural-language question into a structured query plan. This capability is:
- Already well-developed in current LLMs (the same skill used for web search tool calls, Text-to-SQL, and multi-step API orchestration)
- **Easier in the WSE context** than in web search, because the main LLM has the complete ontology (schema) available in its context — it knows exactly what entity types, properties, and relations exist and can formulate precise queries against them, rather than guessing at unknown web content
- Improvable through standard tool-use fine-tuning and feedback loops (the Query Engine can return metadata about empty results, filter statistics, etc., enabling the main LLM to reformulate — the same pattern used in iterative web search)

**The critical property**: accuracy is **independent of session length**. Whether the Entity Store contains 100 or 100,000 facts, the deterministic Query Engine returns exact results for a correctly formulated query. This is the fundamental advantage over attention-based retrieval, where accuracy systematically degrades with context length.

---

## 6. Positioning Against Existing Approaches

### Standard RAG (Retrieval-Augmented Generation)
RAG retrieves text chunks by semantic similarity (embedding distance). WSE differs in three ways: (1) it stores **structured facts**, not text chunks; (2) retrieval is **deterministic** (exact addressing via provenance links), not probabilistic; (3) the retrieval unit is a compact fact (2–12 tokens), not a text passage (100–500 tokens). RAG remains useful as a fallback for exploratory queries; WSE replaces it for factual retrieval.

### GraphRAG (Microsoft Research, 2024)
GraphRAG builds a knowledge graph from source documents and uses it for retrieval. WSE shares the knowledge-graph principle but differs in architecture: (1) WSE uses a **parallel SLM extractor** operating in real time on the dialogue stream, not a batch preprocessing step; (2) the ontology is **owned and evolved by the main LLM** during the session, not fixed at indexing time; (3) WSE explicitly separates the Query Engine (deterministic) from the main LLM (interpretive), rather than using the LLM for graph traversal; (4) WSE introduces **event sourcing** (delta chains) for temporal reasoning, which GraphRAG does not address.

### MemGPT / LLM-based memory management
MemGPT uses the LLM itself to manage its own memory through explicit read/write operations. WSE differs fundamentally: (1) fact extraction is handled by a **dedicated SLM**, not by the main model, avoiding the overhead of self-management; (2) the data store is a **typed property graph**, not an unstructured text buffer; (3) retrieval is **deterministic**, not LLM-mediated. The separation of concerns (Extractor extracts, Engine queries, main LLM reasons) avoids the recursive cost of having the main model manage its own memory.

### Long-context scaling (1M–10M tokens)
The brute-force approach of expanding the context window. WSE's position: this trades compute cost (linearly scaling attention) for unreliable accuracy (lost-in-the-middle). WSE achieves higher accuracy at 4–5 orders of magnitude lower retrieval cost by moving from sequential attention to indexed access. The approaches are not mutually exclusive — WSE operates within a standard (128K–384K) context window and can coexist with moderately long context.

---

## 7. Application Examples

The following examples illustrate WSE's domain-agnostic nature. The architecture is identical across domains; only the ontology and entity types differ.

### 7.1 Creative Writing / Interactive Worldbuilding

**Domain ontology**: character, location, faction, item, skill, event, lore.

A user and an LLM co-create a fantasy world over a session spanning tens of thousands of exchanges. A city is founded, hit by plague, repopulated by royal decree, renamed after a conquest, and mentioned in dozens of scattered dialogues.

The entity record for the city accumulates a delta chain for population, temporal relations for political allegiance, provenance links from user decisions and in-narrative documents. When the user asks "what is the current population?" — the Query Engine retrieves the delta chain, and the main LLM reconstructs the value by interpreting each step.

Without WSE, this query requires the model to scan millions of tokens of creative text, locate all population-related mentions, identify which are corrections of earlier values, and compute the result — with high probability of error.

### 7.2 Medical AI (Clinical Decision Support)

**Domain ontology**: patient, symptom, diagnosis, medication, procedure, lab_result, provider, episode_of_care.

A clinical AI assists with a complex case over multiple consultations. The patient's record accumulates as an entity with property layers (demographics, allergies, chronic conditions) and delta chains (medication changes: drug A at dose X → increased to Y after lab result → switched to drug B due to adverse effect → dose adjusted).

Relations link the patient to providers, diagnoses to supporting evidence, medications to the conditions they treat. Provenance distinguishes subjective complaints (patient-reported), objective findings (lab/imaging), and clinical assessments (provider judgment).

When the clinician asks "has the patient ever been on an ACE inhibitor while also having elevated creatinine?" — the Query Engine intersects the medication history with the lab result timeline. The main LLM interprets the clinical significance.

### 7.3 Legal / Contract Management

**Domain ontology**: party, agreement, clause, obligation, amendment, dispute, ruling, jurisdiction.

A legal AI tracks a complex contract through its lifecycle. The agreement entity accumulates amendments as a delta chain (original terms → amendment 1 → amendment 2 → court ruling modifying clause 7). Relations link parties to obligations, disputes to relevant clauses, rulings to the precedents they cite.

Provenance distinguishes the original contract text, subsequent amendments, court rulings, and party communications. Lifecycle status marks clauses as active, superseded (replaced by amendment), or retracted (voided by ruling).

When the attorney asks "which obligations of Party A were modified after the 2024 ruling and are still active?" — the Query Engine filters obligations by party, applies the temporal predicate (modified after ruling date), checks lifecycle status (active only), and returns the result. The main LLM presents the findings with legal context.

---

## 8. Proposed Benchmark

To evaluate WSE effectiveness, we propose a three-level benchmark using long (1M+ token) synthetic sessions. The benchmark is domain-agnostic — test sessions may be generated for any domain (narrative, medical, legal, technical).

**Level 1 — Direct Retrieval**: single-fact lookup. One entity, one property. ("What is value X of entity Y?") Baseline: even brute-force long context handles this with reasonable accuracy.

**Level 2 — Relational Retrieval**: facts connected through entity relations. ("What is property A of entity B, where B is related to entity C by relation R?") Requires traversing two or more entity records. Long-context accuracy begins to degrade here.

**Level 3 — Temporal Chain Reconstruction**: the answer must be assembled from multiple facts scattered across the entire session, including corrections, renamings, and indirect references. The system must reconstruct a chain of events (initial value → modification₁ → modification₂ → correction → modification₃) distributed across millions of tokens of intervening content, where some modifications override earlier ones and entity identifiers may have changed.

**Evaluation metric**: structural, not binary. The primary measure is whether the system correctly reconstructs the **chain of events and deltas** leading to the answer. An arithmetic error in the final computation (e.g., multiplying instead of adding) is scored differently from a memory error (e.g., missing a correction event or confusing two entities). WSE is responsible for chain reconstruction; final computation is the main LLM's responsibility.

---

## 9. Design Principles

1. **Memory / Compute Separation.** The main LLM does not spend attention on fact storage and retrieval. Facts are externalized to a structured store; the model receives them on demand in compact form.

2. **Structured Storage over Raw Text.** Facts are stored as a typed property graph with explicit properties, relations, and temporal chains — enabling deterministic queries instead of probabilistic similarity search.

3. **Event Sourcing.** Changes are stored as delta operations, not overwritten snapshots. This preserves full history, enables any temporal query from a single record, and supports corrections without data loss.

4. **Separation of Concerns.** Each component does what it does best: the Extractor parses and normalizes (language understanding); the Entity Store holds structured data (storage); the Query Engine filters and traverses (deterministic computation); the main LLM reasons, interprets, and formulates queries (general intelligence).

5. **Deterministic Retrieval.** All data access operations are exact: set intersection, graph traversal, predicate evaluation. No probabilistic matching. Correct query → correct result, guaranteed.

6. **Provenance as a First-Class Citizen.** Every fact carries typed references to its sources. This enables reliability assessment, source verification, and conflict resolution.

7. **LLM-Owned Ontology.** The main LLM controls the type system and its evolution. The Extractor follows rules it receives, not rules it invents. This ensures semantic coherence as the schema grows.

8. **Compact Injection.** Only fact values and minimal identifiers enter the main LLM's context — not metadata, not addressing structure, not retrieval tags. This minimizes context window consumption.

9. **Scale-Independent Accuracy.** Retrieval accuracy does not degrade with the volume of stored facts, because queries address a structured index, not a linear sequence. 100 facts or 100,000 facts — same query, same accuracy.

10. **Asynchronous Extraction.** The Extractor runs in parallel with the main LLM, writing to the Entity Store with a slight delay. This avoids blocking response generation while maintaining memory currency.

---

## 10. Limitations, Failure Modes, and Edge Cases

### 10.1 Cold Start

At the beginning of a session, the Entity Store is empty and the ontology has just been initialized. During the first N exchanges, the Extractor is still populating the store. In this phase, WSE provides no advantage — the system operates as a standard LLM with a context window, because there are no stored facts to retrieve.

WSE's value emerges **only after sufficient facts have been extracted** to justify indexed retrieval over direct context. For short sessions (tens of exchanges, a few thousand tokens), the entire dialogue fits comfortably in the context window and WSE adds overhead without benefit.

The **break-even point** — where WSE begins to outperform pure context — depends on the domain:
- In fact-dense domains (medical case tracking, legal contract analysis), where many distinct entities and their properties are introduced quickly, the break-even may occur within 50–100 exchanges.
- In sparse or exploratory dialogues (brainstorming, philosophical discussion), where few extractable facts emerge per exchange, the break-even may require hundreds of exchanges — or may never arrive, because the dialogue does not produce a "world state" that needs tracking.

**Design implication**: WSE should be an **opt-in module**, not a mandatory component of every LLM session. It is most valuable for long, fact-dense, stateful interactions — and adds unnecessary complexity to short or unstructured ones.

### 10.2 Extractor Errors

The Extractor is a neural model and will make mistakes. The critical distinction is between error types:

**Missed extraction (false negative)**: the Extractor fails to capture a fact from the dialogue. This is a **silent failure** — neither the main LLM nor the user knows that something was lost. When the main LLM later queries for this fact, the Query Engine returns nothing, and the main LLM either falls back to its context window (if the fact is still there) or produces a response without it (if the context has moved on). This is the most dangerous failure mode because it is invisible.

**Incorrect extraction (corruption)**: the Extractor misinterprets a fact — wrong value, wrong entity assignment, wrong normalization. This produces a **confidently wrong result**: the Query Engine returns the corrupted fact, and the main LLM uses it with no reason to doubt it. This is worse than a missed extraction, because a missing fact may trigger uncertainty in the main LLM, while a wrong fact produces false confidence.

**False extraction (false positive)**: the Extractor creates a fact that was never stated — extracting a "fact" from an interpretation, a hypothetical, or a misread. Similar in effect to corruption: the system confidently returns something that should not exist.

**Mitigation strategies** (partially addressed in the current design, partially open):
- The two-pass ingestion (extract → validate) catches some errors before they enter the store.
- Provenance links allow the main LLM (or the user) to verify any fact against its source text — but only if verification is triggered.
- Contradiction detection catches cases where a new extraction conflicts with an existing record — but does not catch the first incorrect extraction (there is nothing to contradict yet).
- **Not yet addressed**: periodic consistency auditing (the Extractor or the main LLM re-checking stored facts against source text), confidence scoring on extractions, or user-facing "WSE thinks X — is this correct?" confirmation flows. These are potential mitigation layers, not yet part of the design.

### 10.3 Query Formulation Failures

The main LLM may formulate a query that is syntactically valid but semantically wrong — asking for the wrong entity type, applying the wrong filter, or missing a necessary traversal step. The Query Engine will faithfully execute the wrong plan and return wrong results.

**Partial mitigation** (already implied in the design): the Query Engine can return metadata alongside results — how many entities matched at each filter stage, which filters produced empty sets, which entity types were traversed. This allows the main LLM to detect anomalies ("zero results for a query that should have matches" → reformulate) — the same feedback loop used in iterative web search.

**Not yet addressed**: how to handle ambiguous queries where the main LLM cannot determine the correct decomposition without additional context from the user. Current assumption is that the main LLM will ask the user for clarification, but the protocol for this is not specified.

### 10.4 Ontology Drift

Over a long session, the ontology may accumulate types and relations that partially overlap, despite the "no duplicate types" rule. The main LLM may introduce a new type that is semantically adjacent to (but not identical with) an existing one, fragmenting facts across two categories. This is analogous to the problem of tag sprawl in any folksonomy.

The current design relies on the main LLM's judgment to prevent this, but provides no automated detection. A possible mitigation: the ontology section in the context window could include not just type definitions but a **similarity matrix** or explicit "this type is NOT the same as type Y" boundaries — but this has not been tested.

---

## 11. Open Research Questions

The following are not merely items requiring testing — they represent fundamental tensions in the design where competing requirements must be balanced.

### 11.1 Extractor Size and Architecture

The Extractor must be **smart enough** to perform semantic normalization, entity resolution, fact/interpretation separation, and plausibility checking. But it must be **light enough** that running it in parallel does not cost as much as a second main model — otherwise, the computational savings of WSE are negated.

This creates a hard trade-off:
- Entity resolution, normalization, and fact/interpretation separation are achievable at 3–4B parameters with task-specific fine-tuning.
- Plausibility checking ("a man weighing 76.2 grams is impossible") requires world knowledge, which correlates with model size. This may push the minimum to 8B+.
- But an 8B+ model running continuously in parallel on every token of dialogue is a significant computational commitment. For a main model of 70B+, this is ~10% overhead — acceptable. For a main model of 8B (edge deployment), it may double the cost.

**The central engineering risk**: if the minimum viable Extractor turns out to require 30B+ parameters, the WSE economic argument weakens substantially. Empirical testing across domains is needed to establish the actual minimum — and whether different domains require different Extractor sizes (medical may need more world knowledge than creative writing).

**Alternative worth exploring**: a tiered Extractor, where a small (1–3B) model handles extraction/normalization/deduplication, and a larger model (or the main LLM itself, on a separate pass) handles plausibility checking only for flagged cases. This would keep the continuous cost low while reserving heavy computation for edge cases.

### 11.2 Optimal Data Representation

The property-graph model described in this proposal must satisfy five requirements simultaneously:

1. **Unambiguous filtering** — the Query Engine must be able to select exact sets of facts without false positives.
2. **Relation preservation** — connections between entities must be first-class, traversable objects.
3. **Temporal reasoning** — delta chains and lifecycle states must be queryable by time.
4. **Extensible schema** — new property types and relation types must be addable mid-session without breaking existing queries.
5. **Compact representation** — storage overhead per fact must remain minimal (the Entity Store must fit in a few MiB per session).

These requirements **partially conflict**. Unambiguous filtering wants a rigid, strongly-typed schema. Extensibility wants flexibility. Compact representation wants minimal structure. Temporal reasoning wants more structure (timestamps, ordering, delta operators).

The property graph is one point in this trade-off space. Alternatives include:
- **Flat tagged records** with set-intersection queries: simpler, faster, but weaker on relation traversal.
- **Hypergraph representations**: richer expressiveness for n-ary relations, but higher complexity and storage cost.
- **Hybrid approaches**: flat tags for properties, explicit edges for relations, separate temporal index for delta chains.

The optimal representation likely depends on the domain and the distribution of query types. A/B testing across domains (creative writing with dense relations vs. medical with dense temporal chains vs. legal with deep hierarchical structures) is needed to identify which representation best balances these five axes — or whether a hybrid is necessary.

### 11.3 Ontology Evolution and Schema Migration

When the main LLM decides mid-session that a flat property must become hierarchical (e.g., `location: "city_name"` → `location: region → city → district`), all existing records tagged with the old schema must be migrated.

Current proposal: the main LLM defines the new schema; the Extractor re-tags affected records following the new rules. But this raises questions:
- How does the Extractor re-tag records whose source text is no longer in any context window? It would need to re-read the provenance-linked source passages — adding retrieval cost during migration.
- What happens if the migration is ambiguous? (A record tagged `location: "Springfield"` — which Springfield? Which region?) The Extractor may lack the context to resolve this.
- Can migration be incremental (re-tag on next access) or must it be atomic (all records updated before the new schema is usable)?

These are solvable problems, but the solutions affect system complexity and latency during migration events.

### 11.4 Query Plan Complexity Ceiling

Current LLMs demonstrate strong query formulation abilities through web search, Text-to-SQL, and API orchestration. WSE query plans leverage this existing capability, and the task is made easier by the availability of the full ontology (schema) in the main LLM's context.

However, deeply interconnected entity graphs may require query plans of a complexity not yet tested in standard tool-use scenarios — multi-step traversals with conditional branching, temporal predicates, and cross-entity aggregation. Establishing the practical ceiling — and whether domain-specific fine-tuning of the main LLM's query formulation ability is needed — requires empirical investigation with realistic Entity Store sizes and topologies.
