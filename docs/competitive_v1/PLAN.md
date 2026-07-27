# Competitive Foundation v1 continuation plan

Read these files in order before changing code:

1. `../../COMPETITIVE_FOUNDATION_V1_MISSION.md` — immutable scope and evidence rules.
2. `../../COMPETITIVE_FOUNDATION_V1_PLAN.md` — original architecture/gap plan.
3. `NEXT_SESSION.md` — verified current state and the single next action.
4. `05_TEAM_HANDOFF.md` — ownership and merge contracts for five people.
5. `07_KNOWN_GAPS_AND_RISKS.md` — claims that must remain blocked/partial.

The implementation sequence from here is data-driven:

```text
official BTC release
  -> read-only six-hour audit and hashes
  -> freeze legal query/qrels slice
  -> run incumbent visual baseline unchanged
  -> connect one real modality at a time
  -> paired benchmark and failure slices
  -> retain only measured gains
  -> official-schema dry run
```

Do not start learned fusion, VLM reranking, KISC automation, or broad model
shopping before the frozen incumbent has been measured on the official data.
Keep `mock` and `exact-numpy` as mandatory offline fallbacks.
