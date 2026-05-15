# Notes

Use this folder for **time-bound debugging trails**, **investigation writeups**,
and **audit notes**—see the **`write-notes`** skill (`.claude/skills/write-notes/`).

Prefer **one domain file** with multiple issue sections over a new file per
session; see [../Guides/content/notes-consolidation-guide.md](../Guides/content/notes-consolidation-guide.md).

Long-running learnings also appear under `README/learnings.md` and other
top-level `README/*.md` files; link from notes instead of duplicating.

## Index

- [Core ML compute-unit ablation](coreml-compute-unit-ablation.md) - F/G/G-prime/G-double-prime benchmark logic for isolating `.all`, ANE, GPU, and CPU behavior in the Swift pipeline.
- [ANE decoder-har-post investigation](ane-decoder-har-post-investigation.md) - Root-cause analysis of why `kokoro_decoder_har_post_10s.mlpackage` engages 0 of ~1238 ops on the Neural Engine; identifies rank-3 `(B,C,T)` body as the disqualifier and scopes a rank-4 rewrite as the fix.
