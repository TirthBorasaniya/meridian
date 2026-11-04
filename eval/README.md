# Evaluation datasets

Datasets are committed once built and reviewed. Results computed from them are written only by
the scripts under `src/meridian/eval/` into `results/`.

| File | Set | Built by | Contents |
|---|---|---|---|
| `retrieval_set.jsonl` | R (R-full, with R-fast flagged by `in_fast`) | `meridian build-retrieval-set`, then `scripts/review_retrieval_set.py` | 150 questions with labeled spans |
| `retrieval_set_meta.json` | R | same | generation statistics and the review summary |
| `answer_set.jsonl` | A (A-fast flagged by `in_fast`) | migrated question set, references from `scripts/build_answer_references.py` | 75 questions; reference answers are drafted by the script and marked `reference_reviewed` after a manual check |
| `out_of_scope.jsonl` | O | written by hand | 25 questions outside the corpus topic |
| `cache_probes.jsonl` | E3 semantic cache probes | written by hand | paraphrase and near-miss question pairs |

## Retrieval set R

- **Sampling.** Chunks of the live collection are drawn stratified by paper and section:
  papers are visited round-robin in a seeded order, and each visit takes a chunk from the
  paper's least-used section. Chunks shorter than 120 tokens and sections such as front
  matter, acknowledgements, and checklists are not eligible
  (`configs/eval/retrieval_set.yaml`).
- **Questions.** The `question_writer` role (`openai/gpt-oss-120b`) writes one question per
  chunk.
- **Overlap filter.** Token overlap is the share of the question's content tokens (lowercase
  alphanumeric, stopwords removed) that also occur in the chunk. Questions above 0.5 are
  dropped; the writer gets one more attempt with a paraphrase instruction, and then the next
  chunk in draw order replaces the chunk.
- **Labels.** Each item records `arxiv_id`, `version`, and the character span
  `[start, end)` of its source chunk in the parsed paper (`data/parsed/{id}v{ver}.md`,
  parser version in `parser_version`). A retrieved chunk is relevant when it is from the same
  paper version and covers at least 50% of a labeled span (`src/meridian/eval/spans.py`), so
  labels remain valid when the chunker changes.
- **Manual review.** Thirty seeded random items are reviewed: each is marked valid or invalid,
  other relevant chunks among the top 10 are added as spans with `source: "review"`, and notes
  are stored in `review`. Items marked invalid are excluded from every metric, and E1 reports
  how many were excluded.
- **Splits.** R-fast is a seeded random subset of 40 items used by the daily gate; R-full is
  every valid item.
- **Limitation.** Synthetic questions favor lexical overlap with their source chunk. The
  overlap filter and the review reduce, but do not remove, this bias in favor of BM25.

## Exclusion list

Training data for the domain-adapted embedder never uses a paper referenced by R (any span,
including invalid items), A (`reference_arxiv_ids`), or O (`arxiv_ids`).
`meridian.eval.datasets.evaluation_paper_ids` builds the list, and training refuses to start if
any training pair comes from an excluded paper.
