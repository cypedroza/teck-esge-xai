# Instructions for AI assistants working on this repository

These rules exist because earlier AI-assisted versions of this code shipped a
simulated market benchmark and an event-dating bug that reached a published
article. Follow them without exception.

## Data integrity
- Never generate, simulate, impute or "placeholder" data in any analysis path.
  If real data cannot be obtained, raise an error and stop. No `np.random` in
  data-producing code; randomness is allowed only in methods that are random by
  design (Monte Carlo, bootstrap, model seeds) and must use
  `np.random.default_rng(config.RANDOM_STATE)`.
- Every external series (prices, indices) is downloaded from its real source,
  cached in `data/raw/`, and its source and ticker are stated in the output.
- Locate years and dates by label (`Series.loc[year]`), never by array position.

## Verification before claiming anything is "correct" or "consistent"
- Checking that code *structure* matches the paper is not verification.
  Recompute at least one headline number from the code's outputs and compare it
  with an independent external source (e.g. published index returns).
- A number that appears in the paper must be reproducible by a script in this
  repository from files in `data/`. If it is not, say so explicitly.
- Run `pytest -q` and `python run_pipeline.py` before reporting work as done.

## Honesty about methods
- Describe what the code does, not what it was intended to do. If the paper's
  methods section and the code disagree, flag it; do not paper over it.
- Keep `README.md` limited to files that exist.
- The Mount Polley tailings breach (2014) was an Imperial Metals operation, not a Teck asset.

## Workflow
- Work on a branch and open a pull request; never push to `main` directly.
- Do not publish, release or announce results without the author's review.
