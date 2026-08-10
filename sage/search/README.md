# `sage/search`

A production search over real strain: score the data, build a background, assign
significance and an astrophysical probability, measure sensitivity, compare the result
against published catalogues, and characterise anything new.

Scope: binary black holes, two detectors, one observing run per campaign.

## Layout

| Area | What it does |
|---|---|
| `spec` `geometry` `stages` | Configuration, time and index conventions, the stage graph |
| `segments` `grid` `slides` | Interval algebra, the window lattice, the time-slide ladder |
| `checkpoint` `reader` `features` `engine` `decode` | Loading the network and scoring the stream |
| `triggers` `cluster` | Trigger storage and reduction to independent events |
| `background` `tail` `far` `calibration` | Background, extrapolation, false-alarm rates, validity checks |
| `injection/` `sensitivity/` | Injection campaign and sensitive volume-time |
| `pastro/` | Rate inference and per-candidate astrophysical probability |
| `catalogue/` `crossmatch` `ood` | Published catalogues and comparison against them |
| `characterize/` | Data quality, spectrograms, follow-up filtering, parameter estimation, localisation |
| `candidates` `store` | The candidate table and the queryable campaign database |
| `figdata/` `figures` `release/` | Figure inputs, rendering and the data release |

Drawing code lives in `sage/plotting/search/`; it reads figure data products and computes
nothing.

## Running a search

Once a network is trained, one call searches an observing run end to end:

```python
from sage.search import run_search

result = run_search(
    checkpoint="/work/nagarajan/sage_runs/o4a/CHECKPOINTS/best.pt",
    config_module="runs.o4a.config_HL",
    observing_run="O4a",
)
print(result.summary())
```

or from the shell:

```bash
cd runs/search
./submit.sh plan   config_o4a_HL     # steps and projected cost, submits nothing
./submit.sh smoke  config_o4a_HL     # same sequence, shallow background
./submit.sh search config_o4a_HL     # the full campaign
```

That runs, in order: read and validate the network, build segments and the window
lattice, score the run, cluster, build the time-slide background, assign false-alarm
rates and validate the background, run the injection campaign and measure sensitivity,
infer rates and assign probabilities, assemble the candidate table, compare against
published catalogues, then build and render the figures and tables.

Every step is resumable, so the same call starts, resumes and extends a campaign.
Individual stages can still be run on their own (`./submit.sh background config_o4a_HL`)
for staged or repeat running.

### Per-event work is separate

Data-quality vetting, spectrograms, follow-up filtering, parameter estimation and
localisation are **not** part of the search. They are per-event, need the
parameter-estimation environment, and are normally applied to a chosen few candidates:

```bash
./submit.sh characterize config_o4a_HL --tier 1
./submit.sh characterize config_o4a_HL --event SGW230814_230901 --level full --pe
```

Results are written back into the same campaign store, and tiers are re-derived so that
anything the vetting rejects is demoted. Tiers assigned by the search itself are marked
provisional, since they rest on significance and probability alone; a provisional tier is
an upper bound on the vetted one.

## Asking questions

Every stage writes into one database per campaign, so any recorded quantity is
queryable and quantities from different stages can be combined in a single condition.

```bash
# everything known about one candidate
./submit.sh query config_o4a_HL --event SGW230814_230901

# any condition over any recorded quantity
./submit.sh query config_o4a_HL --where "pastro > 0.9 AND dq_vetoed = 0 AND ifar_yr > 100"

# what is available to query
./submit.sh query config_o4a_HL --describe
```

From Python:

```python
from sage.search.store import open_store

store = open_store(spec)
store.event("SGW230814_230901")                  # one candidate, everything
store.select(where="pastro > 0.5", order_by="ifar_yr DESC")
store.comparison_matrix()                        # candidates against catalogues
store.export("candidates.tex", fmt="latex", where="tier >= 1")
```

Bulk arrays such as spectrograms and posterior samples stay in their own files; the
database records where they are, so a candidate's record still resolves to everything
about it.

## Method references

Every method cites an equation in a document under `docs/references/`, listed in
[`references.py`](references.py) and indexed in
[`docs/references/README.md`](../../docs/references/README.md). Fetch them with
`python docs/references/fetch.py`.

The two central results: the rate posterior is Eq. (21) of arXiv:1302.5341, and the
per-candidate probability is Eq. (11) of arXiv:2305.00071.

## Conventions worth knowing before reading the code

The strain release is a concatenation of overlapping chunks. Sample indices are
contiguous across the whole file, but the chunks are not ordered by time, consecutive
ones overlap by about 15.6 s, and a time inside an overlap appears in two chunks with
different sample values because each was conditioned on its own boundaries. Time is
therefore a per-chunk coordinate, reads never cross a chunk boundary, and each instant is
assigned to exactly one owning chunk so nothing is analysed twice.

Background livetime is always measured from the slide plan, never inferred from the
number of slides. Triggers are clustered before they are counted, in both the foreground
and the background. Both mixture densities share one threshold and one support. The
ranking statistic is checked for ordering evidence before any probability is assigned.
