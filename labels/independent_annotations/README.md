# Independent annotation tables

This supplement releases the two annotators' anonymized pre-adjudication label tables for **When Is an Agent Really Stuck? Trigger-Site Risk in Software-Engineering Agent Guards**. It supports reproduction of the step- and span-agreement statistics reported in the author response.

The statement in the submitted paper that independent annotation sheets were excluded describes the original artifact release. These tables are an additional release, separate from the final adjudicated labels used for detector evaluation. The original `LoopSense_Paper1_Artifact/` directory and ZIP retain their original scope. This supplement does not contain raw agent trajectories, tool outputs, annotation rationales, or adjudication working records.

## Files

- `coder_01/标注记录.xlsx`: annotator 1's label table.
- `coder_02/标注记录.xlsx`: annotator 2's label table.
- `reproduce_agreement.py`: recomputes counts, Cohen's kappa, span matching, and boundary agreement from the workbooks.
- `agreement_summary.json`: reference results from the released workbooks.
- `requirements.txt`: Python dependency for the reproduction script.
- `SHA256SUMS`: checksums of the supplement files, excluding the checksum file itself.

The workbooks are distributed byte-for-byte as supplied for this supplement. Checksums identify this release version; workbook metadata is not an annotation event log.

## Coverage

| Pool | Trajectories | Paired steps | Annotator 1 spans | Annotator 2 spans | Step kappa |
|---|---:|---:|---:|---:|---:|
| Natural500 | 500 | 8,550 | 1,286 | 1,263 | 0.737 |
| Enriched500 (`Final500` in the files) | 500 | 26,014 | 3,209 | 3,117 | 0.758 |
| StressFresh100 | 100 | 3,132 | 450 | 447 | 0.786 |
| Total | 1,100 | 37,696 | 4,945 | 4,827 | 0.762 |

All Enriched500 and StressFresh100 trajectories have span annotations. In Natural500, the same three trajectories have steps but no span from either annotator: `Natural500_T000166`, `Natural500_T000227`, and `Natural500_T000280`.

OpenHandsExternal330 follows a separate external-label protocol and is not included in these workbooks or the double-annotation reliability claim. Hard-negative (HN) is not a separately coded label in this task; the reported kappa does not establish HN reliability.

## Workbook schema

Each workbook has an instruction sheet (`说明`), three step sheets (`<pool>步骤`), and three span sheets (`<pool>片段`).

| Step column | Meaning |
|---|---|
| 数据集 | Dataset/pool name |
| 轨迹 | Public trajectory identifier |
| 步骤 | Zero-based step index |
| 标签 | `有进展` = progress; `停滞` = stagnant; `不确定` = uncertain |
| 置信度 | Annotator confidence, 1 to 5; blank means not recorded |

| Span column | Meaning |
|---|---|
| 数据集 | Dataset/pool name |
| 轨迹 | Public trajectory identifier |
| 片段序号 | Span sequence number within the trajectory |
| 起始步骤 | Inclusive zero-based start index |
| 结束步骤 | Inclusive zero-based end index |
| 标签 | `有效迭代` = productive iteration; `无效循环` = unproductive cycle |
| 置信度 | Annotator confidence, 1 to 5; blank means not recorded |

One step has one agreement label. Span intervals may overlap and do not partition the steps. Span labels and step labels are evaluated separately. These pre-adjudication records should not replace the final adjudicated labels when reproducing detector-risk metrics.

## Agreement calculation

Step rows are paired by pool, trajectory identifier, and step index. All 37,696 paired steps enter the calculation, including uncertain labels. There are 33,168 agreeing and 4,528 disagreeing pairs: raw agreement is 87.9881%, and Cohen's kappa is 0.762000306. One-versus-rest kappa is 0.764278275 for progress, 0.764666032 for stagnant, and 0.585767262 for uncertain.

Span matching is performed within each pool and trajectory. Intervals include both endpoints. Candidate pairs with interval intersection-over-union (IoU) at least 0.5 are sorted by decreasing IoU and greedily matched one-to-one. Ties use the original annotator 1 row order, followed by annotator 2 row order. Matching does not consult the span labels.

| Span statistic | Result |
|---|---:|
| Matched pairs | 4,298 |
| Coverage of annotator 1 spans | 86.9161% |
| Coverage of annotator 2 spans | 89.0408% |
| Label agreement among matched pairs | 94.9511% |
| Cohen's kappa among matched pairs | 0.884619676 |
| Matched pairs with identical endpoints | 3,224 (75.0116%) |
| Median absolute start deviation | 0 steps |
| Median absolute end deviation | 0 steps |

The span agreement and boundary-deviation statistics describe matched pairs. Unmatched spans are accounted for in the coverage fractions rather than silently included as agreeing pairs. Agreement measures consistency between the recorded labels; it does not independently validate the underlying trajectory semantics.

## Reproduce

Use Python 3.10 or newer. From this directory:

```sh
python -m pip install -r requirements.txt
python reproduce_agreement.py --check
```

The check reads both workbooks, validates their structure and step keys, recomputes all statistics, and compares them with `agreement_summary.json`, including the workbook hashes. It does not modify either workbook. To inspect newly computed results without using the reference JSON:

```sh
python reproduce_agreement.py
```

To save recomputed results to another file:

```sh
python reproduce_agreement.py --output recomputed_agreement.json
```
