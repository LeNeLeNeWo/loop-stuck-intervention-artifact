from __future__ import annotations
import argparse,csv,hashlib,re,shutil,textwrap,zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd
POOLS=['Natural500','Final500','StressFresh100','OpenHandsExternal330']
STATS={'Natural500':(500,8550,17.100,0.268,'SWE-agent / LoopBench-Agent','natural-distribution control'),'Final500':(500,26014,52.028,0.582,'SWE-agent / LoopBench-Agent','enriched difficult loop-heavy set'),'StressFresh100':(100,3132,31.320,0.820,'SWE-agent / LoopBench-Agent','clean stress challenge'),'OpenHandsExternal330':(330,51010,154.576,0.012,'OpenHands','external long-horizon pool')}
TEXT_EXT={'.md','.txt','.csv','.tsv','.json','.jsonl','.py','.yml','.yaml','.cff','.tex'}
DYNAMIC={'manifest_sha256.csv','artifact_integrity_report.md','reproducibility_audit_report.md','reproducibility_audit_results.csv','artifact_release_final_report.md'}
def root_from_script(): return Path(__file__).resolve().parents[2]
def clean_release(d):
    d=d.resolve(); base=d.parent.resolve(); base.mkdir(parents=True,exist_ok=True)
    if d.exists():
        b=base/'_backups'; b.mkdir(exist_ok=True); shutil.move(str(d),str(b/f'{d.name}_{datetime.now():%Y%m%d_%H%M%S}'))
    d.mkdir(parents=True,exist_ok=True)
def san(t:str)->str:
    t=re.sub(r"(?<![A-Za-z0-9_])D:[\\/][^\s,;`\"')]*","<LOCAL_PATH>",t,flags=re.I)
    t=re.sub(r"\\\\"+"wsl"+r"\.localhost[^\s,;`\"')]*","<LOCAL_PATH>",t,flags=re.I)
    t=re.sub(r"/home/[A-Za-z0-9_.-]+[^\s,;`\"')]*","<LOCAL_PATH>",t)
    return t
def wt(p,t): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(san(textwrap.dedent(t).strip()+'\n'),encoding='utf-8')
def cp(src,dst):
    if not src.exists(): return
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix.lower() in TEXT_EXT and src.stat().st_size<120_000_000:
        try: dst.write_text(san(src.read_text(encoding='utf-8',errors='ignore')),encoding='utf-8'); return
        except Exception: pass
    shutil.copy2(src,dst)
def sha_text(s): return hashlib.sha256(s.encode('utf-8',errors='ignore')).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def load(repo): return pd.read_csv(repo/'outputs/paper1_audit/normalized_steps.csv',dtype=str,keep_default_na=False),pd.read_csv(repo/'outputs/paper1_audit/normalized_spans.csv',dtype=str,keep_default_na=False)
def mapping(steps):
    m={}
    for ds in POOLS:
        tids=sorted(steps.loc[steps.dataset==ds,'trajectory_id'].astype(str).unique()); m[ds]={}
        for i,tid in enumerate(tids,1): m[ds][tid]=(f'{ds}_T{i:06d}',sha_text(f'{ds}|{tid}'))
    return m
def pubids(df,m):
    if 'trajectory_id' not in df.columns or 'dataset' not in df.columns: return df
    pubs=[]; hs=[]
    for ds,tid in zip(df.dataset.astype(str),df.trajectory_id.astype(str)):
        pub,h=m.get(ds,{}).get(tid,(f'{ds}_T_UNKNOWN',sha_text(f'{ds}|{tid}'))); pubs.append(pub); hs.append(h)
    ix=df.columns.get_loc('trajectory_id'); df=df.copy(); df.insert(ix,'public_trajectory_id',pubs); df.insert(ix+1,'original_trajectory_sha256',hs); return df.drop(columns=['trajectory_id'])
def copy_csv(src,dst,m):
    if not src.exists(): return
    df=pd.read_csv(src,dtype=str,keep_default_na=False); df=pubids(df,m)
    drop=[c for c in df.columns if c.lower() in {'source_file','raw_field_source','evidence_text','raw_extra_json','annotator_id','task_id'}]
    if drop: df=df.drop(columns=drop)
    dst.parent.mkdir(parents=True,exist_ok=True); df.to_csv(dst,index=False,encoding='utf-8-sig')
def relabel_steps(steps,m):
    keep=['dataset','trajectory_id','step_index','label_norm','span_label_norm','hard_negative','productive_iteration_flag','unproductive_cycle_flag','uncertain_flag','unresolved_flag','confidence']
    return pubids(steps[[c for c in keep if c in steps.columns]].copy(),m).rename(columns={'label_norm':'step_label','span_label_norm':'step_span_label'})
def relabel_spans(spans,m):
    keep=['dataset','trajectory_id','start_step_index','end_step_index','span_label_norm','hard_negative','productive_iteration_flag','unproductive_cycle_flag','uncertain_flag','unresolved_flag','confidence','step_count']
    df=pubids(spans[[c for c in keep if c in spans.columns]].copy(),m); df.insert(3,'public_span_id',[f'S{i:07d}' for i in range(1,len(df)+1)]); return df.rename(columns={'span_label_norm':'span_label'})
def project_span_flags(ss,sp):
    ss=ss.copy()
    for c in ['productive_iteration_flag','unproductive_cycle_flag','hard_negative']:
        if c not in ss.columns: ss[c]='False'
    for r in sp.itertuples(index=False):
        try: start=int(float(getattr(r,'start_step_index'))); end=int(float(getattr(r,'end_step_index')))
        except Exception: continue
        tid=getattr(r,'public_trajectory_id'); mask=(ss.public_trajectory_id==tid)&(pd.to_numeric(ss.step_index,errors='coerce')>=start)&(pd.to_numeric(ss.step_index,errors='coerce')<=end)
        label=str(getattr(r,'span_label',''))
        uc=str(getattr(r,'unproductive_cycle_flag','')).lower() in {'true','1','yes'} or label=='unproductive_cycle'
        pi=str(getattr(r,'productive_iteration_flag','')).lower() in {'true','1','yes'} or label=='productive_iteration'
        hn=str(getattr(r,'hard_negative','')).lower() in {'true','1','yes'}
        if uc:
            ss.loc[mask,'unproductive_cycle_flag']='True'; ss.loc[mask,'step_span_label']='unproductive_cycle'
        elif pi:
            ss.loc[mask,'productive_iteration_flag']='True'; ss.loc[mask,'step_span_label']='productive_iteration'
        if hn: ss.loc[mask,'hard_negative']='True'
    return ss

def write_data(repo,art,m,steps,spans):
    ss=relabel_steps(steps,m); sp=relabel_spans(spans,m); ss=project_span_flags(ss,sp); norm=art/'data/normalized'; norm.mkdir(parents=True,exist_ok=True)
    ss.to_csv(norm/'normalized_steps.csv',index=False,encoding='utf-8-sig'); sp.to_csv(norm/'normalized_spans.csv',index=False,encoding='utf-8-sig')
    for ds in POOLS:
        d=art/'data/final_adjudicated'/ds; d.mkdir(parents=True,exist_ok=True); ss[ss.dataset==ds].to_csv(d/'step_labels.csv',index=False,encoding='utf-8-sig'); sp[sp.dataset==ds].to_csv(d/'span_labels.csv',index=False,encoding='utf-8-sig'); wt(d/'README.md',f'# {ds}\n\nFinal released/adjudicated labels only. Expected counts: {STATS[ds][0]} trajectories and {STATS[ds][1]} steps. Raw independent annotations and disagreement records are not included.')
    for src,dst in [('outputs/paper1_main_tables/table1_evidence_pools.csv','dataset_summary.csv'),('outputs/paper1_construct_mismatch/construct_prevalence_by_dataset.csv','construct_prevalence_by_dataset.csv'),('outputs/paper1_construct_mismatch/length_vs_constructs.csv','length_vs_constructs.csv'),('outputs/paper1_raw_enrichment/raw_field_availability_by_dataset.csv','raw_field_availability_by_dataset.csv')]: copy_csv(repo/src,norm/dst,m)
def copy_dir(src,dst,m,exts):
    if not src.exists(): return
    dst.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith('.') and p.name != 'main_table_notes.md':
            copy_csv(p,dst/p.name,m) if p.suffix.lower()=='.csv' else cp(p,dst/p.name)
def copy_fig(src,dst,stem,out):
    for e in ['.pdf','.png','.svg']:
        if (src/f'{stem}{e}').exists(): cp(src/f'{stem}{e}',dst/f'{out}{e}')
def expected(repo,art,m):
    e=art/'outputs/expected'; copy_dir(repo/'outputs/paper1_main_tables',e/'paper_tables',m,{'.csv','.md','.tex'})
    figsrc=repo/'outputs/paper1_publication_figures_v2'
    if not figsrc.exists(): figsrc=repo/'outputs/paper1_publication_figures'
    for i in range(1,5): copy_fig(figsrc,e/'paper_figures',f'fig{i}',f'fig{i}')
    copy_fig(repo/'outputs/paper1_intervention_validation',e/'paper_figures','fig_stop_counterfactual_tradeoff','fig5_stop_counterfactual_tradeoff')
    for name in ['construct_mismatch','length_robustness','intervention_risk_rich_detectors','intervention_validation','paper_findings']:
        copy_dir(repo/f'outputs/paper1_{name}',e/name,m,{'.csv','.md','.json','.pdf','.png','.svg'} if name=='intervention_validation' else {'.csv','.md'})
    prov=e/'provenance_audit'; prov.mkdir(parents=True,exist_ok=True)
    for n in ['provenance_summary_by_pool.csv','provenance_audit_report.md']:
        src=repo/'outputs/paper1_provenance_audit'/n
        if src.exists(): copy_csv(src,prov/n,m) if n.endswith('.csv') else cp(src,prov/n)
def script(repo,art):
    pairs=[('scripts/paper1_audit/audit_dataset_integrity.py','scripts/audit/audit_dataset_integrity.py'),('scripts/paper1_audit/construct_distribution_analysis.py','scripts/audit/construct_distribution_analysis.py'),('scripts/paper1_audit/load_paper1_datasets.py','scripts/audit/load_paper1_datasets.py'),('scripts/paper1_audit/make_tables_and_figures.py','scripts/audit/make_tables_and_figures.py'),('scripts/paper1_audit/enrich_normalized_steps_with_raw_fields.py','scripts/raw_enrichment/enrich_normalized_steps_with_raw_fields.py'),('scripts/paper1_construct_mismatch/build_construct_mismatch_analysis.py','scripts/construct_mismatch/build_construct_mismatch_analysis.py'),('scripts/paper1_construct_mismatch/length_robustness_checks.py','scripts/construct_mismatch/length_robustness_checks.py'),('scripts/paper1_intervention_risk/build_intervention_risk_evaluation.py','scripts/intervention_risk/build_intervention_risk_evaluation.py'),('scripts/paper1_intervention_risk/build_rich_detector_evaluation.py','scripts/intervention_risk/build_rich_detector_evaluation.py'),('scripts/paper1_intervention_validation/check_intervention_feasibility.py','scripts/intervention_validation/check_intervention_feasibility.py'),('scripts/paper1_intervention_validation/build_stop_counterfactual.py','scripts/intervention_validation/build_stop_counterfactual.py'),('scripts/paper1_intervention_validation/build_prefix_branch_experiment.py','scripts/intervention_validation/build_prefix_branch_experiment.py'),('scripts/paper1_intervention_validation/run_prefix_branch_experiment.py','scripts/intervention_validation/run_prefix_branch_experiment.py'),('scripts/paper1_intervention_validation/analyze_prefix_branch_results.py','scripts/intervention_validation/analyze_prefix_branch_results.py'),('scripts/paper1_synthesis/extract_paper_findings.py','scripts/synthesis/extract_paper_findings.py'),('scripts/paper1_synthesis/build_main_paper_tables.py','scripts/synthesis/build_main_paper_tables.py'),('scripts/paper1_synthesis/build_publication_figures.py','scripts/synthesis/build_publication_figures.py'),('scripts/paper1_synthesis/build_publication_figures_v2.py','scripts/synthesis/build_publication_figures_v2.py'),('scripts/release/build_paper1_artifact.py','scripts/release/build_paper1_artifact.py'),('scripts/release/audit_and_test_paper1_artifact.py','scripts/release/audit_and_test_paper1_artifact.py')]
    for s,d in pairs:
        if (repo/s).exists(): cp(repo/s,art/d)
    for sub in ['audit','construct_mismatch','raw_enrichment','intervention_risk','intervention_validation','synthesis','release']:
        (art/'scripts'/sub).mkdir(parents=True,exist_ok=True); (art/'scripts'/sub/'__init__.py').write_text('',encoding='utf-8')
def docs(art):
    rows='\n'.join([f"| {p} | {STATS[p][4]} | {STATS[p][0]} | {STATS[p][1]} | {STATS[p][2]:.3f} | {STATS[p][3]:.3f} | {STATS[p][5]} | final adjudicated labels |" for p in POOLS])
    wt(art/'README.md',f'''
# LoopSense Paper 1 Artifact

## Paper
**When Is an Agent Really Stuck? Construct Validity and Productive-Interruption Risk in Software-Engineering Agents**

This artifact supports reproduction of the main audit tables, construct-mismatch analyses, detector-risk analyses, publication figures, and offline stop-counterfactual intervention validation.

## What this artifact contains
- Final released/adjudicated labels only.
- Normalized step and span tables with anonymized public trajectory identifiers.
- Analysis scripts, release scripts, expected paper outputs, paper figures/tables, intervention-validation outputs, provenance summaries, and integrity reports.

## What this artifact does NOT contain
- Raw independent double-annotation sheets.
- Disagreement, conflict, negotiation, or private annotation records.
- Raw un-anonymized trajectories, local machine paths, API keys, tokens, or full model execution credentials.
- A live prefix-branch rerun stack; original agent trajectory generation is out of scope.
- Inputs required to recompute Cohen's kappa; only final adjudicated labels are released.

## Evidence pools
| Pool | Framework/source | Trajectories | Steps | Avg. steps | UC prevalence | Role | Label source |
|---|---|---:|---:|---:|---:|---|---|
{rows}

## Reproducibility levels
- **Level 1:** regenerate/check paper tables and figures from included expected CSVs.
- **Level 2:** verify analysis results from released normalized data and included intermediate tables.
- **Level 3:** raw trajectory rerun or live prefix-branch intervention rerun; not included.

The artifact supports Level 1 and Level 2. Raw-runtime-dependent detector reruns are partial by design when non-released raw trajectory logs would be required. The offline stop-counterfactual analysis is included and reproducible from included trigger/intermediate tables.

## Quick start
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python tests/test_artifact_integrity.py
python tests/test_reproduce_key_outputs.py
python scripts/release/audit_and_test_paper1_artifact.py --artifact-root .
```

## Expected headline values
- OpenHandsExternal330 average steps: 154.576; UC prevalence: 0.012.
- Natural500 steps: 8,550; Final500 steps: 26,014; StressFresh100 steps: 3,132; OpenHandsExternal330 steps: 51,010.
- Final500 near-F1 pair: F1 0.719 vs 0.729, delta F1 0.009, delta PIIR 0.461, delta HN-FPR 0.230.
- StressFresh100 file-revisit window 10 repeats 3: UC recall 0.958, PIIR 0.868, HN-FPR 0.797.
- Offline stop-counterfactual trigger rows: 246,856.
- Final500 max-step T=50 PSL/UWS: 11.8 / 48.1; file-revisit w=5,r=3 PSL/UWS: 24.8 / 36.2.

## Directory structure
`data/final_adjudicated/` contains released labels. `data/normalized/` contains normalized tables. `scripts/` contains artifact-relative analysis scripts. `outputs/expected/` contains expected tables, figures, metrics, intervention-validation outputs, and provenance summaries. `docs/` contains schema and reproduction documentation. `tests/` contains integrity and headline reproducibility tests.

## Annotation release policy
Labels are semantic local-process annotations over trajectory, step, and span units. The paper reports two trained annotators and Cohen's kappa 0.762 before adjudication, but this artifact releases final adjudicated labels only. Uncertain and unresolved states are preserved when applicable. Detector outputs and final task outcomes are not annotation evidence.

## Intervention validation
PIIR and HN-FPR are trigger-site risk metrics. The offline stop-counterfactual evaluates what observed suffix would be cut off if the first detector/guard trigger were converted into a hard stop. This is not a live causal rerun and does not evaluate warn, replan, handoff, or escalation policies.

## Provenance
Pool-level provenance summaries are in `outputs/expected/provenance_audit/` and `docs/provenance.md`. Detailed raw provenance hits are not included because they are large and can contain raw source identifiers.

## Known limitations
Raw-field availability differs by pool. Live prefix-branch reruns are not included. Full raw trajectory generation is not reproduced. Independent annotation sheets and disagreement records are intentionally excluded.

## Citation
```bibtex
@inproceedings{{anonymous2027loopsense,
  title={{When Is an Agent Really Stuck? Construct Validity and Productive-Interruption Risk in Software-Engineering Agents}},
  author={{Anonymous Authors}},
  booktitle={{Anonymous submission}},
  year={{2027}}
}}
```

## License
License to be finalized before public release.
''')
    wt(art/'LICENSE','License to be finalized before public release.')
    wt(art/'CITATION.cff','''cff-version: 1.2.0
title: "When Is an Agent Really Stuck? Construct Validity and Productive-Interruption Risk in Software-Engineering Agents"
message: "Please cite the anonymous paper if using this artifact."
authors:
  - name: "Anonymous Authors"
year: 2027''')
    wt(art/'requirements.txt','''numpy
pandas
matplotlib
scipy
scikit-learn
pyarrow''')
    wt(art/'environment.yml','''name: loopsense-paper1-artifact
channels: [conda-forge]
dependencies:
  - python>=3.10
  - numpy
  - pandas
  - matplotlib
  - scipy
  - scikit-learn
  - pyarrow''')
    wt(art/'DATASET_CARD.md','''# Dataset Card

This artifact contains four final released evidence pools: Natural500, Final500, StressFresh100, and OpenHandsExternal330. Natural500 is a natural-distribution control, Final500 is an enriched difficult loop-heavy set, StressFresh100 is a clean stress challenge, and OpenHandsExternal330 is an external long-horizon pool.

Final500 and StressFresh100 are not natural prevalence samples. Raw-field availability differs by pool. Two software-engineering master's students familiar with agent-based debugging independently reviewed trajectory-local evidence. Cohen's kappa was 0.762 before adjudication. Only final adjudicated labels are released.

Labels: productive iteration (PI), hard negative (HN), unproductive cycle (UC), uncertain, and unresolved.''')
    wt(art/'ARTIFACT_EVALUATION.md','# Artifact Evaluation Notes\n\nThis package supports paper-result reproduction and headline checks. It does not rerun original agent trajectories or release raw independent annotation materials.')
    wt(art/'anonymization_and_release_notes.md','# Anonymization and Release Notes\n\nThe release uses public trajectory identifiers and SHA256 digests. Raw trajectory ids, task ids, local source paths, evidence text, raw extra JSON, independent annotation sheets, and disagreement records are excluded.')
    wt(art/'docs/metric_definitions.md','# Metric Definitions\n\nUC Recall = recalled UC spans / all UC spans. PIIR = interrupted PI spans / all PI spans. HN-FPR = triggered HN-eligible productive steps / all HN-eligible productive steps. Burden = triggered eligible steps / all eligible steps. Latency = first trigger step minus UC span start. PSL and UWS are offline stop-counterfactual productive suffix loss and UC waste saved.')
    wt(art/'docs/label_schema.md','# Label Schema\n\nStep tables include dataset, public trajectory id, original trajectory SHA256, step index, step label, span label, hard-negative flag, PI flag, UC flag, uncertain flag, unresolved flag, and confidence where available. Span tables include public trajectory id, public span id, span start/end, span label, PI/UC/HN flags, uncertainty flags, and span length.')
    wt(art/'docs/detector_families.md','# Detector Families\n\nIncluded detector/guard families: max_step_guard, file_revisit_guard, exact_action_repeat, tool_repeat, tool_sequence_repeat, action_observation_repeat, error_signature_repeat, and repeat_without_progress_guard. These are common progress-insensitive guards plus a simple progress-aware contrast, not a SOTA detector claim. Detector features do not use PI/HN/UC labels.')
    wt(art/'docs/reproduction_guide.md','# Reproduction Guide\n\nRun from artifact root:\n\n```bash\npip install -r requirements.txt\npython tests/test_artifact_integrity.py\npython tests/test_reproduce_key_outputs.py\npython scripts/release/audit_and_test_paper1_artifact.py --artifact-root .\n```\n\nExpected outputs are under `outputs/expected/`. Some raw-runtime detector reruns require non-released raw trajectory fields; use included expected intermediate tables and `detector_availability.csv` for partial reproducibility checks.')
    wt(art/'docs/provenance.md','# Provenance\n\nThe release includes pool-level provenance summaries in `outputs/expected/provenance_audit/`. Detailed raw provenance hits are not included because they are large and can contain raw source identifiers.')
    wt(art/'docs/intervention_validation.md','# Intervention Validation\n\nOffline stop-counterfactual validation asks what observed suffix would be cut off if the first detector/guard trigger became a hard stop. It reports Productive Suffix Loss and UC Waste Saved. It does not estimate live causal success effects.')
def tests(art):
    wt(art/'tests/test_artifact_integrity.py',r'''
from __future__ import annotations
import csv,hashlib,re,subprocess,sys,py_compile,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; POOLS=['Natural500','Final500','StressFresh100','OpenHandsExternal330']
LOCAL_PATTERNS=[re.compile(r'(?<![A-Za-z0-9_])D:[\\/][^\s,;]*',re.I),re.compile(r'\\\\'+'wsl'+r'\.localhost',re.I),re.compile(r'/home/[A-Za-z0-9_.-]+')]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def main():
    for x in ['README.md','LICENSE','CITATION.cff','manifest_sha256.csv','data/normalized/normalized_steps.csv','data/normalized/normalized_spans.csv']:
        assert (ROOT/x).exists(),f'missing {x}'
    for pool in POOLS:
        b=ROOT/'data/final_adjudicated'/pool
        assert (b/'step_labels.csv').exists() and (b/'span_labels.csv').exists(),pool
        assert len(pd.read_csv(b/'step_labels.csv'))>0
    for p in ROOT.rglob('*'):
        assert '__pycache__' not in p.parts and '.git' not in p.parts and '.venv' not in p.parts, f'forbidden cache/env {p}'
        assert p.suffix.lower()!='.zip', f'nested zip {p}'
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md','.csv','.json','.py','.yml','.yaml','.cff','.txt'} and p.stat().st_size<80_000_000:
            t=p.read_text(encoding='utf-8',errors='ignore')
            for pat in LOCAL_PATTERNS: assert not pat.search(t), f'local path in {p}'
            assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',t),f'email-like string in {p}'
            assert not re.search(r'sk-[A-Za-z0-9_-]{20,}',t),f'secret-like token in {p}'
    with (ROOT/'manifest_sha256.csv').open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            p=ROOT/r['path']; assert p.exists(),f'manifest missing {r["path"]}'; assert sha(p)==r['sha256'],f'hash mismatch {r["path"]}'
    for s in (ROOT/'scripts').rglob('*.py'):
            with tempfile.TemporaryDirectory() as td:
                py_compile.compile(str(s), cfile=str(Path(td)/'out.pyc'), doraise=True)
    print('artifact integrity test passed')
if __name__=='__main__': main()
''')
    wt(art/'tests/test_reproduce_key_outputs.py',r'''
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; T3=0.0015; T1=0.1
def close(a,e,t): assert abs(float(a)-float(e))<=t,f'{a}!={e}'
def row(df,**conds):
    m=pd.Series([True]*len(df))
    for k,v in conds.items(): m &= df[k].astype(str).eq(str(v))
    assert m.any(),f'missing {conds}'; return df[m].iloc[0]
def main():
    for pool,(traj,steps) in {'Natural500':(500,8550),'Final500':(500,26014),'StressFresh100':(100,3132),'OpenHandsExternal330':(330,51010)}.items():
        df=pd.read_csv(ROOT/'data/final_adjudicated'/pool/'step_labels.csv'); assert len(df)==steps; assert df.public_trajectory_id.nunique()==traj
    t1=pd.read_csv(ROOT/'outputs/expected/paper_tables/table1_evidence_pools.csv'); oh=row(t1,**{'Evidence pool':'OpenHandsExternal330'}); close(oh['Avg. steps'],154.576,0.0005); close(oh['UC prevalence'],0.012,T3)
    met=pd.read_csv(ROOT/'outputs/expected/intervention_risk_rich_detectors/detector_metrics_by_dataset.csv')
    a=row(met,dataset='Final500',detector='max_step_guard',**{'threshold/config':'threshold=50'}); b=row(met,dataset='Final500',detector='file_revisit_guard',**{'threshold/config':'window=5;repeats=3'})
    close(a.F1,0.719,T3); close(b.F1,0.729,T3)
    near=pd.read_csv(ROOT/'outputs/expected/intervention_risk_rich_detectors/near_f1_risk_pairs.csv'); n=row(near,dataset='Final500',detector_a='max_step_guard',config_a='threshold=50',detector_b='file_revisit_guard',config_b='window=5;repeats=3')
    close(n.abs_F1_diff,0.009,T3); close(n.abs_PIIR_diff,0.461,T3); close(n.abs_HN_FPR_diff,0.230,T3)
    s=row(met,dataset='StressFresh100',detector='file_revisit_guard',**{'threshold/config':'window=10;repeats=3'}); close(s.UC_recall,0.958,T3); close(s.PIIR,0.868,T3); close(s.HN_FPR,0.797,T3)
    trig=pd.read_csv(ROOT/'outputs/expected/intervention_validation/stop_counterfactual_triggers.csv',usecols=['trigger_id']); assert len(trig)==246856
    stop=pd.read_csv(ROOT/'outputs/expected/intervention_validation/stop_counterfactual_summary_by_detector.csv')
    sa=row(stop,dataset='Final500',detector='max_step_guard',detector_config='threshold=50'); sb=row(stop,dataset='Final500',detector='file_revisit_guard',detector_config='window=5;repeats=3')
    close(sa.productive_suffix_cut_mean,11.8,T1); close(sa.UC_suffix_saved_mean,48.1,T1); close(sb.productive_suffix_cut_mean,24.8,T1); close(sb.UC_suffix_saved_mean,36.2,T1)
    print('headline reproducibility test passed; raw-runtime-dependent reruns are partial by design')
if __name__=='__main__': main()
''')
def reports(repo,art):
    count=sum(1 for p in (repo/'data/annotations/paper1').rglob('*') if p.is_file()) if (repo/'data/annotations/paper1').exists() else 0
    wt(art/'release_exclusion_report.md',f'# Release Exclusion Report\n\nThe builder does not copy raw annotation trees. Excluded raw/source files considered: {count}. Excluded categories include raw independent annotation sheets, disagreement/conflict/negotiation materials, private notes, draft or old materials, raw trajectory logs, cache files, previous release archives, and local-path-bearing raw reports.')
    pd.DataFrame([{'category':'raw_or_nonrelease_source','excluded_file_count':count,'reason':'not copied into release artifact'}]).to_csv(art/'release_exclusion_report.csv',index=False,encoding='utf-8-sig')
    wt(art/'artifact_integrity_report.md','# Artifact Integrity Report\n\nThe artifact was rebuilt from current final normalized outputs and expected paper outputs. Run the tests and audit script for validation.')
    wt(art/'artifact_release_final_report.md','# Artifact Release Final Report\n\nStatus is determined after running the reproducibility audit.')
def manifest(art):
    rows=[]
    for p in sorted(x for x in art.rglob('*') if x.is_file()):
        r=str(p.relative_to(art)).replace('\\','/')
        if p.name in DYNAMIC or '__pycache__' in p.parts or r.startswith('_reproduction_tmp/'): continue
        rows.append({'path':r,'size_bytes':p.stat().st_size,'sha256':sha_file(p)})
    with (art/'manifest_sha256.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','size_bytes','sha256']); w.writeheader(); w.writerows(rows)
def zipit(art,zp):
    if zp.exists(): zp.unlink()
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(x for x in art.rglob('*') if x.is_file()):
            if '__pycache__' in p.parts or p.suffix.lower()=='.zip': continue
            z.write(p,str(Path(art.name)/p.relative_to(art)).replace('\\','/'))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',type=Path,default=root_from_script()); ap.add_argument('--release-dir',type=Path,default=Path('release_artifacts/LoopSense_Paper1_Artifact')); ap.add_argument('--zip-path',type=Path,default=Path('release_artifacts/LoopSense_Paper1_Artifact.zip')); a=ap.parse_args()
    repo=a.repo_root.resolve(); art=a.release_dir if a.release_dir.is_absolute() else repo/a.release_dir; zp=a.zip_path if a.zip_path.is_absolute() else repo/a.zip_path
    clean_release(art); steps,spans=load(repo); m=mapping(steps); write_data(repo,art,m,steps,spans); expected(repo,art,m); script(repo,art); docs(art); tests(art); reports(repo,art); manifest(art); zipit(art,zp); print(f'Built artifact: {art}'); print(f'Built zip: {zp}')
if __name__=='__main__': main()



