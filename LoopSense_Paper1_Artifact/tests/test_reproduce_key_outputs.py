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
    a=row(met,dataset='Enriched500',detector='max_step_guard',**{'threshold/config':'threshold=50'}); b=row(met,dataset='Enriched500',detector='file_revisit_guard',**{'threshold/config':'window=5;repeats=3'})
    close(a.F1,0.719,T3); close(b.F1,0.729,T3)
    near=pd.read_csv(ROOT/'outputs/expected/intervention_risk_rich_detectors/near_f1_risk_pairs.csv'); n=row(near,dataset='Enriched500',detector_a='max_step_guard',config_a='threshold=50',detector_b='file_revisit_guard',config_b='window=5;repeats=3')
    close(n.abs_F1_diff,0.009,T3); close(n.abs_PIIR_diff,0.461,T3); close(n.abs_HN_FPR_diff,0.230,T3)
    s=row(met,dataset='StressFresh100',detector='file_revisit_guard',**{'threshold/config':'window=10;repeats=3'}); close(s.UC_recall,0.958,T3); close(s.PIIR,0.868,T3); close(s.HN_FPR,0.797,T3)
    trig=pd.read_csv(ROOT/'outputs/expected/intervention_validation/stop_counterfactual_triggers.csv',usecols=['trigger_id']); assert len(trig)==246856
    stop=pd.read_csv(ROOT/'outputs/expected/intervention_validation/stop_counterfactual_summary_by_detector.csv')
    sa=row(stop,dataset='Enriched500',detector='max_step_guard',detector_config='threshold=50'); sb=row(stop,dataset='Enriched500',detector='file_revisit_guard',detector_config='window=5;repeats=3')
    close(sa.productive_suffix_cut_mean,11.8,T1); close(sa.UC_suffix_saved_mean,48.1,T1); close(sb.productive_suffix_cut_mean,24.8,T1); close(sb.UC_suffix_saved_mean,36.2,T1)
    api_manifest=pd.read_json(ROOT/'outputs/expected/api_replan_intervention/api_replan_manifest.json',typ='series')
    assert int(api_manifest['prefixes_with_complete_pairs'])==180
    assert int(api_manifest['ok_parsed_rows'])==360
    api=pd.read_csv(ROOT/'outputs/expected/api_replan_intervention/api_replan_pairwise_effects.csv')
    shift=row(api,stratum='all',metric='action_shift_rate'); close(shift.warn_replan_mean,0.7111111111111111,T3); close(shift.ci_low,0.6444444444444445,T3); close(shift.ci_high,0.7777777777777778,T3)
    score=row(api,stratum='all',metric='progress_seeking_score_mean'); close(score.neutral_mean,2.0,T3); close(score.warn_replan_mean,2.1666666666666665,T3); close(score.warn_minus_neutral,0.16666666666666666,T3)
    repeat=row(api,stratum='all',metric='repeat_pattern_rate'); close(repeat.neutral_mean,0.6388888888888888,T3); close(repeat.warn_replan_mean,0.6555555555555556,T3)
    overstop=row(api,stratum='productive_risk',metric='unsafe_overstop_rate'); close(overstop.neutral_mean,0.027777777777777776,T3); close(overstop.warn_replan_mean,0.013888888888888888,T3)
    live=pd.read_csv(ROOT/'outputs/expected/live_guard_official_oracle/live_guard_official_summary_by_arm.csv')
    live_expect={
        'NO_GUARD':(35,48,42.0,758692),
        'WARN_REPLAN_GUARD':(42,54,40.2,744589),
        'HARD_STOP_GUARD':(8,9,8.0,67225),
    }
    for arm,(resolved,patches,steps,tokens) in live_expect.items():
        r=row(live,arm=arm)
        n=int(round(float(r.N_official_oracle_outcomes)))
        attempted=int(round(float(r.N_attempted)))
        assert n==60
        assert attempted==60
        assert int(round(float(r.resolved_rate)*n))==resolved
        assert int(round(float(r.tests_pass_rate)*n))==resolved
        assert int(round(float(r.patch_generated_rate)*attempted))==patches
        close(r.mean_steps,steps,0.05)
        close(r.mean_tokens,tokens,0.5)
    effects=pd.read_csv(ROOT/'outputs/expected/live_guard_official_oracle/live_guard_official_pairwise_effects.csv')
    wr=row(effects,comparison='WARN_REPLAN_GUARD minus NO_GUARD',metric='resolved_num'); close(wr.mean_difference,0.11666666666666667,T3); close(wr.bootstrap95_low,0.05,T3); close(wr.bootstrap95_high,0.2,T3)
    wp=row(effects,comparison='WARN_REPLAN_GUARD minus NO_GUARD',metric='patch_generated_num'); close(wp.mean_difference,0.1,T3)
    hr=row(effects,comparison='HARD_STOP_GUARD minus NO_GUARD',metric='resolved_num'); close(hr.mean_difference,-0.45,T3)
    hs=row(effects,comparison='HARD_STOP_GUARD minus NO_GUARD',metric='steps'); close(hs.mean_difference,-33.96666666666667,T3)
    hp=row(effects,comparison='HARD_STOP_GUARD minus NO_GUARD',metric='patch_generated_num'); close(hp.mean_difference,-0.65,T3)
    print('headline reproducibility test passed; raw-runtime-dependent reruns are partial by design')
if __name__=='__main__': main()
