#!/usr/bin/env python3
"""Corpus-only distances, independent of optimizer gradients and endpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon, cdist
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs/real_task_eight_theme_cpt'
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
COLORS = ['#377eb8']*8+['#e66101']*4+['#4daf4a']*4


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def hist(a, size):
    v = np.bincount(a.ravel(), minlength=size).astype(float)
    return v/v.sum()


def lex_distance(p, q):
    return {'token_js': float(jensenshannon(p,q,base=2)),
            'token_hellinger': float(np.linalg.norm(np.sqrt(p)-np.sqrt(q))/np.sqrt(2)),
            'token_total_variation': float(np.abs(p-q).sum()/2),
            'token_cosine_distance': float(1-p@q/(np.linalg.norm(p)*np.linalg.norm(q)))}


def format_features(text):
    n=max(len(text),1)
    return [sum(c.isalpha() for c in text)/n, sum(c.isdigit() for c in text)/n,
            sum(c.isspace() for c in text)/n, sum(not c.isalnum() and not c.isspace() for c in text)/n,
            sum(ord(c)>127 for c in text)/n, text.count('\n')/n]


def frechet(a,b):
    ma,mb=a.mean(0),b.mean(0)
    ca,cb=LedoitWolf().fit(a).covariance_,LedoitWolf().fit(b).covariance_
    w,u=np.linalg.eigh(ca); s=(u*np.sqrt(np.maximum(w,0)))@u.T
    tr=np.sqrt(np.maximum(np.linalg.eigvalsh(s@cb@s),0)).sum()
    return float(max(0,np.sum((ma-mb)**2)+np.trace(ca)+np.trace(cb)-2*tr))


def embed_metrics(a,b,bw):
    aa=cdist(a,a,'sqeuclidean'); bb=cdist(b,b,'sqeuclidean'); ab=cdist(a,b,'sqeuclidean')
    out={}
    for scale in [.5,1.,2.]:
        den=2*(bw*scale)**2
        # Biased V-statistic: nonnegative, includes finite-sample diagonal floor.
        out[f'embedding_mmd2_bw{scale:g}']=float(max(0,np.exp(-aa/den).mean()+np.exp(-bb/den).mean()-2*np.exp(-ab/den).mean()))
    out['embedding_energy_distance']=float(max(0,2*np.sqrt(ab).mean()-np.sqrt(aa).mean()-np.sqrt(bb).mean()))
    ma,mb=a.mean(0),b.mean(0)
    out['embedding_centroid_cosine_distance']=float(1-ma@mb/(np.linalg.norm(ma)*np.linalg.norm(mb)))
    return out


def auc(a,b,seed):
    x=np.concatenate([a,b]); y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    sc=StandardScaler().fit(x[tr]); model=LogisticRegression(C=1.,max_iter=1000).fit(sc.transform(x[tr]),y[tr])
    return float(roc_auc_score(y[te],model.predict_proba(sc.transform(x[te]))[:,1]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=False)
    (out/'samples').mkdir(); (out/'embeddings').mkdir()
    endpoints=pd.concat([pd.read_csv(RUNS/f'run{r}'/d/'endpoints.csv') for r,d in [(2,'comparison_v3'),(3,'comparison_v2'),(4,'comparison_v2')]],ignore_index=True)
    tasks=list(dict.fromkeys(endpoints.task)); assert len(tasks)==16
    caches={t:next(RUNS.glob(f'run*/cache/{t}/manifest.json')).parent for t in tasks}
    # Prefer provenance-matched source runs, not the first arbitrary matching cache.
    for t in tasks:
        r=2 if t in tasks[:8] else 3 if t in tasks[8:12] else 4
        caches[t]=RUNS/f'run{r}'/'cache'/t
    base=caches['openwebmath']; train=np.load(base/'train_seed42_rho20_tokens.npy',mmap_mode='r'); sid=np.load(base/'train_seed42_rho20_source_ids.npy')
    old=np.array(train[sid==1,:1024]); pile=np.load(base/'retention_tokens.npy')[:,:1024]
    arrays={}; sampling={}; rng=np.random.default_rng(20261020)
    for t,c in caches.items():
        tape=np.load(c/'train_seed42_rho20_tokens.npy',mmap_mode='r'); ids=np.load(c/'train_seed42_rho20_source_ids.npy')
        assert np.array_equal(tape[ids==1,:1024],old),t
        indices=rng.choice(np.flatnonzero(ids==0),768,replace=False)
        arrays[t]=np.array(tape[indices,:1024]); sampling[t]={'cache':str(c),'row_indices':indices.tolist(),'sample_sha256':sha(arrays[t])}
    for t,a in [('old_replay',old),('pile_heldout',pile)]:
        idx=rng.choice(len(a),768,replace=False); arrays[t]=a[idx]
        sampling[t]={'reference_pool_rows':len(a),'row_indices':idx.tolist(),'sample_sha256':sha(arrays[t])}
    manifest=json.loads((base/'manifest.json').read_text()); tok=AutoTokenizer.from_pretrained(manifest['tokenizer_id'],revision=manifest['tokenizer_revision'],local_files_only=True)
    dump(out/'config.json',dict(tasks=tasks,sampling=sampling,seed=20261020,repeats=5,windows_per_side=384,
         token_length=1024,embedding_pythia_token_slice=[384,640],encoder=ENCODER,encoder_revision=REV,
         embedding_max_wordpieces=256,reference='actual Old replay and weighted Pile held-out windows; neither is the full original pretraining corpus',
         uncertainty='Repeated window subsamples, not independent training seeds or document bootstrap. Packed windows can share source documents.',
         endpoint_source=[str(RUNS/f'run{r}'/d/'endpoints.csv') for r,d in [(2,'comparison_v3'),(3,'comparison_v2'),(4,'comparison_v2')]]))
    texts={}; char={}; formats={}; vocab=max(int(a.max()) for a in arrays.values())+1
    hv=HashingVectorizer(analyzer='char',ngram_range=(3,3),n_features=2**16,alternate_sign=False,norm=None,lowercase=False)
    for t,a in arrays.items():
        np.save(out/'samples'/f'{t}.npy',a)
        texts[t]=tok.batch_decode(a[:,384:640],skip_special_tokens=True)
        char[t]=hv.transform(texts[t]); formats[t]=np.array([format_features(x) for x in texts[t]])
        print('sampled',t,flush=True)
    etok=AutoTokenizer.from_pretrained(ENCODER,revision=REV)
    model=AutoModel.from_pretrained(ENCODER,revision=REV).to(args.device).eval(); emb={}
    torch.set_num_threads(4)
    for t,ss in texts.items():
        vv=[]
        for start in range(0,len(ss),64):
            encoded=etok(ss[start:start+64],return_tensors='pt',padding=True,truncation=True,max_length=256).to(args.device)
            with torch.inference_mode():
                hidden=model(**encoded).last_hidden_state; mask=encoded['attention_mask'].unsqueeze(-1)
                v=(hidden*mask).sum(1)/mask.sum(1); v=torch.nn.functional.normalize(v,p=2,dim=1)
            vv.append(v.float().cpu().numpy())
        emb[t]=np.concatenate(vv); np.save(out/'embeddings'/f'{t}.npy',emb[t]); print('embedded',t,flush=True)
    del model; torch.cuda.empty_cache()
    refemb=np.concatenate([emb['old_replay'],emb['pile_heldout']]); bw=float(np.sqrt(np.median(cdist(refemb[::3],refemb[1::3],'sqeuclidean'))))
    # Reference-only PCA and format scaling avoid outcome/task-dependent fitting.
    pca=PCA(n_components=64,random_state=42).fit(refemb)
    fmt_scale=np.maximum(np.std(np.concatenate([formats['old_replay'],formats['pile_heldout']]),axis=0),.01)
    rows=[]
    for rep in range(5):
        rr=np.random.default_rng(42+rep); perms={t:rr.permutation(768) for t in arrays}
        for reference in ['old_replay','pile_heldout']:
            ib=perms[reference][:384]; q=hist(arrays[reference][ib],vocab)
            cq=np.asarray(char[reference][ib].sum(axis=0)).ravel(); cq/=cq.sum()
            for t in tasks+[reference]:
                ia=perms[t][384:] if t==reference else perms[t][:384]
                a,b=emb[t][ia],emb[reference][ib]
                row=dict(task=t,reference=reference,repeat=rep,is_sampling_floor=t==reference)
                row.update(lex_distance(hist(arrays[t][ia],vocab),q))
                cp=np.asarray(char[t][ia].sum(axis=0)).ravel(); cp/=cp.sum()
                row['char3_js_hashed']=float(jensenshannon(cp,cq,base=2))
                row['format_standardized_centroid_distance']=float(np.linalg.norm((formats[t][ia].mean(0)-formats[reference][ib].mean(0))/fmt_scale))
                row.update(embed_metrics(a,b,bw)); row['embedding_frechet_pca64_shrinkage']=frechet(pca.transform(a),pca.transform(b))
                row['embedding_domain_classifier_auc']=auc(a,b,rep+42)
                rows.append(row)
        print('repeat',rep,'complete',flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(out/'distance_subsamples.csv',index=False)
    metrics=[c for c in raw if c not in ['task','reference','repeat','is_sampling_floor']]
    means=raw.groupby(['task','reference','is_sampling_floor'])[metrics].agg(['mean','std']); means.columns=['_'.join(c) for c in means.columns]; means.reset_index().to_csv(out/'distance_summary.csv',index=False)
    gaps=[]
    for t in tasks:
        es=endpoints[endpoints.task==t].set_index('method'); row={'task':t,'initial_new_ce':es.loc['pion','new_ce_step0']}
        for method in ['muon','adam']:
            for side in ['new','old']:
                row[f'pion_minus_{method}_{side}_ce']=es.loc['pion',f'{side}_ce_step610']-es.loc[method,f'{side}_ce_step610']
        gaps.append(row)
    gaps=pd.DataFrame(gaps); gaps.to_csv(out/'optimizer_gaps.csv',index=False)
    mm=raw[~raw.is_sampling_floor].groupby(['task','reference'])[metrics].mean().reset_index().merge(gaps,on='task')
    mm.to_csv(out/'distances_and_optimizer_gaps.csv',index=False)
    correlations=[]; prng=np.random.default_rng(2727)
    ys=[c for c in gaps if c not in ['task','initial_new_ce']]
    for ref,table in mm.groupby('reference'):
        for metric in metrics:
            x=table[metric].to_numpy(); xr=pd.Series(x).rank().to_numpy(); xr=(xr-xr.mean())/np.linalg.norm(xr-xr.mean())
            for yname in ys:
                y=table[yname].to_numpy(); rho,p=spearmanr(x,y); yr=pd.Series(y).rank().to_numpy(); yr=(yr-yr.mean())/np.linalg.norm(yr-yr.mean())
                null=np.array([xr@prng.permutation(yr) for _ in range(4999)])
                loo=[spearmanr(np.delete(x,i),np.delete(y,i)).statistic for i in range(len(x))]
                # Descriptive sensitivity to structured-four exclusion; no causal interpretation.
                subset=~table.task.isin(['chembl','pdb','uniprot','cve'])
                correlations.append(dict(reference=ref,metric=metric,outcome=yname,spearman_rho=rho,
                    permutation_p=(1+np.sum(np.abs(null)>=abs(rho)))/5000,loo_rho_min=min(loo),loo_rho_max=max(loo),
                    rho_without_structured_four=spearmanr(table.loc[subset,metric],table.loc[subset,yname]).statistic))
    cor=pd.DataFrame(correlations); p=cor.permutation_p.to_numpy(); order=np.argsort(p); q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    adjusted=np.empty_like(q); adjusted[order]=np.minimum(q,1); cor['bh_q_all_tests']=adjusted
    cor.to_csv(out/'correlations.csv',index=False)
    floors=raw[raw.is_sampling_floor].groupby('reference')[metrics].mean()
    with PdfPages(out/'corpus_distance_summary.pdf') as pdf:
        fig=plt.figure(figsize=(12,8)); fig.text(.07,.93,'Corpus distance vs optimizer differences',size=22,weight='bold')
        lines=['16 existing CPT tasks; no new training, no gradients used in distance calculation.',
          'References: exact Old replay; weighted Pile held-out (pretraining-distribution proxy).',
          '5 repeated subsamples, 384 x 1024-token windows per side; SD is sampling variation only.',
          'Token distances: JS (base 2), Hellinger, total variation, cosine.',
          'Text distances: hashed character 3-gram JS; standardized format proportions.',
          'Frozen MiniLM: RBF MMD at 3 bandwidths; energy; centroid; Frechet; classifier AUC.',
          'Embedding input: middle 256 Pythia tokens/window, capped at 256 encoder wordpieces.',
          'MMD is a biased nonnegative V-statistic. Reference split floor is reported.',
          'Frechet: reference-only PCA64 + Ledoit-Wolf covariance. AUC: held-out 30%.',
          'Packed windows may share documents. AUC is window discrimination, not proof of domain novelty.',
          'English embeddings may poorly capture molecules, coordinates and protein sequences.',
          'Outcomes: Pion minus Full/AdamW CE within each task; negative means Pion wins.',
          '16 topics, one training seed. Correlation is exploratory, not causal or cross-seed evidence.',
          'Permutation tests (4999); BH correction across all metric/reference/outcome tests.',
          'Source CSVs, sampled row IDs, token hashes, embeddings, code and raw repeats are saved.']
        for i,line in enumerate(lines): fig.text(.07,.85-i*.047,line,size=11)
        pdf.savefig(fig); plt.close(fig)
        for reference in ['old_replay','pile_heldout']:
            table=mm[mm.reference==reference].set_index('task').loc[tasks]
            fig,axes=plt.subplots(1,3,figsize=(15,9))
            for ax,metric in zip(axes,['token_js','embedding_mmd2_bw1','embedding_domain_classifier_auc']):
                sd=raw[(~raw.is_sampling_floor)&(raw.reference==reference)].groupby('task')[metric].std().reindex(tasks)
                ax.barh(np.arange(16),table[metric],xerr=sd,color=COLORS)
                ax.axvline(floors.loc[reference,metric],color='black',ls='--',label='reference split floor')
                ax.set_yticks(np.arange(16),tasks); ax.invert_yaxis(); ax.set_title(metric); ax.grid(axis='x',alpha=.2); ax.legend(fontsize=8)
            fig.suptitle(f'Dataset distances to {reference}: mean +/- subsample SD'); fig.tight_layout(); pdf.savefig(fig); fig.savefig(out/f'{reference}_distances.png',dpi=160); plt.close(fig)
            for baseline in ['muon','adam']:
                fig,axes=plt.subplots(2,3,figsize=(15,9))
                for col,metric in enumerate(['token_js','embedding_mmd2_bw1','embedding_domain_classifier_auc']):
                    for r,side in enumerate(['new','old']):
                        ax=axes[r,col]; yname=f'pion_minus_{baseline}_{side}_ce'
                        ax.scatter(table[metric],table[yname],c=COLORS,s=35)
                        for t in tasks: ax.annotate(t,(table.loc[t,metric],table.loc[t,yname]),fontsize=6,xytext=(2,3),textcoords='offset points')
                        rho=spearmanr(table[metric],table[yname]).statistic
                        ax.axhline(0,color='gray',ls='--'); ax.set_title(f'{side.title()} CE gap; Spearman {rho:+.2f}')
                        ax.set_xlabel(metric); ax.set_ylabel(f'Pion - {baseline} CE'); ax.grid(alpha=.2)
                fig.suptitle(f'Distance to {reference} vs Pion advantage (below zero is better)'); fig.tight_layout(); pdf.savefig(fig); fig.savefig(out/f'{reference}_vs_{baseline}.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(13,9)); subset=cor[cor.outcome=='pion_minus_muon_new_ce']; pivot=subset.pivot(index='metric',columns='reference',values='spearman_rho')
        im=ax.imshow(pivot,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1); ax.set_yticks(range(len(pivot)),pivot.index); ax.set_xticks(range(2),pivot.columns)
        for i in range(len(pivot)):
            for j in range(2): ax.text(j,i,f'{pivot.iloc[i,j]:+.3f}',ha='center',va='center')
        ax.set_title('All metrics: distance vs Pion - Full New CE (16 topics)'); fig.colorbar(im,ax=ax); fig.tight_layout(); pdf.savefig(fig); fig.savefig(out/'all_metrics_correlations.png',dpi=160); plt.close(fig)
    audit=dict(tasks=len(tasks),references=2,metrics=len(metrics),rows=len(raw),finite=bool(np.isfinite(raw[metrics]).all().all()),all_old_replay_exact=True,
               rbf_bandwidth_reference_median=bw,embedding_model_revision=REV,source_endpoint_rows=len(endpoints))
    dump(out/'audit.json',audit)
    print(json.dumps(audit),flush=True)
    print(cor[(cor.metric.isin(['token_js','embedding_mmd2_bw1','embedding_domain_classifier_auc']))&(cor.outcome=='pion_minus_muon_new_ce')].to_string(index=False),flush=True)


if __name__=='__main__': main()
