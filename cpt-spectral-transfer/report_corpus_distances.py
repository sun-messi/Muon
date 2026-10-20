"""Readable results-first report from saved corpus-only distance measurements."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox


def labels(ax, xs, ys, ids):
    ax.figure.canvas.draw(); occupied=[]
    for x,y,n in zip(xs,ys,ids):
        px,py=ax.transData.transform([x,y]); best=None
        for radius in [10,20,32,46,62,80]:
            for theta in np.linspace(0,2*np.pi,12,endpoint=False):
                dx,dy=radius*np.cos(theta),radius*np.sin(theta)
                box=Bbox.from_bounds(px+dx-11,py+dy-8,22,16)
                overlap=sum(box.overlaps(b) for b in occupied)
                bounds=ax.get_window_extent()
                outside=not(bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1))
                score=10000*overlap+10000*outside+radius
                if best is None or score<best[0]: best=(score,dx,dy,box)
        _,dx,dy,box=best; occupied.append(box)
        ax.annotate(str(n),(x,y),xytext=(dx,dy),textcoords='offset pixels',ha='center',va='center',fontsize=8,
                    bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.85),
                    arrowprops=dict(arrowstyle='-',color='.6',lw=.5))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); args=ap.parse_args(); p=args.directory
    d=pd.read_csv(p/'distances_and_optimizer_gaps.csv'); c=pd.read_csv(p/'correlations.csv'); tasks=list(d.task.drop_duplicates())
    names={t:i+1 for i,t in enumerate(tasks)}; metrics=['token_js','embedding_mmd2_bw1']
    sensitivity=[]
    for ref,df in d.groupby('reference'):
        for metric in metrics:
            x=rankdata(df[metric]); y=rankdata(df.pion_minus_muon_new_ce); z=rankdata(df.initial_new_ce)
            design=np.c_[np.ones(len(z)),z]; xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0]; yr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
            sensitivity.append(dict(reference=ref,metric=metric,partial_rank_rho_controlling_initial_new_ce=np.corrcoef(xr,yr)[0,1],
                rho_gap_divided_by_initial_new_ce=spearmanr(df[metric],df.pion_minus_muon_new_ce/df.initial_new_ce).statistic))
    pd.DataFrame(sensitivity).to_csv(p/'scale_sensitivity.csv',index=False)
    with PdfPages(p/'corpus_distance_results.pdf') as pdf:
        fig=plt.figure(figsize=(12,9)); fig.text(.06,.95,'Dataset distance and optimizer performance',size=21,weight='bold')
        lines=['16 topics, one training seed; existing AdamW / Full Muon / Pion runs, no retraining.',
               'Further from Pile -> smaller Pion New advantage (larger Pion - Full New CE).',
               'Spearman: token JS +0.909; embedding MMD +0.915; classifier AUC +0.943.',
               'Old CE gaps do NOT follow the same pattern: JS +0.159; MMD +0.156.',
               'Distances use data only. These are exploratory associations, not proof of noise filtering.']
        for i,line in enumerate(lines): fig.text(.06,.89-i*.035,line,size=11)
        df=d[d.reference=='pile_heldout'].sort_values('token_js')
        table=[]
        for _,r in df.iterrows(): table.append([names[r.task],r.task,f'{r.token_js:.3f}',f'{r.embedding_mmd2_bw1:.4f}',f'{r.pion_minus_muon_new_ce:+.4f}',f'{r.pion_minus_muon_old_ce:+.4f}'])
        ax=fig.add_axes([.05,.13,.9,.55]); ax.axis('off')
        tab=ax.table(cellText=table,colLabels=['ID','New dataset','Token JS','MMD squared','New CE gap','Old CE gap'],loc='center',cellLoc='center',colWidths=[.06,.29,.14,.17,.17,.17]); tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1,1.5)
        fig.text(.06,.055,'Reference: weighted Pile held-out proxy. CE gap = Pion - Full; negative means Pion is better.',size=10)
        pdf.savefig(fig); fig.savefig(p/'results_overview.png',dpi=170); plt.close(fig)
        fig=plt.figure(figsize=(12,9)); fig.text(.06,.94,'Measurement and robustness checks',size=21,weight='bold')
        lines=['References: actual 3,904-window Old replay pool; 977-window weighted Pile held-out pool.',
          'Both approximate distributions; neither reconstructs Pythia\'s complete pretraining corpus.',
          'Each corpus: 768 saved windows. Five repeats compare 384 windows per side at 1,024 tokens.',
          'Reference-vs-reference uses disjoint halves. Token JS floors: Pile 0.274; Old replay 0.291.',
          'Token JS / Hellinger / total variation / cosine; hashed character 3-gram JS; format distance.',
          'MiniLM: 3-bandwidth MMD; energy distance; centroid cosine; PCA64 Frechet; classifier AUC.',
          'Frozen encoder: all-MiniLM-L6-v2; mean pooling, L2 normalization, fixed model revision.',
          'Encoder sees middle 256 Pythia tokens, capped at 256 wordpieces; lexical counts use 1,024.',
          'MMD bandwidth and PCA are reference-only. AUC uses a held-out 30% window split.',
          'All 16 cached training tapes have exactly the same Old replay tokens (verified).',
          'Pile-reference New is closest but not zero distance: sampling and document selection matter.',
          'Pile New-gap JS rho remains 0.846 after excluding ChEMBL/PDB/UniProt/CVE.',
          'Leave-one-topic-out JS rho: 0.889-0.925; MMD rho: 0.896-0.925.',
          'Permutation p=0.0002 for both; BH q=0.000743 across 104 exploratory tests.',
          '13 distances are related measurements, not 13 independent confirmations.',
          'SD reflects window subsampling, not training-seed uncertainty or independent documents.',
          'Classifier AUC saturates for structured data; ordinary English embeddings have domain limits.',
          'Data sources were selected exploratorily. Do not infer a universal threshold or causality.',
          'Old retention is a separate outcome: Pion beats Full on 15/16 Old endpoints; ORD is the exception.',
          'Raw repeats, row IDs, hashes, encoder features and all optimizer endpoint sources are saved.']
        for i,line in enumerate(lines): fig.text(.06,.87-i*.037,line,size=10.5)
        pdf.savefig(fig); plt.close(fig)
        for ref in ['pile_heldout','old_replay']:
            table=d[d.reference==ref].set_index('task').loc[tasks]
            for base in ['muon','adam']:
                fig,axes=plt.subplots(2,2,figsize=(14,10)); fig.subplots_adjust(left=.07,right=.78,top=.88,bottom=.09,hspace=.30,wspace=.25)
                for i,side in enumerate(['new','old']):
                    for j,metric in enumerate(metrics):
                        ax=axes[i,j]; y=f'pion_minus_{base}_{side}_ce'
                        ax.scatter(table[metric],table[y],s=35,color=['#e66101' if t in ['chembl','pdb','uniprot','cve'] else '#377eb8' for t in tasks])
                        ax.margins(.15); ax.axhline(0,color='.4',ls='--',lw=1); ax.grid(alpha=.2)
                        ax.set_xlabel('Token JS distance' if j==0 else 'Embedding MMD squared'); ax.set_ylabel(f'Pion - {base} {side.title()} CE')
                        rho=spearmanr(table[metric],table[y]).statistic; ax.set_title(f'{side.title()} outcome; Spearman {rho:+.3f}')
                        labels(ax,table[metric],table[y],[names[t] for t in tasks])
                fig.suptitle(f'Distance to {ref}: Pion vs {"Full Muon" if base=="muon" else "AdamW"}',size=17)
                fig.text(.81,.86,'Dataset IDs',weight='bold',size=12)
                for i,t in enumerate(tasks): fig.text(.81,.82-i*.038,f'{names[t]:2d}  {t}',size=10)
                fig.text(.81,.15,'CE gap < 0:\nPion is better.\n\nOrange: structured four.\nBlue: other topics.',size=10)
                pdf.savefig(fig); fig.savefig(p/f'{ref}_vs_{base}_readable.png',dpi=180); plt.close(fig)
        # Keep the full metric/reference correlation overview in the results report.
        fig,axes=plt.subplots(1,2,figsize=(14,9)); pivot=c[c.outcome=='pion_minus_muon_new_ce'].pivot(index='metric',columns='reference',values='spearman_rho')
        for ax,side in zip(axes,['new','old']):
            pv=c[c.outcome==f'pion_minus_muon_{side}_ce'].pivot(index='metric',columns='reference',values='spearman_rho').reindex(pivot.index)
            im=ax.imshow(pv,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto'); ax.set_yticks(range(len(pv)),pv.index,fontsize=8); ax.set_xticks([0,1],pv.columns,rotation=15); ax.set_title(f'{side.title()} gap: Pion - Full')
            for i in range(len(pv)):
                for j in range(2): ax.text(j,i,f'{pv.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=9)
        fig.suptitle('All 13 metrics: Spearman correlation across 16 topics'); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    for file in ['analyze_corpus_distances.py','report_corpus_distances.py']:
        shutil.copy2(Path(__file__).parent/file,p/file)
    print(p/'corpus_distance_results.pdf')


if __name__=='__main__': main()
