def dl(p):
    for line in open(p):
        if line.startswith('%'): continue
        t=line.split()
        if len(t)>=3: yield int(t[0]),int(t[1]),float(t[2])
import sys
m=sys.argv[1]
gmax=0.0; nbc=0; unord=0; prev_r=prev_c=-1; pend=[]
for (r1,c1,v1),(r2,c2,v2) in zip(dl(m+'_off.mtx'),dl(m+'_on.mtx')):
    gmax=max(gmax,abs(v1),abs(v2))
    if r1!=r2 or c1!=c2: nbc+=1
    if r1==prev_r and c1<=prev_c: unord+=1
    prev_r,prev_c=r1,c1
    if v1!=v2: pend.append(abs(v1-v2))
worst=(max(pend) if pend else 0.0)/gmax
print(f"{m}: entries_checked col_mismatch={nbc} unordered={unord} text_diff={len(pend):,} worst_rel_vs_global={worst:.3e}")
print("=> "+("PASS" if nbc==0 and unord==0 and worst<1e-9 else "FAIL"))
