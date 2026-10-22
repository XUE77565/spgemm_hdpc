#!/usr/bin/env bash
# T-B1 SASS 静态对账:hash 梯实例逐实例指令构成(我方 vs ocean)
# 用法: bash scripts/sass_account.sh  (读 /tmp/ours.sass /tmp/ocn.sass)
# ⚠ mawk 1.3.4 错编译陷阱:动作块内 ≥2 语句 + next 会静默丢计数——本脚本两段式
#   单语句规则规避(切片→直方图),勿改回单进程多语句形态。
set -e
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
acct() {  # $1=sass文件 $2=符号子串 -> 一行 csv
  awk -v sym="$2" 'index($0, "Function :") > 0 { inf = index($0, sym) > 0 } inf { print }' "$1" > "$TMP"
  awk '
    /^[ \t]*\/\*[0-9a-f][0-9a-f][0-9a-f][0-9a-f]\*\// {
      line = $0
      sub(/^[ \t]*\/\*[0-9a-f][0-9a-f][0-9a-f][0-9a-f]\*\/[ \t]*/, "", line)
      split(line, w, /[ \t]/)
      op = (w[1] ~ /^@!?P/) ? w[2] : w[1]
      if (op != "") { n++; c[op]++ }
    }
    END {
      br = c["BRA"]; bssy = c["BSSY"]; bsync = c["BSYNC"]; brk = c["BREAK"]
      isetp = c["ISETP"]; atom = 0; ld = 0; st = 0; dfp = 0; shfl = 0; vote = 0
      for (k in c) {
        if (k ~ /^ATOM/) atom += c[k]
        if (k ~ /^LDG|^LDS|^LDC|^LDLU|^LDSTS/) ld += c[k]
        if (k ~ /^STG|^STS|^STL/) st += c[k]
        if (k ~ /^DADD|^DMUL|^DFMA|^DSETP|^F2F|^I2F|^F2I/) dfp += c[k]
        if (k ~ /^SHFL/) shfl += c[k]
        if (k ~ /^VOTE/) vote += c[k]
      }
      printf "%s,total=%d,BRA=%d,BSSY=%d,BSYNC=%d,BREAK=%d,branchmach=%d,ISETP=%d,ATOM=%d,LD=%d,ST=%d,DFP=%d,SHFL=%d,VOTE=%d\n",
        "'"$2"'", n+0, br+0, bssy+0, bsync+0, brk+0, br+bssy+bsync+brk+0, isetp+0, atom+0, ld+0, st+0, dfp+0, shfl+0, vote+0
    }' "$TMP"
}
echo "=== 我方( ours.sass )==="
for s in "hash_spa_kernelILi0ELi1ELi0ELi0ELi0ELi0EE" "hash_spa_kernelILi0ELi1ELi0ELi1ELi0ELi0EE" \
         "hash_spa_kernelILi0ELi1ELi0ELi0ELi9ELi0EE" "hash_spa_kernelILi0ELi1ELi0ELi0ELi10ELi0EE" \
         "hash_spa_kernelILi0ELi1ELi0ELi0ELi11ELi0EE" "hash_spa_kernelILi0ELi1ELi0ELi0ELi12ELi0EE" \
         "hash_spa_kernelILi0ELi1ELi1ELi0ELi0ELi2719EE" "hash_spa_batched_kernelILi128ELi8EE" \
         "hash_spa_batched_kernelILi512ELi4EE"; do acct /tmp/ours.sass "$s"; done
echo "=== ocean( ocn.sass )==="
for s in "hashNumericKernelILi6ELi331ELb0ELb0EjEE" "hashNumericKernelILi7ELi673ELb0ELb0EjEE" \
         "hashNumericKernelILi8ELi1327ELb0ELb0EjEE" "hashNumericKernelILi9ELi2719ELb0ELb0EjEE" \
         "hashNumericKernelILi10ELi5449ELb0ELb0EjEE" "hashNumericKernelILi10ELi16253ELb0ELb1EjEE" \
         "hashNumericSubWarpKernelILi4ELi80EEE" "hashNumericSubWarpKernelILi5ELi160EEE"; do acct /tmp/ocn.sass "$s"; done