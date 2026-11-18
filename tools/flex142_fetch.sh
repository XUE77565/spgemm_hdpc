#!/bin/bash
# FlexSpGEMM 142 测试集下载守护:tamu 文件服 2026-11 从本网络不可达(代理/直连均挂起),
# 每 20 分钟重试一轮,wget -c 断点续传;齐 142 或 60 轮(50h)后退出。日志 results/flex142_fetch.log。
cd "$(dirname "$0")/.." || exit 1
DEST=data/flex142
CSV=external_sota/FlexSpGEMM/data/mat-142.csv
LOG=results/flex142_fetch.log
need() { awk -F, '{print $1"/"$2}' "$CSV" | while read g; do n=${g##*/}; [ -f "$DEST/$n.mtx" ] || echo "$g"; done; }
for round in $(seq 1 60); do
  MISSING=$(need | wc -l)
  echo "[$(date '+%m-%d %H:%M')] round $round: 缺 $MISSING/142" >> $LOG
  [ "$MISSING" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M')] ALL DONE" >> $LOG; exit 0; }
  need | while read gn; do
    grp=${gn%/*}; name=${gn##*/}
    for host in sparse-files.engr.tamu.edu sparse.tamu.edu; do
      wget -c -q -T 90 -t 1 -O "$DEST/$name.tar.gz" "https://$host/MM/$grp/$name.tar.gz" && break
    done
    [ -f "$DEST/$name.tar.gz" ] && tar xzf "$DEST/$name.tar.gz" -C "$DEST" --strip-components=2 2>/dev/null \
      && find "$DEST" -name "*.mtx" -exec mv {} "$DEST/$name.mtx" \; 2>/dev/null && rm -f "$DEST/$name.tar.gz"
  done
  sleep 1200
done
echo "[$(date '+%m-%d %H:%M')] 60 轮到顶退出,仍未齐" >> $LOG
