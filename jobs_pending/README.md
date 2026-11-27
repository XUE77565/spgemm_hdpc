# jobs_pending/ —— 写好但还没放行的作业

轮询器只扫 `jobs/*/run.sh`,这个目录里的东西不会被执行。凡是可能和对方的计时作业抢卡、抢主机的作业,先放这里,
在 dialog 里问清楚(哪张卡、什么时候),得到放行后再 `git mv jobs_pending/<id> jobs/<id>` 入队(run.sh 里引用的是 `jobs/$ID/…`,移过去不用改)。
作业 id 一旦进过 `jobs/` 就不能再用。
