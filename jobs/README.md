# 作业协议与机器约定(A 侧必读)

> 你(A=外部模型)往本仓库 `main` 分支推 `jobs/<id>/run.sh`,B 侧 poller(每分钟轮询)会把
> 当时 HEAD 的代码快照送到 C 上执行,结果经 `results` 分支回传。完整协议见 remote/README.md。
> 任务书 = study/70;本文件只讲**这台 C 的机器事实与项目纪律**。

## C 的机器事实

- H100-187,**2× H100 PCIe 80GB**;作业以 `xueyizhou` 身份跑在 `~/exp-runs/<id>/code/`(提交快照,与
  `/home/xueyizhou/spgemm-challenge` 工作树完全隔离,**永远不要去碰工作树和认证件 spgemm_test**)
- 作业串行(MAX_PARALLEL=1),默认超时 24h,`job.env` 里 `TIMEOUT=30m/6h/2d` 可覆盖;取消 = 推
  `jobs/<id>/CANCEL` 空文件
- 回传:`OUT_DIR` 里小文件(单件 2MB、每作业共 10MB 封顶,超限记 out_skipped.txt)+ 日志末尾 200 行;
  读结果 = `results` 分支 `results/<id>/status.json|log_tail.txt|out/`;poller 活性 = `heartbeat` 分支
- `run.sh` 可用环境变量:`JOB_ID`、`JOB_DIR`、`OUT_DIR`;cwd = 代码快照根目录

## GPU 纪律(红线)

1. **run.sh 必须钉 `CUDA_VISIBLE_DEVICES=0` 或 `1`**并在作业说明里写明用了哪张卡。
   本地(用户侧)计时批可能占另一张;**confirm/ab_pair 级计时作业的双臂必须同卡同批**,
   与历史基线比对的作业要注明卡号(H100 PCIe ×2,同型号)。
2. **计时纪律**(<1ms 中位不作单阵裁决;A/B 交错发射;测量分级判读见 scripts/ab_pair.py)。
3. **报错即停**:kernel 报错不重试掩盖,日志里保留完整报错文本。

## 数据与磁盘(红线)

- 测试阵绝对路径:`/home/xueyizhou/spgemm-challenge/data/`(`ocean/square/`+`first100/`= 428 主账,
  **永不参与训练/标定**;`flex142` 同为测试账)。大数据**不进 OUT_DIR**,路径引用即可。
- **WRITE_MTX 转储即清**:批跑后 `rm -rf` 作业目录里的 `results/matrices/`,磁盘曾被写满三次。
- 作业目录不自动清理;大批量产物自行控制在 GB 级以内并在日志里报告 df。

## 构建与验证协议(每刀标准流程,study/70 §六/§七)

- `make TARGET=别名`(绝不裸 `make`——会覆盖根目录认证件);改 `.h` 后 `touch` 对应 `.cu`;
  构建自动跑符号审计,必须双绿
- 值级:`WRITE_MTX=1` 代表阵 dump + `python3 scripts/verify_sorted_truth.py` 三判(L1=L2=0)
  ——**nnz 对 ≠ 输出对**;电池:`python3 scripts/ab_pair.py <A件> <B件> out.csv 3 0`
- 红不 commit:电池红/值级红 = 停下写明原因;绿 → commit(说明与原形态关系)→ confirm 全量
- 新 kernel:微复现器先行(tools/hh_repro*.cu 模式),全绿才进生产

## 通信惯例

- 作业 id 用 `字母数字._-`(≤64 字符,不可复用);一次推一个作业,别删别人(包括 smoke1)的 results
- 需要用户裁决的事(改门常数、动测试账、新依赖安装)不要自行实施——写成 `OUT_DIR/NEED_USER.md` 回传
- **技术问答/裁决材料交换**:`dialog/` 目录(与本机 Claude 的异步通道,协议见 `dialog/README.md`;只问答不指挥,裁决件仍归用户)
