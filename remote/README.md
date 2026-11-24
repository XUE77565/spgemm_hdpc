# B 上的实验轮询脚本（exp-poller）

A（我们这边）把实验作业 push 到一个私有 GitHub 仓库。B 上的脚本每分钟拉一次仓库，看到新作业就把那一版代码拷到服务器 C，在 C 上后台运行；跑的过程中和跑完以后，把状态、日志末尾和小结果文件 push 回仓库的 `results` 分支。

- 没有人 SSH 进 B。B 只主动连 GitHub 和 C，不开任何端口。
- B 自己不执行仓库里的任何内容。它只用 git 读文件，把代码快照打包传到 C，然后在 C 上启动 `run.sh`。`job.env` 只按固定格式解析，不会被 `source`。
- C 上的作业用 `nohup` 和 `setsid` 在后台跑，B 关机或断网也不影响。B 恢复后会接着收集结果。

## 文件

| 文件 | 放在哪 | 作用 |
|---|---|---|
| `b/poller.sh` | B | 轮询脚本 |
| `b/poller.env.example` | B | 配置模板，复制成 `~/.exp-poller/poller.env` 再改 |
| `a/exp.sh` | A | 提交、查看、取消作业的辅助脚本 |

B 需要：bash、git、ssh、tar。Linux、macOS 都行；Windows 要用 Git Bash，并在配置里设 `SSH_MUX=0`。
C 需要：bash、tar、GNU `timeout`。`setsid` 最好也有。普通 Linux 服务器都有这些。

## B 上的一次性设置（B 的主人做，大约 10 分钟）

1. **生成一把只给这个仓库用的 deploy key**
   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/exp_deploy_key -C exp-poller
   ```
   把 `~/.ssh/exp_deploy_key.pub` 的内容加到 GitHub 仓库的 Settings → Deploy keys，勾选 **Allow write access**。这把钥匙只能读写这一个仓库。

2. **确认 B 能免密登录 C。** 在 `~/.ssh/config` 里给 C 起个别名，比如：
   ```
   Host c-server
       HostName <C 的地址>
       User <B 主人在 C 上的用户名>
       IdentityFile ~/.ssh/<登录 C 用的钥匙>
   ```
   然后测试 `ssh -o BatchMode=yes c-server true`，不弹密码就行。
   C 的 `~/.bashrc` 不能在非交互登录时打印文字，否则结果传输会坏。

3. **放脚本、写配置**
   ```bash
   mkdir -p ~/.exp-poller
   cp poller.sh ~/.exp-poller/ && chmod +x ~/.exp-poller/poller.sh
   cp poller.env.example ~/.exp-poller/poller.env
   # 编辑 poller.env：REPO_URL、GIT_KEY、C_HOST、C_BASE
   ```

4. **初始化并自检**
   ```bash
   ~/.exp-poller/poller.sh init    # 克隆仓库，建 results 分支；仓库里已有的作业标成 skipped，不会运行
   ~/.exp-poller/poller.sh check   # 测 git 和 SSH，列出 C 上的工具和磁盘空间
   ```

5. **让它一直跑**，三选一：
   - cron（Linux/macOS）：`crontab -e` 加一行
     ```
     * * * * * $HOME/.exp-poller/poller.sh once >/dev/null 2>&1
     ```
   - tmux 或 nohup：`nohup ~/.exp-poller/poller.sh loop >/dev/null 2>&1 &`
   - Windows：任务计划程序里每分钟用 Git Bash 运行 `poller.sh once`。

   脚本自带锁，cron 和 loop 同时开也不会重复运行。

停用：删掉 cron 那行，或者结束 loop 进程。日志在 `~/.exp-poller/poller.log`。

## 作业格式（A 这边写）

在仓库的 `main` 分支上：
```
jobs/<id>/run.sh     必须有。在 C 上执行，当前目录是这一版代码的根目录
jobs/<id>/job.env    可选，只认两行：
                       TIMEOUT=6h        超时，格式 30m / 6h / 2d，默认 24h
                       COMMIT=<sha>      指定用哪一版代码，默认用 B 发现作业时 main 的最新一版
jobs/<id>/CANCEL     有这个文件就取消：排队中的不再运行，运行中的会被结束
```
- `<id>` 只能用字母、数字和 `. _ -`，最长 64 个字符，并且不能重复使用。
- `run.sh` 里能用这几个环境变量：
  - `JOB_ID`
  - `JOB_DIR`：C 上的 `<C_BASE>/<id>`
  - `OUT_DIR`：要传回来的小文件，比如 metrics.json、csv、图，都写到这里
- 大数据集、checkpoint 放在 C 上的固定位置，不要放进 `OUT_DIR`。超过大小上限的文件不会上传，文件名会记到 `out_skipped.txt`。上限默认单个 2 MB、每个作业总共 10 MB。
- 作业默认一个一个跑，按提交顺序排队。要同时跑几个就调 `MAX_PARALLEL`，选哪块 GPU 在 `run.sh` 里用 `CUDA_VISIBLE_DEVICES` 指定。

C 上每个作业的目录：`code/`（代码快照）、`out/`、`log.txt`（完整日志）、`pid`、`tpid`、`start`、`end`、`exit`、`runner.sh`。脚本不会自动删任何东西。

## 结果（A 这边读）

- `results` 分支：`results/<id>/status.json`、`log_tail.txt`（最后 200 行）、`out/`、`out_skipped.txt`。只有 B 往这个分支写。
- `heartbeat` 分支：`heartbeat.txt` 每 10 分钟更新一次，写着时间、C 能不能连上、正在跑和排队的作业。这个分支每次强制覆盖，不会越积越多。
- 状态（state）有这几种：
  - `queued`：排队中
  - `running`：运行中
  - `done`：退出码 0
  - `failed`：退出码非 0
  - `timeout`：超时
  - `cancelled`：已取消
  - `lost`：进程没了却没有退出码，比如 C 重启
  - `error`：启动失败，或指定的版本不存在
  - `skipped`：init 之前就有的作业，不运行

A 上的用法（在仓库目录里）：
```bash
exp.sh submit exp001 my_run.sh 6h   # 提交
exp.sh status                       # 心跳加所有作业的状态
exp.sh wait exp001                  # 等它跑完
exp.sh log exp001                   # 日志末尾
exp.sh pull exp001 ./res/exp001     # 把结果拷到本地
exp.sh cancel exp001                # 取消
```

## 安全上要知道的

- 能往这个仓库 `main` 分支 push 的人，就能用 B 主人在 C 上的账号跑任意命令，这是这个方案的本意。所以仓库必须私有，B 上的 deploy key 只加在这一个仓库上。
- 日志和结果会进私有仓库，`run.sh` 里不要打印密码或 token。
- C 的登录钥匙一直留在 B 上，不进仓库，也不经过 A。

## 本地测试（2026-11-24，在 A 上模拟 B 和 C）

用本地裸仓库代替 GitHub，用一个假 ssh 代替 C，测过这些情况：
- 正常完成，输出文件回传，大文件被跳过并记录；
- 超时；
- 非 0 退出；
- 排队时取消、运行中取消，进程都被清干净；
- 不合法的作业名会被忽略；
- 两个轮询同时启动，只有一个在干活；
- C 连不上时保持状态，恢复后继续；
- C 上的进程被杀，状态记为 `lost`；
- 两个作业并行；
- 按提交顺序排队。

还没在真的 GitHub 和 C 上跑过。
