# study/47:CD 臂巨输出 int/size_t 溢出雷(c-73/c-73b,10-15 权威批抓获)

**现象**(batch437,[183]+[191],sidecar: prof/c-73__CD.log):
```
[hash] C_nnz=-778955041 (est=3058565145, -3.93x over-alloc)
[hash] compact in_place=0 (min_margin=-2131672864)
[mempool] cudaMalloc(18446744064362768824 B) failed: out of memory   ← size_t 下溢
```
**死因链**:est 配额(int 域内 3.06e9)→ C_nnz/margin 某环节 int32 溢出成负 →
compact 路径拿负数算尺寸 → size_t 下溢 1.8e19 → cudaMalloc 必败 → NOPROF。
**证书失灵**:溢出证书本应在 est/actual 失配时判废回退 merge3——本例负值绕过了
判废条件(actual>alloc 比较在负数域失效)。

**对照**:C 臂同阵成功(84.8ms)→ 雷在 DCEST 配额/compact 链,非共享路径。
**人群**:c-73 级巨输出阵(337 批"10 缺失阵"同族——can_73/c-73 相关性再+1);
批内预计后续大阵还有实例,同雷不重复查因。

**处置**:
- 批继续(CD=NOPROF 记账,分析时 CD=DNF,oracle 取他臂;不影响批完整性);
- 修雷排 GPU-gated 队列(与 launch 间隙实测并行不冲突):①C_nnz/margin 全链
  int→long long ②compact 尺寸 size_t 卫兵 ③证书判废条件加"负值/溢出即判废"。
