// S4 决策表(study/60):从标定数据自动生成
// 签名格:bin9_frac(0/low/high) × bin45_frac(0/low/high) × 规模(small/medium/large)
// 生成时间: 自动

#ifndef ROUTING_TABLE_H
#define ROUTING_TABLE_H

// 每格的最优配置(H69/H45 开关)
typedef struct { int h69; int h45; int h55; } route_cfg_t;

// 查表:输入 bin9_frac/bin45_frac/total_rows 带的索引
// 返回该格的最优配置
static inline route_cfg_t routing_table_lookup(
    int bin9_level,  // 0=none 1=low 2=high
    int bin45_level, // 0=none 1=low 2=high
    int size_level)  // 0=small 1=medium 2=large
{
    // 默认:全开(基线)
    route_cfg_t cfg = {1, 1, 1};

    if (bin9_level==2 && bin45_level==2 && size_level==1)
        cfg = (route_cfg_t){1, 1, 1};  // high_high_medium n=2
    if (bin9_level==2 && bin45_level==1 && size_level==1)
        cfg = (route_cfg_t){0, 0, 0};  // high_low_medium n=14
    if (bin9_level==2 && bin45_level==0 && size_level==1)
        cfg = (route_cfg_t){1, 1, 1};  // high_none_medium n=7
    if (bin9_level==1 && bin45_level==2 && size_level==1)
        cfg = (route_cfg_t){0, 0, 0};  // low_high_medium n=2
    if (bin9_level==1 && bin45_level==2 && size_level==0)
        cfg = (route_cfg_t){1, 1, 1};  // low_high_small n=1
    if (bin9_level==1 && bin45_level==1 && size_level==1)
        cfg = (route_cfg_t){1, 1, 1};  // low_low_medium n=12
    if (bin9_level==1 && bin45_level==1 && size_level==0)
        cfg = (route_cfg_t){1, 1, 1};  // low_low_small n=1
    if (bin9_level==1 && bin45_level==0 && size_level==1)
        cfg = (route_cfg_t){0, 0, 0};  // low_none_medium n=5
    if (bin9_level==1 && bin45_level==0 && size_level==0)
        cfg = (route_cfg_t){1, 1, 1};  // low_none_small n=1
    if (bin9_level==0 && bin45_level==2 && size_level==2)
        cfg = (route_cfg_t){1, 1, 1};  // none_high_large n=7
    if (bin9_level==0 && bin45_level==2 && size_level==1)
        cfg = (route_cfg_t){1, 1, 1};  // none_high_medium n=144
    if (bin9_level==0 && bin45_level==2 && size_level==0)
        cfg = (route_cfg_t){1, 1, 1};  // none_high_small n=28
    if (bin9_level==0 && bin45_level==1 && size_level==2)
        cfg = (route_cfg_t){0, 1, 1};  // none_low_large n=4
    if (bin9_level==0 && bin45_level==1 && size_level==1)
        cfg = (route_cfg_t){1, 1, 1};  // none_low_medium n=48
    if (bin9_level==0 && bin45_level==1 && size_level==0)
        cfg = (route_cfg_t){0, 0, 0};  // none_low_small n=2
    if (bin9_level==0 && bin45_level==0 && size_level==2)
        cfg = (route_cfg_t){1, 1, 1};  // none_none_large n=19
    if (bin9_level==0 && bin45_level==0 && size_level==1)
        cfg = (route_cfg_t){0, 1, 1};  // none_none_medium n=54
    if (bin9_level==0 && bin45_level==0 && size_level==0)
        cfg = (route_cfg_t){1, 0, 0};  // none_none_small n=58

    return cfg;
}

#endif // ROUTING_TABLE_H
