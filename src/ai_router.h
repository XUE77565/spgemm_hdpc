// AI Router v2: physics + learned residual (log-space, standardized features)
// Trained: 16350 updates

#define AI_NF 10
static const float ai_mean[10] = {14.5360f,17.6384f,-11.4336f,3.1024f,9.2691f,1.7416f,0.0905f,0.3676f,0.0343f,0.1963f};
static const float ai_std[10] = {3.5889f,5.2430f,3.0647f,2.3581f,2.1136f,1.5879f,0.2427f,0.3917f,0.1238f,0.3221f};
static const float ai_W[4][10] = {
  {-0.3300f,-0.3370f,0.1964f,-0.2470f,-2.2380f,0.0996f,0.3190f,-1.0819f,0.4907f,-0.8613f},
  {-0.2998f,-0.1059f,0.5209f,0.2207f,-3.5039f,-0.0373f,-0.1152f,0.1039f,0.2363f,-0.1913f},
  {-0.8717f,-0.7115f,0.8243f,-0.2553f,-2.4588f,0.1232f,-0.0303f,-0.0828f,0.4298f,-0.5639f},
  {-0.2734f,-0.2585f,0.1981f,-0.1586f,-1.9208f,0.0098f,0.0499f,-1.0807f,0.2858f,-0.7914f},
};
static const float ai_b[4] = {1.4032f,1.7955f,1.9055f,1.5316f};

static inline int ai_select(const float* feat, const float* phys_log) {
    float xp[10]; for (int i=0;i<10;i++) xp[i]=(feat[i]-ai_mean[i])/ai_std[i];
    float best=1e30f; int best_v=0;
    for (int v=0;v<4;v++) {
        float c = phys_log[v] + ai_b[v];
        for (int i=0;i<10;i++) c += ai_W[v][i]*xp[i];
        if (c<best) {best=c;best_v=v;}
    }
    return best_v;
}