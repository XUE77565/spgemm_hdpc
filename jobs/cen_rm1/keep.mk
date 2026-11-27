# 附加 makefile(run.sh 用 make -f Makefile -f 本文件):让 .o 不被当中间文件删掉,
# WRITE_MTX 件只重编 main.cu(WRITE_MTX 只影响 main.cu)。编译/链接命令与裸 Makefile 完全相同。
.SECONDARY:
