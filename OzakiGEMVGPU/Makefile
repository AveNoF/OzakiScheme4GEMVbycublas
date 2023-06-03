CXX=g++ -O3
CXXFLA= -std=c++14 
NVCC=nvcc
NVCCFLAGS=   -arch=sm_75 -lcurand -lcublas $(CXXFLAGS)   --compiler-bindir=$(CXX) --expt-relaxed-constexpr
#SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')
#OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
#OBJS:=$(subst .cpp,.o,$(OBJS))
#OBJS:=$(subst .cu,.o,$(OBJS))
TARGET=GEMV

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o  $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $< -c -o $@
all: clean $(OBJS) $(TARGET)

clean:
	rm -rf $(SRCDIR)/%.o
	rm -rf $(TARGET)


































# (1)コンパイラ
#CC  = nvcc 
# (2)コンパイルオプション
#CFLAGS    =  -libstdc++ --expt-relaxed-constexpr
# (3)実行ファイル名
#TARGET  = GEMV
# (4)コンパイル対象のソースコード
#SRCS    = GEMV.cu
# (5)オブジェクトファイル名
#OBJS    = $(SRCS:.cu=.o)

# (6)インクルードファイルのあるディレクトリパス
#INCDIR  = 

# (7)ライブラリファイルのあるディレクトリパス
#LIBDIR  = 

# (8)追加するライブラリファイル
#LIBS    =  -lcublas -03

# (9)ターゲットファイル生成
#$(TARGET): $(OBJS)
#	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

# (10)オブジェクトファイル生成
#$(OBJS): $(SRCS)
#	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

# (11)"make all"で make cleanとmakeを同時に実施。
#all: clean $(OBJS) $(TARGET)
# (12).oファイル、実行ファイル、.dファイルを削除
#clean:
#	-rm -f $(OBJS) $(TARGET) *.d
