#include <cstdint>
#include <cmath>
extern "C" void encode(const double* x, int64_t n, int bits, double full, int32_t* q, double* r) {
 int lim=(1<<(bits-1))-1, prev=0; double d=full/lim;
 for(int64_t i=0;i<n;i++){ int step=(int)std::trunc((x[i]-prev*d)/d); if(step>8)step=8;if(step<-8)step=-8;prev+=step;if(prev>lim)prev=lim;if(prev<-lim)prev=-lim;q[i]=prev;r[i]=prev*d; }
}
