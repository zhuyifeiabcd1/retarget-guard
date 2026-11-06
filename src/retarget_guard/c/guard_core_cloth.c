/* Standalone C99 mesh guard. Row-major doubles, metres, radians.
 * Compile with the package build helper or a C99 compiler.
 * No torch, BLAS, worker threads, or allocation in guard_step.
 * Bind local rotations are followed by input rotations, then Exp(delta).
 * The host supplies semantic joints 0..21; fingers keep their input rotation.
 * L5 consumes a window through t+5 (167 ms at 30 Hz); source observations
 * are cached in chronological order, once per arriving frame.
 *
 * Cloth layer (guard_core.c plus this): every body vertex carries its rest-pose
 * cloth offset. For arm->torso and arm->leg pairs the surface the arm must stay
 * outside is lifted by lift=(1-kappa)*offset at the nearest body vertex, so the
 * allowed depth from the cloth is soft budget + kappa*offset. The search reach
 * grows by the part's largest lift. kappa=1 is guard_core.c bit for bit.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define JMAX 128
#define SMAX 8192
#define HASH 1024
#define CAND 4
#define CELL 0.1
#define PMAX 128
#define N 63
#define MMAX 384
#define WIN 6
#define CFG 55 /* entries of Settings.array(); guard_cfg_count() lets the host check it */

typedef struct {
    int nj, ns, np, par[JMAX], sem[22], slot[JMAX], wi[SMAX][4];
    int part[SMAX], query[SMAX], pairs[PMAX][3], members[2][14][SMAX], count[2][14];
    unsigned char ancestor[JMAX][21], used[2][14][JMAX];
    double lb[JMAX][9], lo[JMAX][3], rb[JMAX][9], rj[JMAX][3];
    double local[SMAX][4][3], normal[SMAX][4][3], w[SMAX][4], soft[SMAX];
    double low[2][14][JMAX][3], high[2][14][JMAX][3], depth[14], arm_extra[2];
    int sparse[SMAX];
    double offset[SMAX], max_offset[14];
    unsigned char clothed[PMAX];
} Pack;
typedef struct {
    double r[JMAX][9], j[JMAX][3], axes[21][9];
    double pb[SMAX][4][3], c[SMAX][3], normal[SMAX][3];
    double low[2][14][3], high[2][14][3];
    unsigned char ready[SMAX], normal_ready[SMAX], near[PMAX];
    int queries, dense_pairs;
    int hint[14][SMAX], head[14][HASH], next[SMAX], cell[SMAX][3], skinned, pairs_near;
} Geo;
typedef struct {
    double gap[PMAX], clear[PMAX], value[PMAX][CAND], normal[PMAX][CAND][3];
    int a[PMAX][CAND], b[PMAX][CAND], ga[PMAX], gb[PMAX];
} Dist;
/* Same joint priors, activation, gated pull and box QP as guard_core3. */
typedef struct {
    Pack *p, *src;
    double cfg[CFG], prev[N], prev2[N], velocity[N], oldallow[PMAX], urg;
    double allow[WIN][PMAX]; unsigned char contact[WIN][PMAX], pull[WIN][PMAX];
    unsigned int history[PMAX]; int runs[PMAX], frame, available, last_a[PMAX], last_b[PMAX];
    Geo current, ref, future, future_ref, source, old_ref;
    double activation[2][PMAX][CAND], pull_weight[2][PMAX];
    double smooth_a[2][PMAX][CAND][N], smooth_b[2][PMAX][CAND];
    double old_ee[10][3], old_ee2[10][3];
    Dist dist;
    double H[N*N], g[N], A[MMAX*N], b[MMAX], weight[MMAX], x[N];
    int rows;
} Guard;
enum {KCFG,SWEEPS,WVEL,WACC,WPEN,FPS,ZERO,XI,INFLUENCE,MARGIN,RHO,WHAND,WROT,WFOOT,WCONTACT,SPRING,EPS,FADE,PULLTRAVEL,ROWALPHA,CALSCALE,SRCSCALE,POSOFFSET,POSVEL,POSACC,WRES,SPINECOST,SPINELIMIT,NECKLIMIT,HIPLIMIT,WRISTLIMIT,ARMLIMIT,LEGLIMIT,DENSITY,DENSENEAR,LIMITGAIN,MAXSPEED,MAXACC,SLACKCAP,HANDRELAX,CLEARANCEBAND,RELEASE,SNAP,SOURCEALLOW,RHOHAND,RHOFOREARM,SETTLE,URGD0,URGD1,URGTAU,URGSMOOTH,URGSPEED,URGACC,URGSLACK,KAPPA};
static const double joint_w[21]={2,2,8,2,2,8,3,3,8,3,3,8,4,4,8,1,1,1,1,2,2};
static const int ee[10]={7,8,10,11,20,21,16,17,18,19};
static inline double min2(double a,double b){return a<b?a:b;}
static inline double max2(double a,double b){return a>b?a:b;}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double dot(const double *a,const double *b,int n) {double x=0;for(int i=0;i<n;i++)x+=a[i]*b[i];return x;}
static void mm(const double *a,const double *b,double *o) {
    double t[9]; for(int i=0;i<3;i++)for(int j=0;j<3;j++)t[3*i+j]=a[3*i]*b[j]+a[3*i+1]*b[3+j]+a[3*i+2]*b[6+j]; memcpy(o,t,sizeof(t));
}
static void mv(const double *a,const double *b,double *o) {for(int i=0;i<3;i++)o[i]=dot(a+3*i,b,3);}
static void exp_jr(const double *v,double *r,double *jr) {
    double t2=dot(v,v,3),t=sqrt(t2),a,b,c;
    if(t2<1e-8){a=1-t2/6+t2*t2/120;b=.5-t2/24+t2*t2/720;c=1./6-t2/120+t2*t2/5040;}
    else{a=sin(t)/t;b=(1-cos(t))/t2;c=(t-sin(t))/(t2*t);}
    double k[9]={0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0},kk[9];mm(k,k,kk);
    for(int i=0;i<9;i++){r[i]=(i%4==0)+a*k[i]+b*kk[i];if(jr)jr[i]=(i%4==0)-b*k[i]+c*kk[i];}
}
void *guard_pack(int nj,int ns,int np,const int *par,const int *sem,const double *lb,const double *lo,
    const double *rb,const double *rj,const double *vertices,const int *wi,const double *w,
    const double *normals,const double *soft,const int *part,const int *query,const int *pairs,const double *depth,const double *offset) {
    if(nj<22||nj>JMAX||ns<1||ns>SMAX||np<1||np>PMAX)return NULL;
    Pack *p=calloc(1,sizeof(Pack));if(!p)return NULL;p->nj=nj;p->ns=ns;p->np=np;
    memcpy(p->par,par,nj*sizeof(int));memcpy(p->sem,sem,22*sizeof(int));
    memcpy(p->lb,lb,nj*9*sizeof(double));memcpy(p->lo,lo,nj*3*sizeof(double));
    memcpy(p->rb,rb,nj*9*sizeof(double));memcpy(p->rj,rj,nj*3*sizeof(double));
    memcpy(p->pairs,pairs,np*3*sizeof(int));memcpy(p->depth,depth,14*sizeof(double));
    for(int j=0;j<nj;j++){p->slot[j]=-1;if(par[j]>=j||par[j]<-1)goto bad;}
    for(int k=0;k<22;k++){if(sem[k]<0||sem[k]>=nj)goto bad;for(int a=0;a<k;a++)if(sem[k]==sem[a])goto bad;if(k)p->slot[sem[k]]=k-1;}
    for(int j=0;j<nj;j++)for(int b=j;b>=0;b=par[b])if(p->slot[b]>=0)p->ancestor[j][p->slot[b]]=1;
    for(int q=0;q<2;q++)for(int a=0;a<14;a++)for(int j=0;j<nj;j++)for(int l=0;l<3;l++){
        p->low[q][a][j][l]=1e30;p->high[q][a][j][l]=-1e30;
    }
    for(int i=0;i<ns;i++) {
        int a=part[i],q=query[i];
        if(a<0||a>=14||q<0||q>1||soft[i]<0||!(offset[i]>=0))goto bad;
        p->part[i]=a;p->query[i]=q;p->soft[i]=soft[i];p->offset[i]=offset[i];
        if(!q)p->max_offset[a]=max2(p->max_offset[a],offset[i]);
        p->members[q][a][p->count[q][a]++]=i;
        double sum=0;
        for(int z=0;z<4;z++) {
            int b=wi[4*i+z];double weight=w[4*i+z];if(b<0||b>=nj||!(weight>=0))goto bad;
            p->wi[i][z]=b;p->w[i][z]=weight;sum+=weight;
            for(int k=0;k<3;k++){
                double v=0,n=0;for(int l=0;l<3;l++){v+=rb[b*9+l*3+k]*(vertices[i*3+l]-rj[b*3+l]);n+=rb[b*9+l*3+k]*normals[i*3+l];}
                p->local[i][z][k]=v;p->normal[i][z][k]=n;
                if(weight>0){p->used[q][a][b]=1;p->low[q][a][b][k]=min2(p->low[q][a][b][k],v);p->high[q][a][b][k]=max2(p->high[q][a][b][k],v);}
            }
        }
        if(fabs(sum-1)>1e-5)goto bad;
    }
    for(int k=0;k<np;k++){int a=pairs[k*3],b=pairs[k*3+1];if(a<0||a>=14||b<0||b>=14||!p->count[1][a]||!p->count[0][b])goto bad;p->clothed[k]=pairs[k*3+2]==0||pairs[k*3+2]==2;}
    /* Density flags and clearance angles depend only on rest pack geometry. */
    for(int a=0;a<14;a++)for(int v=0;v<p->count[1][a];v++)
        p->sparse[p->members[1][a][v]]=v<(a>=4&&a<=7?p->count[1][a]:(int)max2(8,p->count[1][a]/4));
    double lateral[3];for(int l=0;l<3;l++)lateral[l]=rj[3*sem[16]+l]-rj[3*sem[17]+l];
    double width=sqrt(dot(lateral,lateral,3));if(width<1e-6)goto bad;
    for(int l=0;l<3;l++)lateral[l]/=width;
    for(int side=0;side<2;side++){
        int shoulder=sem[16+side],elbow=sem[18+side];double axis[3],len=0;
        for(int l=0;l<3;l++){axis[l]=rj[3*elbow+l]-rj[3*shoulder+l];len+=axis[l]*axis[l];}
        len=sqrt(len);if(len<1e-6)goto bad;for(int l=0;l<3;l++)axis[l]/=len;
        double extent=0,radius=0;
        for(int v=0;v<p->count[0][0];v++){
            int i=p->members[0][0][v];double d[3];for(int l=0;l<3;l++)d[l]=vertices[3*i+l]-rj[3*shoulder+l];
            extent=max2(extent,(side?-1:1)*dot(d,lateral,3));
        }
        for(int v=0;v<p->count[0][2+side];v++){
            int i=p->members[0][2+side][v];double d[3];for(int l=0;l<3;l++)d[l]=vertices[3*i+l]-rj[3*shoulder+l];
            double along=dot(d,axis,3);if(along<.2*len||along>.8*len)continue;radius=max2(radius,sqrt(max2(0,dot(d,d,3)-along*along)));
        }
        double angle=asin(min2(.9,(extent+radius)/len))*180/3.141592653589793;
        p->arm_extra[side]=max2(0,angle-15);
    }
    return p;
bad: free(p);return NULL;
}
void guard_free_pack(void *p){free(p);}
int guard_cfg_count(void){return CFG;}
void *guard_create(void *target,void *source,const double *cfg) {
    Pack *p=target,*s=source;if(!p||!s||p->np!=s->np||memcmp(p->pairs,s->pairs,p->np*3*sizeof(int)))return NULL;
    for(int i=0;i<CFG;i++)if(!isfinite(cfg[i])||cfg[i]<0)return NULL;
    if(cfg[KCFG]<1||cfg[KCFG]>128||cfg[SWEEPS]<1||cfg[FPS]<=0||cfg[INFLUENCE]<=0||cfg[WPEN]<=0)return NULL;
    if(cfg[FADE]<=0||cfg[FADE]>1||cfg[ROWALPHA]<=0||cfg[ROWALPHA]>1||cfg[PULLTRAVEL]<=0)return NULL;
    if(cfg[SPRING]!=0||cfg[WRES]<=0||cfg[SPINECOST]<1)return NULL;
    if(cfg[SOURCEALLOW]>1||cfg[SETTLE]>64||cfg[SETTLE]!=floor(cfg[SETTLE]))return NULL;
    if(cfg[DENSITY]>2||cfg[DENSITY]!=floor(cfg[DENSITY])||cfg[DENSENEAR]>.04)return NULL;
    if((cfg[MAXACC]>0)!=(cfg[MAXSPEED]>0)||(cfg[MAXACC]>0&&cfg[ZERO]>0))return NULL;
    if(cfg[KAPPA]>1)return NULL;
    Guard *g=calloc(1,sizeof(Guard));if(!g)return NULL;g->p=p;g->src=s;memcpy(g->cfg,cfg,CFG*sizeof(double));g->available=-1;
    for(int i=0;i<PMAX;i++)g->last_a[i]=g->last_b[i]=-1;return g;
}
void guard_free(void *g){free(g);}
static void fk(const Pack *p,const double *pose,const double *d,Geo *o) {
    memset(o->ready,0,p->ns);memset(o->normal_ready,0,p->ns);o->skinned=o->pairs_near=o->queries=o->dense_pairs=0;
    for(int j=0;j<p->nj;j++) {
        double r[9],local[9],jr[9];int k=p->slot[j],pa=p->par[j];
        if(d&&k>=0){exp_jr(d+3*k,r,jr);mm(pose+9*j,r,local);}else memcpy(local,pose+9*j,9*sizeof(double));
        mm(p->lb[j],local,r);
        if(pa<0){memcpy(o->r[j],r,9*sizeof(double));memcpy(o->j[j],p->lo[j],3*sizeof(double));}
        else{mm(o->r[pa],r,o->r[j]);mv(o->r[pa],p->lo[j],o->j[j]);for(int a=0;a<3;a++)o->j[j][a]+=o->j[pa][a];}
        if(k>=0){if(d)mm(o->r[j],jr,o->axes[k]);else memcpy(o->axes[k],o->r[j],9*sizeof(double));}
    }
}
static void skin_vertex(const Pack *p,Geo *o,int i) {
    if(o->ready[i])return;o->ready[i]=1;o->skinned++;
    memset(o->c[i],0,3*sizeof(double));
    for(int z=0;z<4;z++)if(p->w[i][z]>0){
        int b=p->wi[i][z];mv(o->r[b],p->local[i][z],o->pb[i][z]);
        for(int j=0;j<3;j++){o->pb[i][z][j]+=o->j[b][j];o->c[i][j]+=p->w[i][z]*o->pb[i][z][j];}
    }
}
static void skin_normal(const Pack *p,Geo *o,int i){
    if(o->normal_ready[i])return;o->normal_ready[i]=1;memset(o->normal[i],0,3*sizeof(double));
    for(int z=0;z<4;z++)if(p->w[i][z]>0){double n[3];mv(o->r[p->wi[i][z]],p->normal[i][z],n);for(int j=0;j<3;j++)o->normal[i][j]+=p->w[i][z]*n[j];}
    double len=sqrt(dot(o->normal[i],o->normal[i],3));for(int j=0;j<3;j++)o->normal[i][j]/=max2(len,1e-12);
}
static unsigned hash(int x,int y,int z) {return ((unsigned)x*73856093u^(unsigned)y*19349663u^(unsigned)z*83492791u)&(HASH-1);}
static double box_gap(const double *al,const double *ah,const double *bl,const double *bh){
    double d=0;for(int j=0;j<3;j++){double v=max2(0,max2(al[j]-bh[j],bl[j]-ah[j]));d+=v*v;}return d;
}
/* lift=1-kappa on the target, 0 on the source. Extra search reach of pair k. */
static inline double reach(const Pack *p,int k,double lift){return p->clothed[k]?lift*p->max_offset[p->pairs[k][1]]:0;}
/* Convex hull of transformed per-influence boxes bounds LBS without skinning.
 * Only parts in a potentially close directed pair are skinned and indexed. */
static void skin(const Pack *p,Geo *o,double lift) {
    unsigned char needed[2][14]={{0}};
    for(int q=0;q<2;q++)for(int a=0;a<14;a++){
        double *lo=o->low[q][a],*hi=o->high[q][a];for(int l=0;l<3;l++){lo[l]=1e30;hi[l]=-1e30;}
        for(int b=0;b<p->nj;b++)if(p->used[q][a][b])for(int l=0;l<3;l++){
            double mn=o->j[b][l],mx=mn;
            for(int j=0;j<3;j++){double x=o->r[b][3*l+j]*p->low[q][a][b][j],y=o->r[b][3*l+j]*p->high[q][a][b][j];mn+=min2(x,y);mx+=max2(x,y);}
            lo[l]=min2(lo[l],mn);hi[l]=max2(hi[l],mx);
        }
    }
    for(int k=0;k<p->np;k++){
        int a=p->pairs[k][0],b=p->pairs[k][1];double r=.04+reach(p,k,lift);o->near[k]=box_gap(o->low[1][a],o->high[1][a],o->low[0][b],o->high[0][b])<=r*r;
        if(o->near[k])needed[1][a]=needed[0][b]=1;
    }
    for(int phase=0;phase<2;phase++){
        int q=1-phase;
        /* Exact query bounds prune additional body parts before their LBS. */
        if(!q){
            memset(needed[0],0,sizeof(needed[0]));
            for(int k=0;k<p->np;k++)if(o->near[k]){
                int a=p->pairs[k][0],b=p->pairs[k][1];double r=.04+reach(p,k,lift);
                o->near[k]=box_gap(o->low[1][a],o->high[1][a],o->low[0][b],o->high[0][b])<=r*r;
                if(o->near[k])needed[0][b]=1;
            }
        }
        for(int a=0;a<14;a++)if(needed[q][a]){
        double *lo=o->low[q][a],*hi=o->high[q][a];for(int l=0;l<3;l++){lo[l]=1e30;hi[l]=-1e30;}
        if(!q)memset(o->head[a],-1,sizeof(o->head[a]));
        for(int v=0;v<p->count[q][a];v++){
            int i=p->members[q][a][v];skin_vertex(p,o,i);
            for(int l=0;l<3;l++){lo[l]=min2(lo[l],o->c[i][l]);hi[l]=max2(hi[l],o->c[i][l]);}
            if(!q){for(int l=0;l<3;l++)o->cell[i][l]=(int)floor(o->c[i][l]/CELL);unsigned h=hash(o->cell[i][0],o->cell[i][1],o->cell[i][2]);o->next[i]=o->head[a][h];o->head[a][h]=i;}
        }
    }
    }
    for(int k=0;k<p->np;k++)if(o->near[k]){
        int a=p->pairs[k][0],b=p->pairs[k][1];double r=.04+reach(p,k,lift);o->near[k]=box_gap(o->low[1][a],o->high[1][a],o->low[0][b],o->high[0][b])<=r*r;
        o->pairs_near+=o->near[k];
    }
}
/* Exact nearest vertex within radius. Expanding cell shells terminate once
 * the unvisited cells' lower bound exceeds the current best distance. */
static int nearest(const Geo *o,int part,const double *v,double radius,int seed,double *distance){
    int c[3];for(int l=0;l<3;l++)c[l]=(int)floor(v[l]/CELL);
    double best=radius*radius;int found=-1,limit=(int)ceil(radius/CELL);
    if(seed>=0){double d[3];for(int l=0;l<3;l++)d[l]=v[l]-o->c[seed][l];double sq=dot(d,d,3);if(sq<=best){best=sq;found=seed;}}
    for(int r=0;r<=limit;r++){
        for(int x=c[0]-r;x<=c[0]+r;x++)for(int y=c[1]-r;y<=c[1]+r;y++)for(int z=c[2]-r;z<=c[2]+r;z++){
            if(r&&abs(x-c[0])<r&&abs(y-c[1])<r&&abs(z-c[2])<r)continue;
            double lo[3]={x*CELL,y*CELL,z*CELL},hi[3]={(x+1)*CELL,(y+1)*CELL,(z+1)*CELL};
            if(box_gap(v,v,lo,hi)>best)continue;
            for(int i=o->head[part][hash(x,y,z)];i>=0;i=o->next[i]){
                if(o->cell[i][0]!=x||o->cell[i][1]!=y||o->cell[i][2]!=z)continue;
                double d[3];for(int l=0;l<3;l++)d[l]=v[l]-o->c[i][l];double sq=dot(d,d,3);
                if(sq<best||(sq==best&&(found<0||i<found))){best=sq;found=i;}
            }
        }
        double edge=1e30;for(int l=0;l<3;l++)edge=min2(edge,min2(v[l]-(c[l]-r)*CELL,(c[l]+r+1)*CELL-v[l]));
        if(edge*edge>=best)break;
    }
    *distance=sqrt(best);return found;
}
static inline double part_rho(int a,int b,const double *r){return (a==6||a==7)?r[1]:((a==4||a==5)&&b==0)?r[2]:r[0];}
static void distances(const Pack *p,Geo *o,const double *rho,int density,double dense_near,double lift,Dist *d){
    for(int k=0;k<p->np;k++){
        int a=p->pairs[k][0],b=p->pairs[k][1];d->gap[k]=d->clear[k]=.04;d->ga[k]=d->gb[k]=-1;
        for(int u=0;u<CAND;u++){d->a[k][u]=d->b[k][u]=-1;d->value[k][u]=.04;}
        if(!o->near[k])continue;
        double extra=reach(p,k,lift),full=dense_near+extra,cut=.04+extra;
        int dense=density==2||(density==1&&box_gap(o->low[1][a],o->high[1][a],o->low[0][b],o->high[0][b])<=full*full);
        o->dense_pairs+=dense;
        for(int v=0;v<p->count[1][a];v++){
            int i=p->members[1][a][v];if(!dense&&!p->sparse[i])continue;
            const double *q=o->c[i];
            /* Part boundary gate: negative planes outside the part AABB by
             * more than S0 near=12 mm are not anatomical penetration. */
            double outside=box_gap(q,q,o->low[0][b],o->high[0][b]);
            if(outside>cut*cut)continue;
            double distance,radius=outside<=.012*.012?max2(cut,p->depth[b]/.7):cut;
            int seed=o->hint[b][i];if(seed<0||seed>=p->ns||p->query[seed]||p->part[seed]!=b)seed=-1;
            o->queries++;int j=nearest(o,b,q,radius,seed,&distance);o->hint[b][i]=j;if(j<0)continue;
            double disp[3];for(int l=0;l<3;l++)disp[l]=q[l]-o->c[j][l];
            skin_normal(p,o,j);double phi=dot(disp,o->normal[j],3);
            int inside=phi<0&&(distance<.012||phi<-.7*distance)&&phi>-p->depth[b]&&outside<=.012*.012;
            /* The body test is unchanged; only the surface compared against moves out. */
            double gap=inside?phi:distance,lifted=p->clothed[k]?gap-lift*p->offset[j]:gap;
            double eff=lifted<0?min2(0,lifted+part_rho(a,b,rho)*(p->soft[i]+p->soft[j])):lifted;
            if(gap<d->gap[k])d->gap[k]=gap;
            if(eff<d->clear[k]){d->clear[k]=eff;d->ga[k]=i;d->gb[k]=j;}
            for(int u=0;u<CAND;u++)if(eff<d->value[k][u]){
                for(int h=CAND-1;h>u;h--){d->value[k][h]=d->value[k][h-1];d->a[k][h]=d->a[k][h-1];d->b[k][h]=d->b[k][h-1];memcpy(d->normal[k][h],d->normal[k][h-1],3*sizeof(double));}
                d->value[k][u]=eff;d->a[k][u]=i;d->b[k][u]=j;
                for(int l=0;l<3;l++)d->normal[k][u][l]=inside?o->normal[j][l]:disp[l]/max2(distance,1e-12);
                break;
            }
        }
    }
}
/* Jacobian of n dot (vertex a - vertex b), including blended influences
 * and the SO(3) right Jacobian for additive exponential coordinates. */
static void mesh_row(const Pack *p,const Geo *o,int a,int b,const double *n,double *row) {
    for(int k=0;k<21;k++) {
        double moment[3]={0,0,0};int joint=p->sem[k+1];
        for(int z=0;z<4;z++) {
            double wa=p->ancestor[p->wi[a][z]][k]*p->w[a][z],wb=p->ancestor[p->wi[b][z]][k]*p->w[b][z];
            for(int l=0;l<3;l++){if(wa)moment[l]+=wa*(o->pb[a][z][l]-o->j[joint][l]);if(wb)moment[l]-=wb*(o->pb[b][z][l]-o->j[joint][l]);}
        }
        double mn[3]={moment[1]*n[2]-moment[2]*n[1],moment[2]*n[0]-moment[0]*n[2],moment[0]*n[1]-moment[1]*n[0]};
        for(int l=0;l<3;l++)row[3*k+l]=o->axes[k][l]*mn[0]+o->axes[k][3+l]*mn[1]+o->axes[k][6+l]*mn[2];
    }
}
static void add_cost(Guard *g,const double *j,double error,double weight) {
    for(int a=0;a<N;a++){if(j[a]==0)continue;g->g[a]+=weight*j[a]*error;for(int b=0;b<=a;b++)g->H[a*N+b]+=weight*j[a]*j[b];}
}
static void objective(Guard *g) {
    Pack *p=g->p;Geo *o=&g->current,*ref=&g->ref;
    memset(g->H,0,sizeof(g->H));
    /* Reduce smoothing for deep penetration; keep shallow contact stable. */
    const double sm=g->cfg[URGSMOOTH]>1?1-g->urg*(1-1/g->cfg[URGSMOOTH]):1;
    for(int i=0;i<N;i++) {
        int j=i/3+1;
        double w=joint_w[j-1]*((j==3||j==6||j==9||j==12||j==15)?g->cfg[SPINECOST]:1);
        /* x is the increment at prev. Residual, velocity and acceleration
         * all retain the anatomical prior; a steady residual pays WRES. */
        g->H[i*N+i]=w*(g->cfg[WRES]+sm*g->cfg[WVEL]+sm*g->cfg[WACC]);
        g->g[i]=w*(g->cfg[WRES]*g->prev[i]+sm*g->cfg[WACC]*(g->prev2[i]-g->prev[i]));
    }
    for(int e=0;e<(g->cfg[POSVEL]+g->cfg[POSACC]>0?10:6);e++) {
        int b=p->sem[ee[e]];double jp[3][N]={{0}};
        for(int k=0;k<21;k++)if(p->ancestor[b][k]) {
            int j=p->sem[k+1];double v[3];for(int l=0;l<3;l++)v[l]=o->j[b][l]-o->j[j][l];
            for(int l=0;l<3;l++){const double *ax=o->axes[k];jp[0][k*3+l]=ax[3+l]*v[2]-ax[6+l]*v[1];jp[1][k*3+l]=ax[6+l]*v[0]-ax[l]*v[2];jp[2][k*3+l]=ax[l]*v[1]-ax[3+l]*v[0];}
        }
        for(int l=0;l<3;l++) {
            double error=o->j[b][l]-ref->j[b][l];
            if(e<6)add_cost(g,jp[l],error,(e<4?g->cfg[WFOOT]:g->cfg[WHAND]/(1+g->cfg[HANDRELAX]*max2(0,p->arm_extra[e-4]-g->cfg[CLEARANCEBAND])/15)));
            /* Preserve Cartesian residual velocity/acceleration, including
             * FK transport under a changing input pose, not just delta angles. */
            if(g->cfg[POSVEL]>0)add_cost(g,jp[l],error-g->old_ee[e][l],sm*g->cfg[POSVEL]);
            if(g->cfg[POSACC]>0)add_cost(g,jp[l],error-2*g->old_ee[e][l]+g->old_ee2[e][l],sm*g->cfg[POSACC]);
        }
        if(e>=4&&e<6)for(int col=0;col<3;col++) {
            memset(jp,0,sizeof(jp));
            for(int k=0;k<21;k++)if(p->ancestor[b][k])for(int l=0;l<3;l++) {
                const double *ax=o->axes[k],*r=o->r[b];jp[0][k*3+l]=ax[3+l]*r[6+col]-ax[6+l]*r[3+col];jp[1][k*3+l]=ax[6+l]*r[col]-ax[l]*r[6+col];jp[2][k*3+l]=ax[l]*r[3+col]-ax[3+l]*r[col];
            }
            for(int l=0;l<3;l++)add_cost(g,jp[l],o->r[b][3*l+col]-ref->r[b][3*l+col],g->cfg[WROT]);
        }
    }
}
static double smoothstep(double x) {x=max2(0,min2(1,x));return x*x*(3-2*x);}
/* Fixed per-pair state: spatial fade, temporal fade and an absolute-coordinate
 * row filter. The latter avoids a jump when the closest material pair changes.
 * Updating both A and b in absolute delta coordinates preserves the same plane. */
static int rows(Guard *g,Geo *o,Geo *ref,Dist *dist,int time,int future,double *switches) {
    Pack *p=g->p;int slot=time%WIN,selected[PMAX][CAND]={{0}},count=0;
    /* Reserve one worst row per pair before filling extra rows globally.
     * K is a real per-pose budget, not the unused sphere3 layout field. */
    for(int pass=0;pass<2;pass++)while(count<(int)g->cfg[KCFG]){
        double best=g->cfg[INFLUENCE];int bk=-1,bu=-1;
        for(int k=0;k<p->np;k++)for(int u=pass?1:0;u<(pass?CAND:1);u++){
            double clear=dist->value[k][u]+g->allow[slot][k]-g->cfg[EPS];
            if(!selected[k][u]&&dist->a[k][u]>=0&&clear<best){best=clear;bk=k;bu=u;}
        }
        if(bk<0)break;selected[bk][bu]=1;count++;
    }
    /* Retiring rows retain their filtered plane while fading, within K. */
    for(int k=0;k<p->np&&count<(int)g->cfg[KCFG];k++)for(int u=0;u<CAND&&count<(int)g->cfg[KCFG];u++)
        if(!selected[k][u]&&g->activation[future][k][u]>=1e-5){selected[k][u]=1;count++;}
    for(int k=0;k<p->np;k++)for(int u=0;u<CAND;u++){
        double clear=dist->value[k][u]+g->allow[slot][k]-g->cfg[EPS];
        double desired=selected[k][u]&&dist->a[k][u]>=0?smoothstep((g->cfg[INFLUENCE]-clear)/g->cfg[INFLUENCE]):0;
        double *activation=&g->activation[future][k][u];int fresh=*activation==0;
        *activation+=g->cfg[FADE]*(desired-*activation);
        if(*activation<1e-5&&desired==0){*activation=0;continue;}
        /* Fading state survives deselection, but only K rows enter the QP. */
        if(!selected[k][u])continue;
        int a=dist->a[k][u],b=dist->b[k][u];double row[N];const double *normal=dist->normal[k][u];
        if(a>=0){
        mesh_row(p,o,a,b,normal,row);double rhs=-clear;
        if(!future&&u==0){
            if(g->cfg[XI]>0&&g->frame){
                skin_vertex(p,ref,a);skin_vertex(p,ref,b);skin_vertex(p,&g->old_ref,a);skin_vertex(p,&g->old_ref,b);
                double approach=0;for(int j=0;j<3;j++)approach+=normal[j]*(ref->c[a][j]-ref->c[b][j]-g->old_ref.c[a][j]+g->old_ref.c[b][j]);
                approach+=g->allow[slot][k]-g->oldallow[k];
                rhs=max2(rhs,-g->cfg[XI]/g->cfg[FPS]*clear/g->cfg[INFLUENCE]-approach);
            }
            *switches+=(g->last_a[k]>=0&&(g->last_a[k]!=a||g->last_b[k]!=b));g->last_a[k]=a;g->last_b[k]=b;
        }
        rhs+=dot(row,g->prev,N);double alpha=fresh?1:g->cfg[ROWALPHA];
        for(int j=0;j<N;j++)g->smooth_a[future][k][u][j]+=alpha*(row[j]-g->smooth_a[future][k][u][j]);
        g->smooth_b[future][k][u]+=alpha*(rhs-g->smooth_b[future][k][u]);
        }
        int r=g->rows++;if(r>=MMAX)return -1;
        memcpy(g->A+r*N,g->smooth_a[future][k][u],N*sizeof(double));
        g->b[r]=g->smooth_b[future][k][u]-dot(g->A+r*N,g->prev,N);g->weight[r]=g->cfg[WPEN]*(*activation);
        if(g->cfg[SLACKCAP]>0){double cap=g->cfg[SLACKCAP]*(g->cfg[URGSLACK]>1?1+g->urg*(g->cfg[URGSLACK]-1):1);g->weight[r]*=min2(1,cap/max2(cap,-clear));}
    }
    for(int k=0;k<p->np;k++){
        int a=dist->ga[k],b=dist->gb[k];double gate=0;
        /* A kept contact rests on the lifted surface, not through it. */
        double rest=.01+(a>=0&&p->clothed[k]?(1-g->cfg[KAPPA])*p->offset[b]:0);
        if(a>=0&&g->cfg[WCONTACT]>0&&g->pull[slot][k]){
            skin_vertex(p,ref,a);skin_vertex(p,ref,b);
            double v[3];for(int j=0;j<3;j++)v[j]=ref->c[a][j]-ref->c[b][j];
            gate=1-smoothstep(max2(0,sqrt(dot(v,v,3))-rest)/g->cfg[PULLTRAVEL]);
        }
        double *pw=&g->pull_weight[future][k];*pw+=g->cfg[FADE]*(gate-*pw);
        if(gate==0||*pw<1e-5||a<0)continue;
        int r=g->rows++;if(r>=MMAX)return -1;
        double normal[3];for(int j=0;j<3;j++)normal[j]=o->c[a][j]-o->c[b][j];double len=sqrt(dot(normal,normal,3));
        if(len<1e-12){g->rows--;continue;}for(int j=0;j<3;j++)normal[j]/=-len;
        mesh_row(p,o,a,b,normal,g->A+r*N);g->b[r]=len-rest;g->weight[r]=g->cfg[WCONTACT]*(*pw)*gate;
    }
    return 0;
}
/* Box-constrained squared-hinge QP. Fixed cyclic coordinate sweeps,
 * each coordinate uses at most eight safeguarded piecewise-Newton updates.
 * Every iterate is feasible; no soft bound rows or post-solve clipping.
 * H is lower triangular on entry. Work O(sweeps*N*(N+8*rows)), no allocation. */
int guard_box(int n,int m,int sweeps,const double *h,const double *g,const double *a,
              const double *b,const double *w,const double *lower,const double *upper,double *x) {
    if(n<1||n>N||m<0||m>MMAX||sweeps<1)return -1;
    double hs[N*N],hx[N],res[MMAX],at[N*MMAX];
    for(int i=0;i<n;i++) {
        if(lower[i]>upper[i]||!(h[i*n+i]>0))return -2;
        x[i]=max2(lower[i],min2(upper[i],0));
        for(int j=0;j<n;j++)hs[i*n+j]=i>=j?h[i*n+j]:h[j*n+i];
        for(int r=0;r<m;r++)at[i*m+r]=a[r*n+i];
    }
    for(int i=0;i<n;i++)hx[i]=g[i]+dot(hs+i*n,x,n);
    for(int r=0;r<m;r++){if(!(w[r]>0))return -3;res[r]=b[r]-dot(a+r*n,x,n);}
    for(int sweep=0;sweep<sweeps;sweep++)for(int i=0;i<n;i++) {
        double lo=lower[i]-x[i],hi=upper[i]-x[i],v=0,base=hx[i],diag=hs[i*n+i];
        const double *ai=at+i*m;
        for(int inner=0;inner<8;inner++) {
            double grad=base+diag*v,curvature=diag;
            for(int r=0;r<m;r++) {
                double e=res[r]-ai[r]*v;
                if(e>0){grad-=w[r]*ai[r]*e;curvature+=w[r]*ai[r]*ai[r];}
            }
            if(fabs(grad)<1e-10||(v==lo&&grad>=0)||(v==hi&&grad<=0))break;
            if(grad>0)hi=v;else lo=v;
            v=max2(lo,min2(hi,v-grad/curvature));
        }
        /* Roundoff aside each update minimizes a convex 1D objective.
         * The final bounded inner step is accepted only if it decreases it. */
        double change=base*v+.5*diag*v*v;
        for(int r=0;r<m;r++){double e=max2(0,res[r]),en=max2(0,res[r]-ai[r]*v);change+=.5*w[r]*(en*en-e*e);}
        if(change>1e-12)v=0;
        x[i]+=v;
        for(int j=0;j<n;j++)hx[j]+=hs[i*n+j]*v;
        for(int r=0;r<m;r++)res[r]-=ai[r]*v;
    }
    return 0;
}
/* Inverse of the exact discrete stopping distance:
 * d = u + max(u-a,0) + max(u-2a,0) + ... . A continuous stopping
 * approximation underestimates this by up to a/8 near a joint bound. */
static double stop_speed(double distance,double acceleration) {
    double d=max2(0,distance),n=floor((sqrt(1+8*d/acceleration)-1)*.5);
    return d/(n+1)+.5*acceleration*n;
}
static double joint_limit(const Guard *g,int j) {
    double deg;
    if(j==3||j==6||j==9)deg=g->cfg[SPINELIMIT]/3;
    else if(j==12||j==15)deg=g->cfg[NECKLIMIT];
    else if(j==1||j==2)deg=g->cfg[HIPLIMIT];
    else if(j==20||j==21)deg=g->cfg[WRISTLIMIT];
    else if(j>=13)deg=min2(75,g->cfg[ARMLIMIT]+g->cfg[LIMITGAIN]*max2(0,g->p->arm_extra[(j==14||j==17||j==19)]-g->cfg[CLEARANCEBAND]));
    else deg=g->cfg[LEGLIMIT];
    /* Inscribed component box: ||delta_j|| <= degree limit. In particular
     * sum of the three spine norms <= SPINELIMIT, regardless of input pose. */
    return deg*(3.141592653589793/180)/sqrt(3.0);
}
/* timing[16:22] = queries, dense pairs, maximum slack (m), rows beyond
 * dynamic capacity by 5 mm, max component acceleration (rad/frame^2),
 * max component correction (rad).
 * timing[0:7] = total/FK/skin+grid/query/assembly/box/update milliseconds;
 * [7:11] = rows, source evaluations, switches, pull count; [11:13] =
 * source total / target geometry ms; [13:16] = near pairs, skinned, idle exit.
 * pose windows are [horizon+1,nj,3,3]. Outputs are 21 local rotation residuals
 * and the corrected [nj,3,3] local pose (zero deltas preserve input bits).
 * A failing return must be handled by the host, never treated as valid output. */
static int step_once(void *ctx,const double *target,const double *source,int horizon,double *out,double *pose_out,double *timing) {
    Guard *g=ctx;if(!g||horizon<0||horizon>=WIN)return -1;
    Pack *p=g->p,*s=g->src;double start=now(),t,lift=1-g->cfg[KAPPA];memset(timing,0,22*sizeof(double));
    const double rr[3]={g->cfg[RHO],g->cfg[RHOHAND],g->cfg[RHOFOREARM]};
    for(int i=0;i<(horizon+1)*p->nj*9;i++)if(!isfinite(target[i]))return -4;
    for(int i=0;i<(horizon+1)*s->nj*9;i++)if(!isfinite(source[i]))return -4;
    for(int i=g->available+1;i<=g->frame+horizon;i++) {
        int offset=i-g->frame,slot=i%WIN;if(offset<0)return -2;
        double source_start=now();t=now();fk(s,source+offset*s->nj*9,NULL,&g->source);timing[1]+=now()-t;
        t=now();skin(s,&g->source,0);timing[2]+=now()-t;
        t=now();distances(s,&g->source,rr,(int)g->cfg[DENSITY],g->cfg[DENSENEAR],0,&g->dist);timing[3]+=now()-t;
        for(int k=0;k<p->np;k++) {
            double gap=g->dist.clear[k],depth=max2(0,-gap);
            g->allow[slot][k]=(p->pairs[k][2]==6?.04:.003)+g->cfg[SOURCEALLOW]*depth;
            g->history[k]=((g->history[k]<<1)|(gap<.015))&31;g->contact[slot][k]=g->history[k]!=0;
            int hand=s->pairs[k][0]==6||s->pairs[k][0]==7;
            g->runs[k]=hand&&gap<.008?g->runs[k]+1:0;g->pull[slot][k]=g->runs[k]>=12;
        }
        timing[16]+=g->source.queries;timing[17]+=g->source.dense_pairs;timing[8]++;g->available=i;timing[11]+=now()-source_start;
    }
    double target_start=now();t=now();fk(p,target,g->prev,&g->current);fk(p,target,NULL,&g->ref);if(horizon){fk(p,target+horizon*p->nj*9,g->prev,&g->future);fk(p,target+horizon*p->nj*9,NULL,&g->future_ref);}timing[1]+=now()-t;
    t=now();skin(p,&g->current,lift);if(horizon)skin(p,&g->future,lift);timing[2]+=now()-t;
    timing[12]=now()-target_start;
    timing[13]=g->current.pairs_near;timing[14]=g->current.skinned;
    if(!g->current.pairs_near&&(!horizon||!g->future.pairs_near)&&dot(g->prev,g->prev,N)==0&&dot(g->velocity,g->velocity,N)==0){
        memset(out,0,N*sizeof(double));memcpy(pose_out,target,p->nj*9*sizeof(double));
        memset(g->prev2,0,sizeof(g->prev2));memset(g->old_ee,0,sizeof(g->old_ee));memset(g->old_ee2,0,sizeof(g->old_ee2));g->urg=0;
        for(int f=0;f<2;f++)for(int k=0;k<p->np;k++){g->pull_weight[f][k]*=1-g->cfg[FADE];for(int u=0;u<CAND;u++)g->activation[f][k][u]*=1-g->cfg[FADE];}
        if(g->cfg[XI]>0){memcpy(g->old_ref.r,g->ref.r,sizeof(g->ref.r));memcpy(g->old_ref.j,g->ref.j,sizeof(g->ref.j));memset(g->old_ref.ready,0,p->ns);}
        memcpy(g->oldallow,g->allow[g->frame%WIN],p->np*sizeof(double));g->frame++;
        timing[0]=now()-start;for(int i=0;i<7;i++)timing[i]*=1000;timing[11]*=1000;timing[12]*=1000;timing[15]=1;return 0;
    }
    g->rows=0;
    if(g->cfg[URGTAU]>0)g->urg*=1-1/g->cfg[URGTAU];
    for(int f=0;f<1+(horizon>0);f++) {
        int frame=g->frame+(f?horizon:0);Geo *geo=f?&g->future:&g->current;
        t=now();distances(p,geo,rr,(int)g->cfg[DENSITY],g->cfg[DENSENEAR],lift,&g->dist);timing[16]+=geo->queries;timing[17]+=geo->dense_pairs;double query_time=now()-t;timing[3]+=query_time;timing[12]+=query_time;
        if(g->cfg[URGD1]>g->cfg[URGD0]){
            /* Use hand/forearm penetration into the torso; soft tissue cannot absorb it. Take the larger current/future urgency, rise immediately and decay with URGTAU. */
            double worst=0;
            for(int k=0;k<p->np;k++){
                int a=p->pairs[k][0];
                if(a>=4&&a<=7&&p->pairs[k][1]==0&&g->dist.a[k][0]>=0)worst=max2(worst,-(g->dist.value[k][0]+g->allow[frame%WIN][k]-g->cfg[EPS]));
            }
            g->urg=max2(g->urg,min2(1,max2(0,(worst-g->cfg[URGD0])/(g->cfg[URGD1]-g->cfg[URGD0]))));
        }
        t=now();int rc=rows(g,geo,f?&g->future_ref:&g->ref,&g->dist,frame,f,&timing[9]);timing[4]+=now()-t;if(rc)return rc-10;
    }
    t=now();objective(g);timing[4]+=now()-t;
    t=now();double lower[N],upper[N];
    const double uv=g->cfg[URGSPEED]>1?1+g->urg*(g->cfg[URGSPEED]-1):1,ua=g->cfg[URGACC]>1?1+g->urg*(g->cfg[URGACC]-1):1;
    for(int i=0;i<N;i++){
        double limit=joint_limit(g,i/3+1);lower[i]=-limit-g->prev[i];upper[i]=limit-g->prev[i];
        if(g->cfg[MAXACC]>0&&g->cfg[MAXSPEED]>0){
            double a=ua*g->cfg[MAXACC]*(3.141592653589793/180)/(sqrt(3.)*g->cfg[FPS]*g->cfg[FPS]);
            double vmax=uv*g->cfg[MAXSPEED]*(3.141592653589793/180)/(sqrt(3.)*g->cfg[FPS]);
            /* Braking distance includes the next discrete frame. These bounds
             * outrank collision/contact costs, including impossible rows. */
            double vlo=max2(lower[i],-min2(vmax,stop_speed(-lower[i],a)));
            double vhi=min2(upper[i],min2(vmax,stop_speed(upper[i],a)));
            double hard_lo=lower[i],hard_hi=upper[i];
            lower[i]=max2(vlo,g->velocity[i]-a);upper[i]=min2(vhi,g->velocity[i]+a);
            if(lower[i]>upper[i]){ /* Roundoff at a braking boundary. */
                if(lower[i]-upper[i]<=1e-10)lower[i]=upper[i]=.5*(lower[i]+upper[i]);
                else {
                    /* As urgency decays the speed bound tightens. Brake at the current acceleration while preserving the hard joint bounds. */
                    double v=g->velocity[i]>0?g->velocity[i]-a:g->velocity[i]+a;
                    lower[i]=upper[i]=max2(hard_lo,min2(hard_hi,v));
                }
            }
        }
    }
    int rc=guard_box(N,g->rows,(int)g->cfg[SWEEPS],g->H,g->g,g->A,g->b,g->weight,lower,upper,g->x);timing[5]=now()-t;if(rc)return rc-20;
    t=now();
    /* Release unconstrained joints toward zero with the time constant, subject to speed and acceleration bounds. */
    int held[21]={0};
    if(g->cfg[RELEASE]>0){
        double step[N];
        for(int i=0;i<N;i++)step[i]=min2(upper[i],max2(lower[i],-g->prev[i]/g->cfg[RELEASE]));
        /* Hold joints only when this release step would violate a constraint. */
        for(int r=0;r<g->rows;r++){
            if(dot(g->A+r*N,step,N)>=g->b[r])continue;
            for(int k=0;k<21;k++)for(int a=0;a<3;a++)if(fabs(g->A[r*N+k*3+a])>1e-9)held[k]=1;
        }
        for(int k=0;k<21;k++)if(!held[k])for(int a=0;a<3;a++)g->x[k*3+a]=step[k*3+a];
    }
    for(int r=0;r<g->rows;r++){
        double slack=g->b[r]-dot(g->A+r*N,g->x,N),capacity=0;
        for(int i=0;i<N;i++)capacity+=g->A[r*N+i]*(g->A[r*N+i]>0?upper[i]:lower[i]);
        timing[18]=max2(timing[18],slack);timing[19]+=g->b[r]>capacity+.005;
    }
    for(int k=0;k<21;k++) {
        double v[3],norm=0;
        for(int a=0;a<3;a++){int i=k*3+a;v[a]=g->prev[i]+g->x[i];norm+=v[a]*v[a];}
        norm=sqrt(norm);if(!isfinite(norm))return -30;double scale=norm>g->cfg[ZERO]?1-g->cfg[ZERO]/norm:0;
        if(g->cfg[SNAP]>0&&!held[k]&&norm<g->cfg[SNAP]*3.141592653589793/180)scale=0;
        for(int a=0;a<3;a++){int i=k*3+a;out[i]=v[a]*scale;g->prev2[i]=g->prev[i];g->prev[i]=out[i];double v=out[i]-g->prev2[i];timing[20]=max2(timing[20],fabs(v-g->velocity[i]));g->velocity[i]=v;timing[21]=max2(timing[21],fabs(out[i]));}
    }
    for(int j=0;j<p->nj;j++) {
        int k=p->slot[j];
        if(k<0||(out[3*k]==0&&out[3*k+1]==0&&out[3*k+2]==0))memcpy(pose_out+9*j,target+9*j,9*sizeof(double));
        else {double r[9];exp_jr(out+3*k,r,NULL);mm(target+9*j,r,pose_out+9*j);}
    }
    if(g->cfg[POSVEL]+g->cfg[POSACC]>0) {
        fk(p,target,out,&g->current);
        for(int e=0;e<10;e++)for(int l=0;l<3;l++) {
            g->old_ee2[e][l]=g->old_ee[e][l];
            g->old_ee[e][l]=g->current.j[p->sem[ee[e]]][l]-g->ref.j[p->sem[ee[e]]][l];
        }
    }
    if(g->cfg[XI]>0){memcpy(g->old_ref.r,g->ref.r,sizeof(g->ref.r));memcpy(g->old_ref.j,g->ref.j,sizeof(g->ref.j));memset(g->old_ref.ready,0,p->ns);}memcpy(g->oldallow,g->allow[g->frame%WIN],p->np*sizeof(double));
    for(int k=0;k<p->np;k++)timing[10]+=g->pull[g->frame%WIN][k];g->frame++;
    timing[6]=now()-t;timing[0]=now()-start;for(int i=0;i<7;i++)timing[i]*=1000;timing[11]*=1000;timing[12]*=1000;timing[7]=g->rows;return 0;
}
/* The first frame has no history, so the speed limits would make a pose that
 * already penetrates climb out over several output frames. Settle on it first. */
int guard_step(void *ctx,const double *target,const double *source,int horizon,double *out,double *pose_out,double *timing) {
    Guard *g=ctx;if(!g)return -1;
    if(g->frame==0)for(int i=0;i<(int)g->cfg[SETTLE];i++){int rc=step_once(ctx,target,source,horizon,out,pose_out,timing);if(rc)return rc;}
    return step_once(ctx,target,source,horizon,out,pose_out,timing);
}
/* Diagnostics use the identical grid/gates as the streaming step. */
int guard_geometry(void *pack,const double *pose,const double *delta,double rho,double kappa,double *vertices,double *joints,double *world,double *gap,double *soft) {
    Pack *p=pack;Geo *o=calloc(1,sizeof(Geo));Dist d;if(!o)return -1;const double rr[3]={rho,rho,rho};fk(p,pose,delta,o);skin(p,o,1-kappa);distances(p,o,rr,2,.04,1-kappa,&d);
    for(int i=0;i<p->ns;i++)skin_vertex(p,o,i);
    memcpy(vertices,o->c,p->ns*3*sizeof(double));memcpy(joints,o->j,p->nj*3*sizeof(double));memcpy(world,o->r,p->nj*9*sizeof(double));
    memcpy(gap,d.gap,p->np*sizeof(double));memcpy(soft,d.clear,p->np*sizeof(double));free(o);return 0;
}
int guard_jacobian(void *pack,const double *pose,const double *delta,int a,int b,const double *normal,double *row) {
    Pack *p=pack;if(a<0||a>=p->ns||b<0||b>=p->ns)return -1;Geo *o=calloc(1,sizeof(Geo));if(!o)return -2;fk(p,pose,delta,o);skin_vertex(p,o,a);skin_vertex(p,o,b);mesh_row(p,o,a,b,normal,row);free(o);return 0;
}
void guard_set_delta(void *ctx,const double *delta){Guard *g=ctx;memcpy(g->prev,delta,N*sizeof(double));memcpy(g->prev2,delta,N*sizeof(double));}
int guard_problem(void *ctx,double *h,double *grad,double *a,double *b,double *w){Guard *g=ctx;for(int i=0;i<N;i++)for(int j=0;j<N;j++)h[i*N+j]=i>=j?g->H[i*N+j]:g->H[j*N+i];memcpy(grad,g->g,N*sizeof(double));memcpy(a,g->A,g->rows*N*sizeof(double));memcpy(b,g->b,g->rows*sizeof(double));memcpy(w,g->weight,g->rows*sizeof(double));return g->rows;}

void guard_clearance(void *pack,double *out){Pack *p=pack;out[0]=p->arm_extra[0];out[1]=p->arm_extra[1];}
