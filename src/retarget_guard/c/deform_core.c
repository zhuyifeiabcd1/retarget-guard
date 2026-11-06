/* C99, metres, row-major float arrays. Only the C library and libm.
 * Static topology/weights in deform_create. deform_step takes rendered skin
 * v/n [nv,3] and separately posed collider cv/cn [nc,3], all float arrays.
 * They may alias for one mesh. Colliders are real surface vertices including
 * fingers, forearms and upper arms; the bounding spheres only prune queries.
 * Scratch is reused but no previous pose/displacement affects the result.
 * AABB/index construction is O(V); narrow phase, fields and fold checks visit
 * only nearby cells and affected triangles. No allocation or threads in step.
 * Labels: torso/head 0/1, arms 2..5, hands 6/7, legs 8..13 (S0 convention).
 * create arrays: triangles [nf,3], body/collider part and soft [nv]/[nc],
 * unique undirected pairs [np,2], feathered weights [nv,npart], seam masks
 * [nv,np]/[nc,np], query bit masks and pooling groups [nc]. Normals are
 * outward and unit length. query bits: 2 broad, 4 hand/hand; group maps to
 * a collider vertex in the same part. The caller owns output [nv,3].
 * The 12 cm search supports local compression, not arbitrarily deep overlap.
 */
#ifdef DEFORM_BENCHMARK
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define PARTS 32
#define HASH 2048
#define CELL .06f
typedef struct {
    int count[PARTS];
    int nv,nf,nc,np,npart,*faces,*part,*pairs,*query,*offset,*adj;
    float *soft,*pw,*mask,*csoft,*cmask,rho,rim;
    int *cpart,*ccell;
    int *next,*gnext,*cell,*members,*start,*fmembers,*fstart;
    int head[PARTS][HASH],global[PARTS][HASH];
    float lo[PARTS][3],hi[PARTS][3],centre[PARTS][3],radius[PARTS];
    float *acc,*weight,*scale,*event,*fgeom;
    int *events,*goffset,*gmember;
    int *touched,*fids,nt,nft;
    unsigned char *seen,*fseen;
} Deform;
static float mn(float a,float b){return a<b?a:b;}
static float mx(float a,float b){return a>b?a:b;}
static float dot(const float *a,const float *b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void cross(const float *a,const float *b,float *o){o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
static unsigned hash(int x,int y,int z){return ((unsigned)x*73856093u^(unsigned)y*19349663u^(unsigned)z*83492791u)&(HASH-1);}
static float gap(const float *al,const float *ah,const float *bl,const float *bh){float r=0;for(int k=0;k<3;k++){float a=mx(0,mx(al[k]-bh[k],bl[k]-ah[k]));r+=a*a;}return r;}
static float smooth(float x){x=mx(0,mn(1,x));return x*x*(3-2*x);}
void deform_free(void *ctx){
    Deform *g=ctx;if(!g)return;
    free(g->faces);free(g->part);free(g->pairs);free(g->query);free(g->event);free(g->events);free(g->offset);free(g->adj);
    free(g->csoft);free(g->cmask);free(g->cpart);free(g->ccell);free(g->soft);free(g->pw);free(g->mask);free(g->next);free(g->gnext);
    free(g->cell);free(g->members);free(g->start);free(g->fmembers);free(g->fstart);free(g->acc);free(g->weight);
    free(g->goffset);free(g->gmember);free(g->fgeom);free(g->scale);free(g->touched);free(g->fids);free(g->seen);free(g->fseen);free(g);
}
void *deform_create(int nv,int nf,int nc,int np,int npart,const int *faces,const int *part,
    const int *pairs,const int *cpart,const int *query,const int *group,const int *thin_group,const float *soft,const float *pw,const float *mask,
    const float *csoft,const float *cmask,float rho,float rim){
    if(nv<1||nf<1||nc<1||np<0||npart<1||npart>PARTS||!isfinite(rho)||rho<0||!isfinite(rim)||rim<=0)return NULL;
    Deform *g=calloc(1,sizeof(*g));if(!g)return NULL;
    g->nv=nv;g->nf=nf;g->nc=nc;g->np=np;g->npart=npart;g->rho=rho;g->rim=rim;
#define NEW(name,count,type) g->name=calloc((size_t)(count)+1,sizeof(type));if(!g->name)goto bad
    NEW(goffset,2*(nc+1),int);NEW(gmember,2*nc,int);NEW(cpart,nc,int);NEW(ccell,3*nc,int);NEW(csoft,nc,float);NEW(cmask,nc*np,float);
    NEW(event,nc*13,float);NEW(events,nc,int);NEW(query,nc,int);NEW(faces,3*nf,int);NEW(part,nv,int);NEW(pairs,2*np,int);NEW(offset,nv+1,int);NEW(adj,3*nf,int);
    NEW(soft,nv,float);NEW(pw,nv*npart,float);NEW(mask,nv*np,float);
    NEW(next,nv+nc,int);NEW(gnext,nv*npart,int);NEW(fmembers,nv*npart,int);NEW(fstart,npart+1,int);NEW(cell,3*nv,int);NEW(members,nc,int);NEW(start,npart+1,int);
    NEW(fgeom,10*nf,float);NEW(acc,3*nv,float);NEW(weight,nv,float);NEW(scale,nv,float);NEW(touched,nv,int);NEW(fids,nf,int);
    NEW(seen,nv,unsigned char);NEW(fseen,nf,unsigned char);
#undef NEW
    memcpy(g->cpart,cpart,nc*sizeof(int));memcpy(g->csoft,csoft,nc*sizeof(float));
    for(int p=0;p<np;p++)for(int i=0;i<nc;i++)g->cmask[p*nc+i]=cmask[i*np+p];
    memcpy(g->query,query,nc*sizeof(int));memcpy(g->faces,faces,3*nf*sizeof(int));memcpy(g->part,part,nv*sizeof(int));memcpy(g->pairs,pairs,2*np*sizeof(int));
    memcpy(g->soft,soft,nv*sizeof(float));
    for(int p=0;p<npart;p++)for(int i=0;i<nv;i++)g->pw[p*nv+i]=pw[i*npart+p];
    for(int p=0;p<np;p++)for(int i=0;i<nv;i++)g->mask[p*nv+i]=mask[i*np+p];
    for(int p=0;p<npart;p++){
        g->fstart[p+1]=g->fstart[p];
        for(int i=0;i<nv;i++)if(pw[i*npart+p]>0)g->fmembers[g->fstart[p+1]++]=i;
    }
    for(int i=0;i<nv;i++){
        if(part[i]<0||part[i]>=npart||!isfinite(soft[i])||soft[i]<0)goto bad;

        for(int p=0;p<npart;p++)if(!isfinite(pw[i*npart+p])||pw[i*npart+p]<0)goto bad;
        for(int p=0;p<np;p++)if(!isfinite(mask[i*np+p])||mask[i*np+p]<0||mask[i*np+p]>1)goto bad;
    }
    for(int i=0;i<nc;i++){
        if(cpart[i]<0||cpart[i]>=npart||!isfinite(csoft[i])||csoft[i]<0||group[i]<0||group[i]>=nc)goto bad;
        if(cpart[group[i]]!=cpart[i]||query[i]<0||query[i]>7)goto bad;
        g->count[cpart[i]]++;if(query[i])g->start[cpart[i]+1]++;
        for(int p=0;p<np;p++)if(!isfinite(cmask[i*np+p])||cmask[i*np+p]<0||cmask[i*np+p]>1)goto bad;
    }
    for(int k=0;k<2*np;k++)if(pairs[k]<0||pairs[k]>=npart)goto bad;
    for(int k=0;k<3*nf;k++){if(faces[k]<0||faces[k]>=nv)goto bad;g->offset[faces[k]+1]++;}
    for(int i=1;i<=nv;i++)g->offset[i]+=g->offset[i-1];
    for(int p=1;p<=npart;p++)g->start[p]+=g->start[p-1];
    for(int k=0;k<3*nf;k++){int i=faces[k];g->adj[g->offset[i]+g->next[i]++]=k/3;}
    int count[PARTS]={0};for(int i=0;i<nc;i++)if(query[i])g->members[g->start[cpart[i]]+count[cpart[i]]++]=i;
    for(int mode=0;mode<2;mode++){
        const int *grouping=mode?thin_group:group;int *off=g->goffset+mode*(nc+1),*members=g->gmember+mode*nc;
        memset(g->next,0,nc*sizeof(int));
        for(int i=0;i<nc;i++){
            int id=grouping[i];if(id<0||id>=nc||cpart[id]!=cpart[i])goto bad;off[id+1]++;
        }
        for(int i=1;i<=nc;i++)off[i]+=off[i-1];
        for(int i=0;i<nc;i++){int id=grouping[i];members[off[id]+g->next[id]++]=i;}
    }
    return g;
bad:deform_free(g);return NULL;
}
/* Exact nearest collider vertex inside the requested radius. */
static int nearest(Deform *g,const float *v,int part,const float *q,float radius,float *distance){
    int c[3];for(int k=0;k<3;k++)c[k]=(int)floorf(q[k]/CELL);
    float best=radius*radius;int found=-1,limit=(int)ceilf(radius/CELL);
    for(int r=0;r<=limit;r++){
        for(int x=c[0]-r;x<=c[0]+r;x++)for(int y=c[1]-r;y<=c[1]+r;y++)for(int z=c[2]-r;z<=c[2]+r;z++){
            if(r&&abs(x-c[0])<r&&abs(y-c[1])<r&&abs(z-c[2])<r)continue;
            float lo[3]={x*CELL,y*CELL,z*CELL},hi[3]={(x+1)*CELL,(y+1)*CELL,(z+1)*CELL};
            if(gap(q,q,lo,hi)>best)continue;
            for(int j=g->head[part][hash(x,y,z)];j>=0;j=g->next[j]){
                if(g->ccell[3*j]!=x||g->ccell[3*j+1]!=y||g->ccell[3*j+2]!=z)continue;
                float d[3];for(int k=0;k<3;k++)d[k]=q[k]-v[3*j+k];float ds=dot(d,d);
                if(ds<best||(ds==best&&(found<0||j<found))){best=ds;found=j;}
            }
        }
        float edge=1e20f;for(int k=0;k<3;k++)edge=mn(edge,mn(q[k]-(c[k]-r)*CELL,(c[k]+r+1)*CELL-q[k]));
        if(edge*edge>=best)break;
    }
    *distance=sqrtf(best);return found;
}
/* Smooth the sampled skin plane on broad body surfaces. The support depends
 * on nearest distance, not nearest identity, so vertex switches are continuous.
 * Thin hands keep their exact nearest surface to avoid averaging across fingers. */
static void surface(Deform *g,const float *v,const float *n,int part,const float *q,int nearest_id,
    float distance,float *point,float *normal,float *soft){
    if(part==6||part==7){memcpy(point,v+3*nearest_id,3*sizeof(float));memcpy(normal,n+3*nearest_id,3*sizeof(float));*soft=g->csoft[nearest_id];return;}
    const float support=.03f*.03f;float best=distance*distance,radius=sqrtf(best+support),sum=0;
    int lo[3],hi[3];for(int k=0;k<3;k++){point[k]=normal[k]=0;lo[k]=(int)floorf((q[k]-radius)/CELL);hi[k]=(int)floorf((q[k]+radius)/CELL);}*soft=0;
    for(int x=lo[0];x<=hi[0];x++)for(int y=lo[1];y<=hi[1];y++)for(int z=lo[2];z<=hi[2];z++){
        float cl[3]={x*CELL,y*CELL,z*CELL},ch[3]={(x+1)*CELL,(y+1)*CELL,(z+1)*CELL};if(gap(q,q,cl,ch)>best+support)continue;
        for(int i=g->head[part][hash(x,y,z)];i>=0;i=g->next[i]){
            if(g->ccell[3*i]!=x||g->ccell[3*i+1]!=y||g->ccell[3*i+2]!=z)continue;
            float d[3];for(int k=0;k<3;k++)d[k]=q[k]-v[3*i+k];float u=mx(0,(dot(d,d)-best)/support);if(u>=1)continue;
            float w=(1-u)*(1-u)*(1-u)*(1+3*u);sum+=w;*soft+=w*g->csoft[i];
            for(int k=0;k<3;k++){point[k]+=w*v[3*i+k];normal[k]+=w*n[3*i+k];}
        }
    }
    float norm=mx(sqrtf(dot(normal,normal)),1e-12f);for(int k=0;k<3;k++){point[k]/=sum;normal[k]/=norm;}*soft/=sum;
}
static void field(Deform *g,const float *v,int part,int pair,const float *point,
    const float *dir,float depth,float width,double *visits){
    int lo[3],hi[3];for(int k=0;k<3;k++){lo[k]=(int)floorf((point[k]-width)/CELL);hi[k]=(int)floorf((point[k]+width)/CELL);}
    float inv=1/(width*width);int visit_count=0;
    for(int x=lo[0];x<=hi[0];x++)for(int y=lo[1];y<=hi[1];y++)for(int z=lo[2];z<=hi[2];z++){
        float cl[3]={x*CELL,y*CELL,z*CELL},ch[3]={(x+1)*CELL,(y+1)*CELL,(z+1)*CELL};
        if(gap(point,point,cl,ch)>=width*width)continue;
        for(int j=g->global[part][hash(x,y,z)];j>=0;j=g->gnext[part*g->nv+j]){
            if(g->cell[3*j]!=x||g->cell[3*j+1]!=y||g->cell[3*j+2]!=z)continue;
            float w=g->pw[part*g->nv+j]*g->mask[pair*g->nv+j];if(w<=0)continue;
            float d[3];for(int k=0;k<3;k++)d[k]=v[3*j+k]-point[k];float u=dot(d,d)*inv;if(u>=1)continue;
            visit_count++;
            float a=(1-u)*(1-u)*(1-u)*(1+3*u)*depth*w;if(a<1e-8f)continue;
            float blend=a*100;blend*=blend;blend*=blend;
            if(!g->seen[j]){g->seen[j]=1;g->touched[g->nt++]=j;}
            for(int k=0;k<3;k++)g->acc[3*j+k]+=blend*a*dir[k];g->weight[j]+=blend;
        }
    }
    *visits+=visit_count;
}
static float face_scale(const Deform *g,const float *d,int f){
    int a=g->faces[3*f],b=g->faces[3*f+1],c=g->faces[3*f+2];float z[3],w[3],t[3],s[3];
    const float *e=g->fgeom+10*f,*h=e+3,*n=e+6;float inv=e[9];
    for(int k=0;k<3;k++){z[k]=d[3*b+k]-d[3*a+k];w[k]=d[3*c+k]-d[3*a+k];}
    cross(e,w,t);cross(z,h,s);float B=mx(0,-(dot(t,n)+dot(s,n))*inv);cross(z,w,t);float C=mx(0,-dot(t,n)*inv);
    return mn(1,1.6f/mx(1e-12f,B+sqrtf(B*B+3.2f*C)));
}
/* stats: contacts, field vertex visits, moved vertices, affected triangles,
 * minimum local scale, final scale, nearest queries, near pairs.
 * out must be a separate [nv,3] buffer. A negative return is invalid output. */
int deform_step(void *ctx,const float *v,const float *n,const float *cv,const float *cn,float *out,double *stats){
    Deform *g=ctx;if(!g)return -1;memset(stats,0,8*sizeof(double));stats[4]=stats[5]=1;
    memset(out,0,g->nv*3*sizeof(float));memset(g->acc,0,g->nv*3*sizeof(float));memset(g->weight,0,g->nv*sizeof(float));
    memset(g->seen,0,g->nv);memset(g->fseen,0,g->nf);g->nt=g->nft=0;
    for(int p=0;p<g->npart;p++)for(int k=0;k<3;k++){g->lo[p][k]=1e20f;g->hi[p][k]=-1e20f;g->centre[p][k]=0;g->radius[p]=0;}
    for(int i=0;i<g->nv*3;i++)if(!isfinite(v[i])||!isfinite(n[i]))return -2;
    for(int i=0;i<g->nc;i++)for(int k=0;k<3;k++){
        float a=cv[3*i+k];if(!isfinite(a)||!isfinite(cn[3*i+k]))return -2;
        int p=g->cpart[i];g->lo[p][k]=mn(g->lo[p][k],a);g->hi[p][k]=mx(g->hi[p][k],a);g->centre[p][k]+=a;
    }
    for(int p=0;p<g->npart;p++)for(int k=0;k<3;k++)g->centre[p][k]/=mx(1,g->count[p]);
    for(int i=0;i<g->nc;i++){int p=g->cpart[i];float q[3];for(int k=0;k<3;k++)q[k]=cv[3*i+k]-g->centre[p][k];g->radius[p]=mx(g->radius[p],dot(q,q));}
    for(int p=0;p<g->npart;p++)g->radius[p]=sqrtf(g->radius[p]);
    unsigned char needed[PARTS]={0};
    for(int p=0;p<g->np;p++){int a=g->pairs[2*p],b=g->pairs[2*p+1];if(gap(g->lo[a],g->hi[a],g->lo[b],g->hi[b])<.12f*.12f){needed[a]=needed[b]=1;stats[7]++;}}
    if(!stats[7])return 0;
    memset(g->head,-1,g->npart*HASH*sizeof(int));memset(g->global,-1,g->npart*HASH*sizeof(int));
    for(int i=0;i<g->nv;i++){
        for(int k=0;k<3;k++)g->cell[3*i+k]=(int)floorf(v[3*i+k]/CELL);
    }
    for(int i=0;i<g->nc;i++)if(needed[g->cpart[i]]){
        for(int k=0;k<3;k++)g->ccell[3*i+k]=(int)floorf(cv[3*i+k]/CELL);
        unsigned h=hash(g->ccell[3*i],g->ccell[3*i+1],g->ccell[3*i+2]);
        g->next[i]=g->head[g->cpart[i]][h];g->head[g->cpart[i]][h]=i;
    }
    for(int p=0;p<g->npart;p++)if(needed[p])for(int it=g->fstart[p];it<g->fstart[p+1];it++){
        int i=g->fmembers[it];unsigned h=hash(g->cell[3*i],g->cell[3*i+1],g->cell[3*i+2]);
        g->gnext[p*g->nv+i]=g->global[p][h];g->global[p][h]=i;
    }
    for(int p=0;p<g->np;p++){
        int a=g->pairs[2*p],b=g->pairs[2*p+1];
        if(gap(g->lo[a],g->hi[a],g->lo[b],g->hi[b])>=.12f*.12f)continue;
        /* Prefer hands and arms as contact owners; otherwise use the smaller skin. */
        {
            int pa=(a==6||a==7)?0:(a>=2&&a<=5)?1:2;
            int pb=(b==6||b==7)?0:(b>=2&&b<=5)?1:2;
            int swap=pa>pb||(pa==pb&&g->count[a]>g->count[b]);
            int source=swap?b:a,target=swap?a:b;
            int ne=0,thin=(source==6||source==7)&&(target==6||target==7);
            const int *off=g->goffset+thin*(g->nc+1),*members=g->gmember+thin*g->nc;
            for(int it=g->start[source];it<g->start[source+1];it++){
                int seed=g->members[it];if(!(g->query[seed]&(thin?4:2)))continue;
                int size=off[seed+1]-off[seed];if(!size)continue;
                float q[3]={0};
                for(int z=off[seed];z<off[seed+1];z++)for(int k=0;k<3;k++)q[k]+=cv[3*members[z]+k]/size;
                float rad=0;
                for(int z=off[seed];z<off[seed+1];z++){int i=members[z];float diff[3];for(int k=0;k<3;k++)diff[k]=cv[3*i+k]-q[k];rad=mx(rad,dot(diff,diff));}
                rad=sqrtf(rad);
                if(gap(q,q,g->lo[target],g->hi[target])>(.12f+rad)*(.12f+rad))continue;
                float bound[3];for(int k=0;k<3;k++)bound[k]=q[k]-g->centre[target][k];
                if(dot(bound,bound)>(g->radius[target]+rad+.012f)*(g->radius[target]+rad+.012f))continue;
                float distance;stats[6]++;int j=nearest(g,cv,target,q,.12f+rad,&distance);if(j<0)continue;
                float point[3],normal[3],target_soft;surface(g,cv,cn,target,q,j,distance,point,normal,&target_soft);
                float *ev=g->event+13*seed;
                for(int z=off[seed];z<off[seed+1];z++){
                    int i=members[z];float diff[3];for(int k=0;k<3;k++)diff[k]=cv[3*i+k]-point[k];
                    float phi=dot(diff,normal),facing=mx(0,-dot(cn+3*i,normal)),dist2=dot(diff,diff);
                    if(facing<=0||dist2>.12f*.12f)continue;
                    float local=facing*facing*g->cmask[p*g->nc+i];
                    float weight=expf(-phi/.004f)*local;ev[0]+=weight;
                    for(int k=0;k<3;k++){
                        ev[1+k]+=weight*cv[3*i+k];ev[4+k]+=weight*(cv[3*i+k]-phi*normal[k]);ev[7+k]+=weight*normal[k];
                    }
                    ev[10]+=weight*g->csoft[i];ev[12]+=local;
                }
                if(ev[0]>0){
                    float diff[3];for(int k=0;k<3;k++)diff[k]=(ev[1+k]-ev[4+k])/ev[0];
                    float raw=mx(0,-dot(diff,normal)),sa=ev[10]/ev[0],total=sa+target_soft;
                    float dep=mn(raw*smooth(raw/.002f)*mn(1,ev[12]/.05f)*1.15f,g->rho*total);
                    if(dep>1e-7f&&total>0){ev[10]=ev[0]*dep;ev[11]=ev[0]*dep*sa/total;ev[12]=ev[0]*dep*target_soft/total;g->events[ne++]=seed;}
                    else memset(ev,0,13*sizeof(float));
                }
            }
            for(int e=0;e<ne;e++){
                float *ev=g->event+13*g->events[e];float q[3],pt[3],normal[3],neg[3];
                for(int k=0;k<3;k++){q[k]=ev[1+k]/ev[0];pt[k]=ev[4+k]/ev[0];normal[k]=ev[7+k]/ev[0];}
                float norm=mx(sqrtf(dot(normal,normal)),1e-12f);
                for(int k=0;k<3;k++){normal[k]/=norm;neg[k]=-normal[k];}
                float dep=ev[10]/ev[0],da=ev[11]/ev[0],db=ev[12]/ev[0];
                float width=g->rim+2.0f*sqrtf(.08f*dep);
                float wa=(source==6||source==7)?(.035f+1.5f*sqrtf(.012f*da)):width;
                float wb=(target==6||target==7)?(.035f+1.5f*sqrtf(.012f*db)):width;
                field(g,v,target,p,pt,neg,db,wb,&stats[1]);
                field(g,v,source,p,q,normal,da,wa,&stats[1]);
                stats[0]++;memset(ev,0,13*sizeof(float));
            }
        }
    }
    for(int t=0;t<g->nt;t++){
        int i=g->touched[t];float *d=out+3*i;
        for(int k=0;k<3;k++)d[k]=g->acc[3*i+k]/mx(g->weight[i],1e-25f);
        float scale=mn(1,g->rho*g->soft[i]/mx(sqrtf(dot(d,d)),1e-12f));for(int k=0;k<3;k++)d[k]*=scale;
        for(int k=g->offset[i];k<g->offset[i+1];k++){int f=g->adj[k];if(!g->fseen[f]){g->fseen[f]=1;g->fids[g->nft++]=f;}}
    }
    for(int it=0;it<g->nft;it++){
        int f=g->fids[it],a=g->faces[3*f],b=g->faces[3*f+1],c=g->faces[3*f+2];float *e=g->fgeom+10*f;
        for(int k=0;k<3;k++){e[k]=v[3*b+k]-v[3*a+k];e[3+k]=v[3*c+k]-v[3*a+k];}
        cross(e,e+3,e+6);e[9]=1/mx(dot(e+6,e+6),1e-20f);
    }
    /* Limit only the local triangles; a crossed finger must not scale down
     * an unrelated chest dent. Repeat the local bound, then certify the
     * interpolation with the same analytic oriented-area bound. */
    for(int pass=0;pass<16;pass++){
        for(int t=0;t<g->nt;t++)g->scale[g->touched[t]]=1;
        int limited=0;
        for(int it=0;it<g->nft;it++){
            int f=g->fids[it];float s=face_scale(g,out,f);
            if(s<.9999f){
                limited=1;stats[4]=mn(stats[4],s);
                for(int k=0;k<3;k++){int i=g->faces[3*f+k];if(g->seen[i])g->scale[i]=mn(g->scale[i],s);}
            }
        }
        if(!limited)break;
        for(int t=0;t<g->nt;t++){int i=g->touched[t];for(int k=0;k<3;k++)out[3*i+k]*=g->scale[i];}
    }
    float final=1;for(int it=0;it<g->nft;it++)final=mn(final,face_scale(g,out,g->fids[it]));stats[5]=final;
    if(final<1)for(int t=0;t<g->nt;t++)for(int k=0;k<3;k++)out[3*g->touched[t]+k]*=final;
    stats[2]=g->nt;stats[3]=g->nft;return 0;
}

#ifdef DEFORM_BENCHMARK
/* Optional Linux measurement adapter, excluded from a normal engine build.
 * Times the same public C call with preallocated buffers. Python allocation
 * and garbage collection are measured separately by the Python harness. */
static double wall_seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int deform_benchmark(void *ctx,int frames,int repeats,const float *v,const float *n,const float *cv,const float *cn,
    float *out,double *stats,double *milliseconds){
    Deform *g=ctx;if(!g||frames<1||repeats<1)return -1;
    for(int r=0;r<repeats;r++)for(int t=0;t<frames;t++){
        size_t a=(size_t)t*g->nv*3,b=(size_t)t*g->nc*3;
        double start=wall_seconds();int rc=deform_step(g,v+a,n+a,cv+b,cn+b,out+a,stats+8*t);
        milliseconds[r*frames+t]=(wall_seconds()-start)*1000;if(rc)return rc;
    }
    return 0;
}
#endif
