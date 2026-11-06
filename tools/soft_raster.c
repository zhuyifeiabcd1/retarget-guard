/* Small CPU triangle rasterizer with Gouraud shading and a z-buffer.
   Occluded limb surface flagged as penetrating is drawn through at 75 % red.
   Used only to render the demonstration video. */
#include <stdlib.h>
#include <float.h>
static inline float mn3(float a,float b,float c){float m=a<b?a:b;return m<c?m:c;}
static inline float mx3(float a,float b,float c){float m=a>b?a:b;return m>c?m:c;}
void raster(int W,int H,const float*p,const float*c,const float*hot,int nf,const int*f,const unsigned char*limb,
            float tol,const float*bg,float*img){
  float*z=malloc(sizeof(float)*W*H),*z2=malloc(sizeof(float)*W*H);
  float*red=calloc(W*H*3,sizeof(float));
  for(int i=0;i<W*H;i++){z[i]=FLT_MAX;z2[i]=FLT_MAX;img[3*i]=bg[0];img[3*i+1]=bg[1];img[3*i+2]=bg[2];}
  for(int pass=0;pass<2;pass++)for(int t=0;t<nf;t++){
    int a=f[3*t],b=f[3*t+1],d=f[3*t+2];
    if(pass==1&&(!limb[t]||(hot[a]<=tol&&hot[b]<=tol&&hot[d]<=tol)))continue;
    float x0=p[3*a],y0=p[3*a+1],z0=p[3*a+2],x1=p[3*b],y1=p[3*b+1],z1=p[3*b+2],x2=p[3*d],y2=p[3*d+1],z2v=p[3*d+2];
    float area=(x1-x0)*(y2-y0)-(y1-y0)*(x2-x0);
    if(area==0)continue;
    int xa=(int)mn3(x0,x1,x2),xb=(int)mx3(x0,x1,x2)+1,ya=(int)mn3(y0,y1,y2),yb=(int)mx3(y0,y1,y2)+1;
    if(xa<0)xa=0;if(ya<0)ya=0;if(xb>=W)xb=W-1;if(yb>=H)yb=H-1;
    for(int y=ya;y<=yb;y++)for(int x=xa;x<=xb;x++){
      float px=x+.5f,py=y+.5f;
      float w0=((x1-px)*(y2-py)-(y1-py)*(x2-px))/area,w1=((x2-px)*(y0-py)-(y2-py)*(x0-px))/area,w2=1-w0-w1;
      if(w0<0||w1<0||w2<0)continue;
      float zz=w0*z0+w1*z1+w2*z2v;int k=y*W+x;
      if(pass==0){if(zz<z[k]){z[k]=zz;for(int e=0;e<3;e++)img[3*k+e]=w0*c[3*a+e]+w1*c[3*b+e]+w2*c[3*d+e];}}
      else if(zz<z2[k]&&zz>z[k]+.002f){z2[k]=zz;for(int e=0;e<3;e++)red[3*k+e]=w0*c[3*a+e]+w1*c[3*b+e]+w2*c[3*d+e];}
    }
  }
  for(int k=0;k<W*H;k++)if(z2[k]<FLT_MAX)for(int e=0;e<3;e++)img[3*k+e]=.25f*img[3*k+e]+.75f*red[3*k+e];
  free(z);free(z2);free(red);
}
