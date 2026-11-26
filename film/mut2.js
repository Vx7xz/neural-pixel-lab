// Simple formula mutations. Neuron: z = w·x + b, a = tanh(z). Update: w ← w − η·g
const N=16;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const HEART=["................","................","...###....###...","..#####..#####..",".##############.",".##############.",".##############.","..############..","...##########...","....########....",".....######.....","......####......",".......##.......","................","................","................"];
// neuron forms: returns z, and derivative helpers
const NEURON={
  base:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=w[k]*x[k];return s+b;}, dw:(w,x,b,k,S)=>x[k], dx:(w,x,b,k,S)=>w[k], db:(w,x,b,S)=>1},
  plus:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=w[k]+x[k];return s+b;}, dw:()=>1, dx:()=>1, db:()=>1},
  xsq:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=w[k]*x[k]*x[k];return s+b;}, dw:(w,x,b,k)=>x[k]*x[k], dx:(w,x,b,k)=>2*w[k]*x[k], db:()=>1},
  wsq:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=w[k]*w[k]*x[k];return s+b;}, dw:(w,x,b,k)=>2*w[k]*x[k], dx:(w,x,b,k)=>w[k]*w[k], db:()=>1},
  timesb:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=w[k]*x[k];return s*b;}, dw:(w,x,b,k)=>x[k]*b, dx:(w,x,b,k)=>w[k]*b, db:(w,x,b,S)=>S},
  swapwx:{z:(w,x,b)=>{let s=0;for(let k=0;k<x.length;k++)s+=Math.pow(Math.abs(x[k])+1e-9,w[k]);return s+b;}, dw:(w,x,b,k)=>Math.pow(Math.abs(x[k])+1e-9,w[k])*Math.log(Math.abs(x[k])+1e-9), dx:(w,x,b,k)=>w[k]*Math.pow(Math.abs(x[k])+1e-9,w[k]-1)*Math.sign(x[k]||1), db:()=>1},
};
const ACT={tanh:{f:Math.tanh,d:(z,a)=>1-a*a},linear:{f:z=>z,d:()=>1},sq:{f:z=>z*z,d:z=>2*z},sqrt:{f:z=>Math.sign(z)*Math.sqrt(Math.abs(z)),d:z=>0.5/Math.sqrt(Math.abs(z)+1e-4)},abs:{f:Math.abs,d:z=>Math.sign(z)}};
const UPD={
  base:(w,g,lr)=>w-lr*g, flip:(w,g,lr)=>w+lr*g, gsq:(w,g,lr)=>w-lr*g*g, groot:(w,g,lr)=>w-lr*Math.sqrt(Math.abs(g)),
  div:(w,g,lr)=>w-lr/(g||1e-9), swap:(w,g,lr)=>g-lr*w, wonly:(w,g,lr)=>w-lr*w, times:(w,g,lr)=>w*lr*g, etasq:(w,g,lr)=>w-lr*lr*g,
};
function run(opt){
  const sizes=[2,16,16,1],L=3,rnd=mulberry32(7),nf=NEURON[opt.neuron||'base'],act=ACT[opt.act||'tanh'],upd=UPD[opt.upd||'base'],lr=opt.lr??1.0;
  const W=[],b=[];for(let l=0;l<L;l++){const i=sizes[l],o=sizes[l+1],lim=Math.sqrt(6/(i+o));W.push([]);b.push(new Float64Array(o).fill(opt.b0||0));for(let j=0;j<o;j++)W[l].push(Float64Array.from({length:i},()=>(rnd()*2-1)*lim));}
  const X=[],Y=[];for(let y=0;y<N;y++)for(let x=0;x<N;x++){X.push([(x+.5)/N*2-1,(y+.5)/N*2-1]);Y.push(HEART[y][x]==='#'?1:0);}
  const a=sizes.map(s=>new Float64Array(s)),z=sizes.map(s=>new Float64Array(s)),S=sizes.map(s=>new Float64Array(s));
  const last=L-1,nfl=l=>l===last?NEURON.base:nf; // output neuron stays normal
  function fwd(x,y){a[0][0]=x;a[0][1]=y;for(let l=0;l<L;l++){const f=nfl(l);for(let j=0;j<sizes[l+1];j++){let s=0;for(let k=0;k<sizes[l];k++)s+=W[l][j][k]*a[l][k];S[l+1][j]=s;let zz=f.z(W[l][j],a[l],b[l][j]);if(!isFinite(zz))zz=0;zz=Math.max(-50,Math.min(50,zz));z[l+1][j]=zz;a[l+1][j]=l===last?1/(1+Math.exp(-zz)):act.f(zz);}}return a[L][0];}
  const frames=[],snap=new Set(opt.snaps);
  for(let e=0;e<=opt.epochs;e++){
    if(snap.has(e)){const g=[];let acc=0,loss=0;for(let s=0;s<256;s++){let p=fwd(X[s][0],X[s][1]);if(!isFinite(p))p=.5;g.push(p);if((p>.5)===(Y[s]===1))acc++;const pc=Math.min(1-1e-7,Math.max(1e-7,p));loss-=Y[s]*Math.log(pc)+(1-Y[s])*Math.log(1-pc);}frames.push({e,g,acc:acc/256,loss:loss/256});}
    if(e===opt.epochs)break;
    const gW=W.map(l=>l.map(r=>new Float64Array(r.length))),gb=b.map(v=>new Float64Array(v.length));
    for(let s=0;s<256;s++){const p=fwd(X[s][0],X[s][1]);let d=[p-Y[s]];
      for(let l=L-1;l>=0;l--){const f=nfl(l),i=sizes[l],o=sizes[l+1],din=new Float64Array(i);
        for(let j=0;j<o;j++){const dj=d[j];if(!dj)continue;gb[l][j]+=dj*f.db(W[l][j],a[l],b[l][j],S[l+1][j]);
          for(let k=0;k<i;k++){gW[l][j][k]+=dj*f.dw(W[l][j],a[l],b[l][j],k);din[k]+=dj*f.dx(W[l][j],a[l],b[l][j],k);}}
        if(l>0){d=Array.from(din,(v,k)=>v*act.d(z[l][k],a[l][k]));}}}
    for(let l=0;l<L;l++){for(let j=0;j<sizes[l+1];j++){for(let k=0;k<sizes[l];k++){let v=upd(W[l][j][k],gW[l][j][k]/256,lr);W[l][j][k]=isFinite(v)?Math.max(-1e3,Math.min(1e3,v)):0;}
      let v=upd(b[l][j],gb[l][j]/256,lr);b[l][j]=isFinite(v)?Math.max(-1e3,Math.min(1e3,v)):0;}}
  }
  return frames;
}
module.exports={run,HEART,N};
if(require.main===module){
  const E={base:{},flip:{upd:'flip'},gsq:{upd:'gsq'},groot:{upd:'groot',lr:0.05},div:{upd:'div',lr:1e-5},swap:{upd:'swap'},wonly:{upd:'wonly',lr:0.05},times:{upd:'times'},etasq:{upd:'etasq',lr:0.1},
    plus:{neuron:'plus'},xsq:{neuron:'xsq'},wsq:{neuron:'wsq'},timesb:{neuron:'timesb',b0:1},swapwx:{neuron:'swapwx'},linear:{act:'linear'},actsq:{act:'sq'},actabs:{act:'abs'}};
  const only=process.argv[2]?process.argv[2].split(','):Object.keys(E);
  for(const k of only){const f=run({...E[k],epochs:800,snaps:[0,50,200,800]});console.log(k.padEnd(8),f.map(x=>Math.round(x.acc*100)).join(' '));
    const g=f[f.length-1].g;for(let y=0;y<N;y++)console.log('   '+g.slice(y*N,y*N+N).map(v=>' .:-=+*#%@'[Math.min(9,Math.floor(v*10))]).join(''));}
}
