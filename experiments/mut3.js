// Broad formula-mutation lab. One symbol edit per experiment.
// Network 2-16-16-1. Hidden: z = w·x + b, a = tanh(z). Output: sigmoid. Loss: BCE, so dL/dz_out = p − y.
// Backprop: δ_l = (Wᵀ δ_{l+1}) ⊙ f'(z).  Update: w ← w − η·g.
const N=16;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const PAT={
 heart:["................","................","...###....###...","..#####..#####..",".##############.",".##############.",".##############.","..############..","...##########...","....########....",".....######.....","......####......",".......##.......","................","................","................"],
 smile:["................",".....######.....","...##########...","..############..",".####..##..####.",".####..##..####.","################","################","################",".##.########.##.",".###.######.###.","..###......###..","...##########...",".....######.....","................","................"],
};
PAT.R=["................","..##########....","..###....####...","..###.....###...","..###.....###...","..###....####...","..##########....","..#########.....","..###...###.....","..###....###....","..###.....###...","..###......###..","..###.......###.","..###.......###.","................","................"];
PAT.seven=["................",".##############.",".##############.","...........###..","..........###...",".........###....","........###.....",".......###......","......###.......",".....###........","....###.........","...###..........","..###...........","..###...........","................","................"];
(function(){const r=[];for(let y=0;y<N;y++){let s='';for(let x=0;x<N;x++){const u=(x+.5)/N*2-1,v=(y+.5)/N*2-1,d=Math.hypot(u,v);s+=d>.4&&d<.85?'#':'.';}r.push(s);}PAT.ring=r;})();

const sgn=v=>v>0?1:v<0?-1:0;
// neuron forms (hidden layers only): z(w,x,b) summed over k, with partials
const NEU={
 base:  {t:(w,x)=>w*x,        dw:(w,x)=>x,        dx:(w,x)=>w},
 plus:  {t:(w,x)=>w+x,        dw:()=>1,           dx:()=>1},
 xsq:   {t:(w,x)=>w*x*x,      dw:(w,x)=>x*x,      dx:(w,x)=>2*w*x},
 xcube: {t:(w,x)=>w*x*x*x,    dw:(w,x)=>x*x*x,    dx:(w,x)=>3*w*x*x},
 xabs:  {t:(w,x)=>w*Math.abs(x),dw:(w,x)=>Math.abs(x),dx:(w,x)=>w*sgn(x)},
 xsin:  {t:(w,x)=>w*Math.sin(3*x),dw:(w,x)=>Math.sin(3*x),dx:(w,x)=>3*w*Math.cos(3*x)},
 wsq:   {t:(w,x)=>w*w*x,      dw:(w,x)=>2*w*x,    dx:(w,x)=>w*w},
 wabs:  {t:(w,x)=>Math.abs(w)*x,dw:(w,x)=>sgn(w)*x,dx:(w,x)=>Math.abs(w)},
 minus: {t:(w,x)=>w-x,        dw:()=>1,           dx:()=>-1},
 wxsq:  {t:(w,x)=>(w*x)*(w*x),dw:(w,x)=>2*w*x*x,  dx:(w,x)=>2*w*w*x},
 wminx2:{t:(w,x)=>-(w-x)*(w-x),dw:(w,x)=>-2*(w-x),dx:(w,x)=>2*(w-x)},   // distance neuron (RBF-like)
};
// bias placement: 'add' z=Σ+b ; 'mul' z=(Σ)·b ; 'paren' z=Σ w(x+b) handled via flag
const ACT={
 tanh:  [Math.tanh,(z,a)=>1-a*a],
 linear:[z=>z,()=>1],
 sq:    [z=>z*z,z=>2*z],
 cube:  [z=>z*z*z,z=>3*z*z],
 abs:   [Math.abs,z=>sgn(z)],
 sin:   [Math.sin,z=>Math.cos(z)],
 cos:   [Math.cos,z=>-Math.sin(z)],
 relu:  [z=>z>0?z:0,z=>z>0?1:0],
 step:  [z=>z>0?1:-1,()=>0],
 tanh2: [z=>Math.tanh(z)**2,(z,a)=>{const t=Math.tanh(z);return 2*t*(1-t*t);}],
 bump:  [z=>1/(1+z*z),z=>-2*z/((1+z*z)**2)],
 exp:   [z=>Math.exp(Math.min(z,20)),(z,a)=>a],
 tanhz2:[z=>Math.tanh(z*z),z=>{const t=Math.tanh(z*z);return (1-t*t)*2*z;}],
};
// derivative-formula edits (used in backprop instead of the true f')
const DER={
 true:null,
 onePlus:(z,a)=>1+a*a, asq:(z,a)=>a*a, oneMinusA:(z,a)=>1-a, one:()=>1, sq:(z,a)=>(1-a*a)**2, neg:(z,a)=>-(1-a*a),
};
const UPD={
 base:(w,g,h)=>w-h*g, flip:(w,g,h)=>w+h*g, gsq:(w,g,h)=>w-h*g*g, gcube:(w,g,h)=>w-h*g*g*g,
 groot:(w,g,h)=>w-h*Math.sqrt(Math.abs(g)), gcbrt:(w,g,h)=>w-h*Math.cbrt(g), sign:(w,g,h)=>w-h*sgn(g),
 div:(w,g,h)=>w-h/(g||1e-12), swap:(w,g,h)=>g-h*w, negw:(w,g,h)=>-w-h*g, mulw:(w,g,h)=>w-h*g*w,
 overw:(w,g,h)=>w-h*g/(Math.abs(w)<1e-3?1e-3*sgn(w||1):w), times:(w,g,h)=>w*h*g, tanhg:(w,g,h)=>w-h*Math.tanh(g),
 gabs:(w,g,h)=>w-h*Math.abs(g), half:(w,g,h)=>w/2-h*g, double:(w,g,h)=>2*w-h*g, wsq:(w,g,h)=>w*w-h*g,
};
const LOSSG={base:(p,y)=>p-y, cube:(p,y)=>(p-y)**3, sign:(p,y)=>sgn(p-y), sq:(p,y)=>(p-y)**2, swap:(p,y)=>y-p, ratio:(p,y)=>(p-y)/(p*(1-p)+1e-6)};
const BACK={W:0,B:1,signW:2,absW:3,W2:4,negW:5};

function run(o){
  const pat=PAT[o.pat||'heart'],E=o.epochs||1000,lr=o.lr??1.0,neu=NEU[o.neuron||'base'],act=ACT[o.act||'tanh'],der=o.der&&o.der!=='true'?DER[o.der]:act[1];
  const upd=UPD[o.upd||'base'],lg=LOSSG[o.loss||'base'],back=BACK[o.back||'W'],bmode=o.bias||'add';
  const sizes=[2,16,16,1],L=3,rnd=mulberry32(o.seed||7);
  const W=[],B=[],b=[];
  for(let l=0;l<L;l++){const i=sizes[l],k=sizes[l+1],lim=Math.sqrt(6/(i+k));
    W.push(Float64Array.from({length:k*i},()=>(rnd()*2-1)*lim));B.push(Float64Array.from({length:k*i},()=>(rnd()*2-1)*lim));
    b.push(new Float64Array(k).fill(bmode==='mul'?1:0));}
  const X=[],Y=[];for(let y=0;y<N;y++)for(let x=0;x<N;x++){X.push([(x+.5)/N*2-1,(y+.5)/N*2-1]);Y.push(pat[y][x]==='#'?1:0);}
  const a=sizes.map(s=>new Float64Array(s)),z=sizes.map(s=>new Float64Array(s)),S=sizes.map(s=>new Float64Array(s));
  const isOut=l=>l===L-1;
  function fwd(x0,y0){a[0][0]=x0;a[0][1]=y0;
    for(let l=0;l<L;l++){const i=sizes[l],k=sizes[l+1],Wl=W[l],out=isOut(l);
      for(let j=0;j<k;j++){let s=0;const r=j*i;
        if(out||neu===NEU.base){if(bmode==='paren'&&!out){for(let q=0;q<i;q++)s+=Wl[r+q]*(a[l][q]+b[l][j]);}else for(let q=0;q<i;q++)s+=Wl[r+q]*a[l][q];}
        else{for(let q=0;q<i;q++)s+=neu.t(Wl[r+q],a[l][q]);}
        S[l+1][j]=s;let zz;
        if(out)zz=s+b[l][j];else if(bmode==='mul')zz=s*b[l][j];else if(bmode==='paren')zz=s;else zz=s+b[l][j];
        if(!isFinite(zz))zz=0;zz=Math.max(-60,Math.min(60,zz));z[l+1][j]=zz;
        a[l+1][j]=out?1/(1+Math.exp(-zz)):act[0](zz);}}
    return a[L][0];}
  const accs=new Float32Array(E+1);let grid=null;const gW=W.map(w=>new Float64Array(w.length)),gb=b.map(v=>new Float64Array(v.length));
  const snaps={};const snapAt=new Set(o.snaps||[]);
  for(let e=0;e<=E;e++){
    gW.forEach(v=>v.fill(0));gb.forEach(v=>v.fill(0));let acc=0;const g=(e===E||snapAt.has(e))?new Float32Array(256):null;
    for(let s=0;s<256;s++){let p=fwd(X[s][0],X[s][1]);if(!isFinite(p))p=.5;if(g)g[s]=p;if((p>.5)===(Y[s]===1))acc++;
      let d=new Float64Array([lg(p,Y[s])]);
      for(let l=L-1;l>=0;l--){const i=sizes[l],k=sizes[l+1],Wl=W[l],out=isOut(l),din=new Float64Array(i);
        for(let j=0;j<k;j++){const dj=d[j];if(!dj||!isFinite(dj))continue;const r=j*i;
          if(out||neu===NEU.base){const bm=(!out&&bmode==='mul')?b[l][j]:1;
            for(let q=0;q<i;q++){const xin=(bmode==='paren'&&!out)?a[l][q]+b[l][j]:a[l][q];gW[l][r+q]+=dj*xin*bm;
              const Bq=back===0?Wl[r+q]:back===1?B[l][r+q]:back===2?sgn(Wl[r+q]):back===3?Math.abs(Wl[r+q]):back===4?Wl[r+q]*Wl[r+q]:-Wl[r+q];
              din[q]+=dj*Bq*bm;}
            gb[l][j]+=out?dj:bmode==='mul'?dj*S[l+1][j]:bmode==='paren'?dj*(()=>{let t=0;for(let q=0;q<i;q++)t+=Wl[r+q];return t;})():dj;}
          else{for(let q=0;q<i;q++){gW[l][r+q]+=dj*neu.dw(Wl[r+q],a[l][q]);din[q]+=dj*neu.dx(Wl[r+q],a[l][q]);}
            gb[l][j]+=bmode==='mul'?dj*S[l+1][j]:dj;}}
        if(l>0){d=new Float64Array(i);for(let q=0;q<i;q++)d[q]=din[q]*der(z[l][q],a[l][q]);}}}
    accs[e]=acc/256;if(g){if(e===E)grid=g;else snaps[e]=Array.from(g,v=>+v.toFixed(3));}
    if(e===E)break;
    for(let l=0;l<L;l++){for(let q=0;q<W[l].length;q++){let v=upd(W[l][q],gW[l][q]/256,lr);W[l][q]=isFinite(v)?Math.max(-1e4,Math.min(1e4,v)):0;}
      for(let j=0;j<b[l].length;j++){let v=upd(b[l][j],gb[l][j]/256,lr);b[l][j]=isFinite(v)?Math.max(-1e4,Math.min(1e4,v)):0;}}
  }
  let hit=-1;for(let e=0;e<=E;e++)if(accs[e]>=0.95){hit=e;break;}
  let m=0,sd=0;for(let e=E-99;e<=E;e++)m+=accs[e];m/=100;for(let e=E-99;e<=E;e++)sd+=(accs[e]-m)**2;sd=Math.sqrt(sd/100);
  let best=0;for(const v of accs)best=Math.max(best,v);
  return {final:accs[E],mean100:m,sd100:sd,best,hit95:hit,grid:Array.from(grid,v=>+v.toFixed(3)),snaps,curve:Array.from(accs.filter((_,i)=>i%10===0),v=>+v.toFixed(3))};
}
module.exports={run,PAT,N};
