const fs=require('fs');const {PAT}=require('./mut3.js');
const A=JSON.parse(fs.readFileSync('sweep.json')),B=JSON.parse(fs.readFileSync('sweep_asym.json'));
const enc=g=>g.map(v=>Math.max(0,Math.min(15,Math.round(v*15))).toString(16)).join('');
const tgt=p=>PAT[p].join('').split('').map(c=>c==='#'?'f':'0').join('');
const SEEDS=[7,11,23];
function summ(R,pats){const o={};for(const p of pats){const v=SEEDS.map(s=>R[p+'_'+s]);
  const f=v.map(x=>x.final*100),h=v.map(x=>x.hit95).filter(x=>x>=0).sort((a,b)=>a-b);
  o[p]={m:+(f.reduce((a,b)=>a+b)/3).toFixed(1),lo:Math.round(Math.min(...f)),hi:Math.round(Math.max(...f)),hit:h.length===3?h[1]:null,nh:h.length,osc:+(Math.max(...v.map(x=>x.sd100))*100).toFixed(1),g:enc(v[0].grid)};}return o;}
const out={targets:{},exps:[]};for(const p of ['heart','smile','ring','R','seven'])out.targets[p]=tgt(p);
for(const m of A.meta){const e={id:m.id,grp:m.grp,label:m.label,r:summ(A.res[m.id],['heart','smile','ring'])};
  if(B.res[m.id])Object.assign(e.r,summ(B.res[m.id],['R','seven']));out.exps.push(e);}
out.lr={};for(const m of B.meta.filter(m=>m.grp==='LR'))out.lr[m.id]={label:m.label,r:summ(B.res[m.id],['R','seven'])};
const s=JSON.stringify(out);fs.writeFileSync('atlas_data.json',s);console.log('bytes',s.length);
