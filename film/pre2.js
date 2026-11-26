const {run}=require('./mut2.js');const fs=require('fs');
const R={normal:{},flip:{upd:'flip'},gsq:{upd:'gsq'},groot:{upd:'groot',lr:0.05},div:{upd:'div',lr:1e-5},swap:{upd:'swap'},
 plus:{neuron:'plus'},xsq:{neuron:'xsq'},wsq:{neuron:'wsq'},timesb:{neuron:'timesb',b0:1},abs:{act:'abs'}};
const E=1200,snaps=[];for(let e=0;e<=E;e+=2)snaps.push(e);
const out={step:2,runs:{}};
for(const k in R){const f=run({...R[k],epochs:E,snaps});
 out.runs[k]={g:Buffer.from(f.flatMap(x=>x.g.map(v=>Math.round(Math.max(0,Math.min(1,isFinite(v)?v:.5))*255)))).toString('base64'),loss:f.map(x=>+Math.min(20,x.loss).toFixed(4)),acc:f.map(x=>+x.acc.toFixed(4))};
 const a=f.map(x=>x.acc);console.log(k.padEnd(7),[0,20,50,100,200,400,800,1200].map(e=>e+':'+Math.round(a[e/2]*100)).join(' '));}
fs.writeFileSync('data.json',JSON.stringify(out));
