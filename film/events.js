const TL=require('./timeline.js'),fs=require('fs');
const D=JSON.parse(fs.readFileSync('data.json')),tm=JSON.parse(fs.readFileSync('audio/timing.json'));
const R={};for(const k in D.runs){R[k]={g:Buffer.from(D.runs[k].g,'base64'),acc:D.runs[k].acc,n:D.runs[k].acc.length};}
const B=TL.build(tm),fps=30,out={total:B.total,scenes:[],frames:[]};
for(const sc of B.scenes)out.scenes.push({id:sc.id,start:sc.start,dur:sc.dur,lead:sc.lead,mutT:sc.mutT??null,boomT:sc.boomT??null,sent:sc.sent});
let prev=null;
for(let f=0;f<Math.ceil(B.total*fps);f++){const T=f/fps;let sc=B.scenes[0];for(const s of B.scenes)if(T>=s.start)sc=s;const t=T-sc.start;
  if(!sc.k){out.frames.push([0,0]);prev=null;continue;}
  const ep=TL.epochAt(sc,t),r=R[sc.run],i=Math.max(0,Math.min(r.n-1,Math.round(ep/2)));const g=r.g.subarray(i*256,i*256+256);
  let act=0;if(prev&&prev.run===sc.run){for(let j=0;j<256;j++)act+=Math.abs(g[j]-prev.g[j]);act/=256*255;}
  out.frames.push([+act.toFixed(4),r.acc[i]]);prev={run:sc.run,g:Buffer.from(g)};}
fs.writeFileSync('events.json',JSON.stringify(out));console.log('total',B.total.toFixed(2),'frames',out.frames.length);
