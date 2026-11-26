const fs=require('fs');const S=JSON.parse(fs.readFileSync(process.env.IN||'sweep.json'));
const P=process.env.PATS?process.env.PATS.split(','):['heart','smile','ring'],SD=[7,11,23];
const asc=g=>{let s='';for(let y=0;y<16;y++){s+='      '+g.slice(y*16,y*16+16).map(v=>' .:-=+*#%@'[Math.min(9,Math.floor(v*10))]).join('')+'\n';}return s;};
for(const m of S.meta){const R=S.res[m.id];const row=[];let err=0;
  const per={};for(const p of P){const v=SD.map(s=>R[p+'_'+s]);if(v.some(x=>x.error)){err++;continue;}
    per[p]={f:v.map(x=>Math.round(x.final*100)),hit:v.map(x=>x.hit95),sd:Math.max(...v.map(x=>x.sd100))};}
  const line=P.map(p=>per[p]?`${p}:${per[p].f.join('/')} hit:${per[p].hit.join('/')} osc:${(per[p].sd*100).toFixed(1)}`:p+':ERR').join(' | ');
  console.log(m.id.padEnd(12),m.label.padEnd(26),line);
  if(process.argv[2]&&process.argv[2].split(',').includes(m.id)){for(const p of P)console.log(p+'\n'+asc(R[p+'_7'].grid));}
}
