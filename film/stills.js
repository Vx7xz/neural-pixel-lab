const {chromium}=require('playwright');const fs=require('fs');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
p.on('pageerror',e=>console.log('ERR',e.message));p.on('console',m=>{if(m.type()==='error')console.log('CONSOLE',m.text())});
await p.goto('file://'+__dirname+'/render.html');await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(1500);
const total=await p.evaluate(([d,t,s])=>init(d,t,s),[JSON.parse(fs.readFileSync('data.json')),JSON.parse(fs.readFileSync('audio/timing.json')),JSON.parse(fs.readFileSync('script.json'))]);
console.log('total',total);
const sc=await p.evaluate(()=>TLB.scenes.map(s=>({id:s.id,start:+s.start.toFixed(2),dur:+s.dur.toFixed(2),mut:s.mutT,k:s.k})));console.log(JSON.stringify(sc));
const times=(process.argv[2]||'8,30,60').split(',').map(Number);
for(const T of times){const url=await p.evaluate(T=>frame(T,0.85),T);fs.writeFileSync(`still_${T}.jpg`,Buffer.from(url.split(',')[1],'base64'));}
await b.close();})();
