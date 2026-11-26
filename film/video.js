const {chromium}=require('playwright');const fs=require('fs');const {spawn}=require('child_process');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
p.on('pageerror',e=>console.log('ERR',e.message));
await p.goto('file://'+__dirname+'/render.html');await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(1500);
const total=await p.evaluate(([d,t,s])=>init(d,t,s),[JSON.parse(fs.readFileSync('data.json')),JSON.parse(fs.readFileSync('audio/timing.json')),JSON.parse(fs.readFileSync('script.json'))]);
const fps=30,nf=Math.ceil(total*fps);
const ff=spawn('ffmpeg',['-y','-hide_banner','-loglevel','error','-f','image2pipe','-framerate',String(fps),'-c:v','mjpeg','-i','-','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','video_only.mp4']);
ff.stderr.on('data',d=>process.stderr.write(d));
const t0=Date.now();
for(let f=0;f<nf;f++){const url=await p.evaluate(T=>frame(T,0.93),f/fps);const buf=Buffer.from(url.slice(url.indexOf(',')+1),'base64');
  if(!ff.stdin.write(buf))await new Promise(r=>ff.stdin.once('drain',r));
  if(f%300===0)console.log('frame',f,'/',nf,((Date.now()-t0)/1000).toFixed(0)+'s');}
ff.stdin.end();await new Promise(r=>ff.on('close',r));await b.close();console.log('done',((Date.now()-t0)/1000).toFixed(0)+'s');})();
