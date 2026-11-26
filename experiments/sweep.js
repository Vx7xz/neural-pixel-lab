const {Worker,isMainThread,parentPort,workerData}=require('worker_threads');const fs=require('fs');
const EXP=[
 ['base','ORIGINAL','w ← w − η·g',{}],
 // update rule
 ['u_flip','UPDATE','− → +  (w + η·g)',{upd:'flip'}],
 ['u_gsq','UPDATE','g → g²',{upd:'gsq'}],
 ['u_gcube','UPDATE','g → g³',{upd:'gcube'}],
 ['u_groot','UPDATE','g → √|g|',{upd:'groot'}],
 ['u_gcbrt','UPDATE','g → ∛g',{upd:'gcbrt'}],
 ['u_sign','UPDATE','g → sign(g)',{upd:'sign'}],
 ['u_sign01','UPDATE','g → sign(g), η=0.01',{upd:'sign',lr:0.01}],
 ['u_gabs','UPDATE','g → |g|',{upd:'gabs'}],
 ['u_tanhg','UPDATE','g → tanh(g)',{upd:'tanhg'}],
 ['u_div','UPDATE','· → ÷  (w − η/g)',{upd:'div',lr:1e-5}],
 ['u_swap','UPDATE','w ⇄ g  (g − η·w)',{upd:'swap'}],
 ['u_negw','UPDATE','w → −w  (−w − η·g)',{upd:'negw'}],
 ['u_mulw','UPDATE','g → g·w',{upd:'mulw'}],
 ['u_overw','UPDATE','g → g/w',{upd:'overw'}],
 ['u_times','UPDATE','− → ×  (w × η·g)',{upd:'times'}],
 ['u_half','UPDATE','w → w/2',{upd:'half'}],
 ['u_double','UPDATE','w → 2w',{upd:'double'}],
 ['u_wsq','UPDATE','w → w²',{upd:'wsq'}],
 // neuron
 ['n_plus','NEURON','w·x → w+x',{neuron:'plus'}],
 ['n_minus','NEURON','w·x → w−x',{neuron:'minus'}],
 ['n_xsq','NEURON','x → x²',{neuron:'xsq'}],
 ['n_xcube','NEURON','x → x³',{neuron:'xcube'}],
 ['n_xabs','NEURON','x → |x|',{neuron:'xabs'}],
 ['n_xsin','NEURON','x → sin(3x)',{neuron:'xsin'}],
 ['n_wsq','NEURON','w → w²',{neuron:'wsq'}],
 ['n_wabs','NEURON','w → |w|',{neuron:'wabs'}],
 ['n_wxsq','NEURON','w·x → (w·x)²',{neuron:'wxsq'}],
 ['n_dist','NEURON','w·x → −(w−x)²',{neuron:'wminx2'}],
 ['n_timesb','NEURON','+b → ×b',{bias:'mul'}],
 ['n_paren','NEURON','w·x+b → w·(x+b)',{bias:'paren'}],
 // activation
 ['a_linear','ACTIVATION','tanh(z) → z',{act:'linear'}],
 ['a_sq','ACTIVATION','tanh(z) → z²',{act:'sq'}],
 ['a_cube','ACTIVATION','tanh(z) → z³',{act:'cube'}],
 ['a_abs','ACTIVATION','tanh(z) → |z|',{act:'abs'}],
 ['a_sin','ACTIVATION','tanh(z) → sin(z)',{act:'sin'}],
 ['a_cos','ACTIVATION','tanh(z) → cos(z)',{act:'cos'}],
 ['a_relu','ACTIVATION','tanh(z) → max(0,z)',{act:'relu'}],
 ['a_step','ACTIVATION','tanh(z) → sign(z)',{act:'step'}],
 ['a_tanh2','ACTIVATION','tanh(z) → tanh(z)²',{act:'tanh2'}],
 ['a_tanhz2','ACTIVATION','tanh(z) → tanh(z²)',{act:'tanhz2'}],
 ['a_bump','ACTIVATION','tanh(z) → 1/(1+z²)',{act:'bump'}],
 ['a_exp','ACTIVATION','tanh(z) → eᶻ',{act:'exp'}],
 // derivative in backprop
 ['d_onePlus','DERIVATIVE','1−a² → 1+a²',{der:'onePlus'}],
 ['d_asq','DERIVATIVE','1−a² → a²',{der:'asq'}],
 ['d_oneMinusA','DERIVATIVE','1−a² → 1−a',{der:'oneMinusA'}],
 ['d_one','DERIVATIVE','1−a² → 1',{der:'one'}],
 ['d_sq','DERIVATIVE','1−a² → (1−a²)²',{der:'sq'}],
 ['d_neg','DERIVATIVE','1−a² → −(1−a²)',{der:'neg'}],
 // backward matrix
 ['b_B','BACKPROP','Wᵀ → random B',{back:'B'}],
 ['b_signW','BACKPROP','Wᵀ → sign(W)ᵀ',{back:'signW'}],
 ['b_absW','BACKPROP','Wᵀ → |W|ᵀ',{back:'absW'}],
 ['b_W2','BACKPROP','Wᵀ → (W²)ᵀ',{back:'W2'}],
 ['b_negW','BACKPROP','Wᵀ → −Wᵀ',{back:'negW'}],
 // loss gradient
 ['l_cube','LOSS','(p−y) → (p−y)³',{loss:'cube'}],
 ['l_sign','LOSS','(p−y) → sign(p−y)',{loss:'sign'}],
 ['l_sq','LOSS','(p−y) → (p−y)²',{loss:'sq'}],
 ['l_ratio','LOSS','(p−y) → (p−y)/(p(1−p))',{loss:'ratio'}],
];
const PATS=process.env.PATS?process.env.PATS.split(','):['heart','smile','ring'],SEEDS=[7,11,23];
if(process.env.ONLY){const keep=new Set(process.env.ONLY.split(','));for(let i=EXP.length-1;i>=0;i--)if(!keep.has(EXP[i][0]))EXP.splice(i,1);}
if(process.env.EXTRA){for(const x of JSON.parse(process.env.EXTRA))EXP.push(x);}
if(isMainThread){
  const jobs=[];for(const [id,,,o] of EXP)for(const pat of PATS)for(const seed of SEEDS)jobs.push({id,o:{...o,pat,seed,epochs:1000}});
  const res={};let done=0,next=0;const t0=Date.now();const NW=4;
  const finish=()=>{const meta=EXP.map(([id,grp,label])=>({id,grp,label}));fs.writeFileSync(process.env.OUT||'sweep.json',JSON.stringify({meta,res}));console.log('ALL DONE',((Date.now()-t0)/1000).toFixed(0)+'s');};
  for(let w=0;w<NW;w++){const wk=new Worker(__filename,{workerData:null});
    const feed=()=>{if(next<jobs.length)wk.postMessage(jobs[next++]);else wk.terminate();};
    wk.on('message',m=>{(res[m.id]??={})[m.pat+'_'+m.seed]=m.r;done++;if(done%60===0)console.log('done',done,'/',jobs.length,((Date.now()-t0)/1000).toFixed(0)+'s');if(done===jobs.length)finish();feed();});
    wk.on('error',e=>console.log('WORKER ERROR',e.message));feed();}
}else{const {run}=require('./mut3.js');parentPort.on('message',j=>{let r;try{r=run(j.o);}catch(e){r={error:e.message};}parentPort.postMessage({id:j.id,pat:j.o.pat,seed:j.o.seed,r});});}
