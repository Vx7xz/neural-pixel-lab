// Shared timeline: used by the renderer (browser) and the audio builder (node).
(function(root){
const LEAD=0.9, TAIL=2.4, FPS=30;
const U=['w ← w − η · g'],Z=['z = w · x + b'];
const up=(chap,chapAr,f,note,mutDelay=0.9)=>({chap,chapAr,f,note,ref:'normal',mut:s=>s(0)+mutDelay,keys:s=>[[s(0)+1.3,0],['end',1200]],pow:2.0});
const SCENES=[
 {id:'intro', run:'normal'},
 {id:'normal', run:'normal', chap:'THE ORIGINAL · UPDATE RULE', chapAr:'الأصل · قاعدة التحديث', f:[['w ← w '],['−',null,'hi'],[' η · g']], note:'g = gradient of the error', keys:s=>[[s(0)+1.0,0],['end',1200]], pow:2.0},
 {id:'flip', run:'flip', ...up('UPDATE RULE · FLIP','قاعدة التحديث · قلب الإشارة',[['w ← w '],['−','+'],[' η · g']],'minus becomes plus')},
 {id:'gsq', run:'gsq', ...up('UPDATE RULE · SQUARE','قاعدة التحديث · تربيع',[['w ← w − η · '],['g','g²']],'the gradient, squared')},
 {id:'groot', run:'groot', ...up('UPDATE RULE · ROOT','قاعدة التحديث · جذر',[['w ← w − η · '],['g','√|g|']],'square root of the gradient')},
 {id:'div', run:'div', ...up('UPDATE RULE · DIVIDE','قاعدة التحديث · قسمة',[['w ← w − η '],['·','÷'],[' g']],'times becomes divide')},
 {id:'swap', run:'swap', ...up('UPDATE RULE · SWAP','قاعدة التحديث · تبديل',[['w ← '],['w','g'],[' − η · '],['g','w']],'weight and gradient trade places')},
 {id:'plus', run:'plus', ...up('NEURON · TIMES TO PLUS','العصبون · ضرب إلى جمع',[['z = w '],['·','+'],[' x + b']],'inputs no longer multiply')},
 {id:'xsq', run:'xsq', ...up('NEURON · SQUARE THE INPUT','العصبون · تربيع المدخل',[['z = w · '],['x','x²'],[' + b']],'the input, squared')},
 {id:'wsq', run:'wsq', ...up('NEURON · SQUARE THE WEIGHT','العصبون · تربيع الوزن',[['z = '],['w','w²'],[' · x + b']],'the weight, squared')},
 {id:'timesb', run:'timesb', ...up('NEURON · PLUS TO TIMES','العصبون · جمع إلى ضرب',[['z = w · x '],['+','×'],[' b']],'multiply by the bias')},
 {id:'abs', run:'abs', ...up('ACTIVATION · ABSOLUTE','دالة التفعيل · القيمة المطلقة',[['a = '],['tanh(z)','|z|']],'a sharp V instead of a soft S',1.6)},
 {id:'outro', run:'normal'}
];
const OUTRO_GRID=[['normal',1200,'ORIGINAL'],['flip',1200,'− → +'],['gsq',1200,'g → g²'],['groot',1200,'g → √|g|'],['div',1200,'· → ÷'],['swap',1200,'w ⇄ g'],
  ['plus',1200,'· → +'],['xsq',1200,'x → x²'],['wsq',1200,'w → w²'],['timesb',1200,'+ → ×'],['abs',1200,'tanh → |z|']];

function build(timing){
  let t0=0;const out=[];
  for(const sc of SCENES){
    const tm=timing[sc.id], dur=LEAD+tm.dur+TAIL+(sc.id==='outro'?2.2:0)+(sc.id==='intro'?1.0:0);
    const s=i=>LEAD+tm.sent[Math.min(i,tm.sent.length-1)][0];
    const o={...sc,start:t0,dur,lead:LEAD,sent:tm.sent.map(([a,b])=>[a+LEAD,b+LEAD]),narr:tm.dur};
    if(sc.keys)o.k=sc.keys(s).map(([t,e])=>[t==='end'?dur-0.6:t,e]);
    if(sc.mut)o.mutT=sc.mut(s);
    if(sc.boom)o.boomT=sc.boom(s);
    out.push(o);t0+=dur;
  }
  return {scenes:out,total:t0};
}
// epoch at local scene time
function epochAt(sc,t){
  if(!sc.k)return 0;const k=sc.k;
  if(t<=k[0][0])return 0;
  for(let i=0;i<k.length-1;i++){const [ta,ea]=k[i],[tb,eb]=k[i+1];
    if(t<=tb){let u=(t-ta)/(tb-ta);if(i===k.length-2&&k.length===2)u=Math.pow(u,sc.pow||1);else if(i===0&&sc.pow>1)u=Math.pow(u,sc.pow);
      return ea+(eb-ea)*u;}}
  return k[k.length-1][1];
}
const api={LEAD,TAIL,FPS,SCENES,OUTRO_GRID,build,epochAt};
if(typeof module!=='undefined')module.exports=api;else root.TL=api;
})(this);
