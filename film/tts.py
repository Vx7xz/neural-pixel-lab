import json,sys,numpy as np,soundfile as sf
from kokoro_onnx import Kokoro
T=sys.argv[1]; L=sys.argv[2]; voice=sys.argv[3] if len(sys.argv)>3 else 'bm_george'
k=Kokoro(T+'/kokoro.onnx',T+'/voices.bin')
scenes=json.load(open(L+'/script.json'))
only=sys.argv[4].split(',') if len(sys.argv)>4 else None
import os
timing=json.load(open(L+"/audio/timing.json")) if os.path.exists(L+"/audio/timing.json") else {}
for sc in scenes:
    if only and sc['id'] not in only: continue
    parts=[];t=0.0;sent=[]
    for i,line in enumerate(sc['en']):
        a,sr=k.create(line.replace('Tanh','Tan-H').replace('eta','eeta'),voice=voice,speed=0.88,lang='en-gb' if voice.startswith('b') else 'en-us')
        gap=0.55 if i<len(sc['en'])-1 else 0.0
        sent.append([round(t,3),round(t+len(a)/sr,3)]);t+=len(a)/sr+gap
        parts.append(a);parts.append(np.zeros(int(gap*sr),dtype=a.dtype))
    audio=np.concatenate(parts);sf.write(f"{L}/audio/{sc['id']}.wav",audio,sr)
    timing[sc['id']]={'dur':round(len(audio)/sr,3),'sent':sent}
    print(sc['id'],round(len(audio)/sr,2))
json.dump(timing,open(f"{L}/audio/timing.json",'w'))
