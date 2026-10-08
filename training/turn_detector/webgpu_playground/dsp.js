/* Causal browser frontend: pinned NeMo window/mel coefficients, no global normalization. */
(function(root){
const f=Math.fround;
function half(x){const a=new Float32Array([x]),u=new Uint32Array(a.buffer)[0],sign=u>>>31? -1:1,exp=(u>>>23)&255,m=u&0x7fffff;if(exp===255)return x;if(exp>142)return sign*Infinity;if(exp<102)return sign*0;const shift=exp<113?126-exp:13;let sig=exp<113?(m|0x800000):m;const base=Math.floor(sig/2**shift),rem=sig-base*2**shift,rounded=base+(rem>2**(shift-1)||(rem===2**(shift-1)&&(base&1))?1:0);return sign*(exp<113?rounded*2**-24:(1024+rounded)*2**(exp-127-10));}
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}for(let len=2;len<=n;len*=2){const ang=-2*Math.PI/len,wr=Math.cos(ang),wi=Math.sin(ang);for(let i=0;i<n;i+=len){let ar=1,ai=0;for(let j=0;j<len/2;j++){const k=i+j,l=k+len/2,tr=re[l]*ar-im[l]*ai,ti=re[l]*ai+im[l]*ar;re[l]=re[k]-tr;im[l]=im[k]-ti;re[k]+=tr;im[k]+=ti;[ar,ai]=[ar*wr-ai*wi,ar*wi+ai*wr];}}}}
class Frontend{
 constructor(cfg){this.cfg=cfg;this.reset();}
 reset(){this.samples=[[],[]];this.base=0;this.total=0;this.nextFrame=0;this.frames=[];this.frameBase=0;this.chunkIndex=0;this.first=true;this.previous=[0,0];}
 append(channels){for(let c=0;c<2;c++){for(let i=0;i<channels[c].length;i++){const x=channels[c][i];this.samples[c].push(this.total+i===0?x:f(x-f(f(.97)*this.previous[c])));this.previous[c]=x;}}this.total+=channels[0].length;
  const cfg=this.cfg,n=cfg.fftSize,offset=(n-cfg.window.length)/2;
  while(this.nextFrame*cfg.hop+n/2<this.total){const mel=[];for(let c=0;c<2;c++){const re=new Float64Array(n),im=new Float64Array(n);for(let i=0;i<cfg.window.length;i++){let index=this.nextFrame*cfg.hop-n/2+offset+i;if(index<0)index=-index;re[offset+i]=f(this.samples[c][index-this.base]*cfg.window[i]);}fft(re,im);const power=new Float32Array(n/2+1);for(let k=0;k<power.length;k++)power[k]=f(re[k]*re[k]+im[k]*im[k]);const row=new Float32Array(128);for(let m=0;m<128;m++){let sum=0;for(let k=0;k<power.length;k++)sum+=cfg.mel[m][k]*power[k];row[m]=f(Math.log(f(f(sum)+cfg.logGuard)));}mel.push(row);}this.frames.push(mel);this.nextFrame++;}
  const keep=Math.max(0,this.nextFrame*cfg.hop-n/2-1);if(keep>this.base){for(let c=0;c<2;c++)this.samples[c].splice(0,keep-this.base);this.base=keep;}
 }
 next(){const size=this.first?9:16,end=this.chunkIndex+size;if(end>this.nextFrame)return null;const start=this.first?0:this.chunkIndex-9,t=end-start,x=new Float32Array(2*128*t);for(let c=0;c<2;c++)for(let m=0;m<128;m++)for(let j=0;j<t;j++)x[(c*128+m)*t+j]=this.frames[start+j-this.frameBase][c][m];const result={x,t,phase:this.first?'startup':'steady',availableSamples:(end-1)*160+257};this.chunkIndex=end;this.first=false;const keep=this.chunkIndex-9;if(keep>this.frameBase){this.frames.splice(0,keep-this.frameBase);this.frameBase=keep;}return result;}
}
class Features{
 constructor(){this.reset();}
 reset(){this.rows=[];this.cumulative=[0,0];this.starts=[.16,.16];this.previous=[false,false];this.armed=[false,false];this.lastFire=[-Infinity,-Infinity];}
 step(encoded,vad,pcm,time,threshold=.8214424509124978){const before=this.cumulative.slice();for(let c=0;c<2;c++)this.cumulative[c]=f(this.cumulative[c]+vad[c]);this.rows.push({time,before,after:this.cumulative.slice()});if(this.rows.length>40)this.rows.shift();const own=[];for(let c=0;c<2;c++){const speech=vad[c]>=.5;if(this.rows.length>1&&speech!==this.previous[c])this.starts[c]=time;this.previous[c]=speech;const age=Math.min(time-this.starts[c],10)/10;let sum=0;for(const x of pcm[c])sum+=x*x;const energy=f(Math.log(Math.sqrt(sum/pcm[c].length)+1e-7));const h=[vad[c],speech?0:age,speech?age:0];for(const seconds of [.32,.64,1.28,2.56,5.12]){const entries=this.rows.filter(x=>x.time>time-seconds);h.push((this.cumulative[c]-entries[0].before[c])/entries.length);}own.push([...encoded[c].map(half),...h,energy]);}return new Float32Array([...own[0],...own[1],...own[1],...own[0]]);}
 events(vad,probabilities,time,threshold,recommit){const events=[];for(let c=0;c<2;c++){if(vad[c]>=.5){this.armed[c]=true;this.lastFire[c]=-Infinity;}else if(this.armed[c]&&probabilities[c]>=threshold&&time-this.lastFire[c]>=recommit){this.lastFire[c]=time;events.push({speaker:c+1,time,score:probabilities[c]});}}return events;}
}
root.PardonDSP={Frontend,Features,half,fft};if(typeof module!=='undefined')module.exports=root.PardonDSP;
})(globalThis);
