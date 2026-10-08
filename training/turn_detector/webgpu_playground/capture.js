class Capture extends AudioWorkletProcessor{
 constructor(){super();this.buffers=[new Float32Array(2560),new Float32Array(2560)];this.used=0;this.active=true;this.port.onmessage=e=>{if(e.data==='stop')this.active=false;};}
 process(inputs,outputs){const input=inputs[0];for(const channel of outputs[0]||[])channel.fill(0);if(!this.active)return false;if(!input?.length)return true;for(let i=0;i<input[0].length;i++){this.buffers[0][this.used]=input[0][i];this.buffers[1][this.used]=input[1]?.[i]||0;if(++this.used===2560){this.port.postMessage(this.buffers,this.buffers.map(b=>b.buffer));this.buffers=[new Float32Array(2560),new Float32Array(2560)];this.used=0;}}return true;}
}
registerProcessor('pardon-capture',Capture);
