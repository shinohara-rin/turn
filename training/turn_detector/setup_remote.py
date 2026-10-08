from pathlib import Path
import os, subprocess, sys
os.environ['HF_TOKEN']=Path('/content/.hf_token').read_text().strip()
os.environ['HF_HOME']='/content/hf'
p=Path('/content/turn-recreation'); p.mkdir(exist_ok=True)
def run(args):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(result.stdout[-10000:],flush=True)
    result.check_returncode()
if not Path('/content/turnbench/.git').exists():run(['git','clone','https://github.com/SesameAILabs/turnbench.git','/content/turnbench'])
run(['git','-C','/content/turnbench','checkout','38a6f874322430cb3ca71d8a52aa1e636e88bad8'])
run([sys.executable,'-m','pip','install','-q','nemo_toolkit[asr]','silero-vad','datasets','pyarrow','pydantic>=2','soundfile','scikit-learn','srt'])
print('SETUP_COMPLETE',flush=True)
