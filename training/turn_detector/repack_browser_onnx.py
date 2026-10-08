"""Lossless ONNX external-data repacking; model execution stays remote."""
from pathlib import Path
import hashlib,os,json

def graph_tensors(graph):
    """Include Constant attributes and nested graphs, not just initializers."""
    import onnx
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.TENSOR:yield attr.t
            elif attr.type==onnx.AttributeProto.TENSORS:yield from attr.tensors
            elif attr.type==onnx.AttributeProto.GRAPH:yield from graph_tensors(attr.g)
            elif attr.type==onnx.AttributeProto.GRAPHS:
                for nested in attr.graphs:yield from graph_tensors(nested)


def externalize_shared(paths,directory):
    """Deduplicate raw initializers across phase graphs in one external data file.

    Keeping two graphs must not silently double the delivered parameter bytes.
    Small typed constants remain inline. This does not imply shared GPU weights
    across two browser sessions; WebGPU runtime memory still needs measurement.
    """
    import onnx
    from onnx.external_data_helper import set_external_data
    directory=Path(directory);directory.mkdir(exist_ok=False)
    offsets={};outputs=[]
    with (directory/'weights.bin').open('wb') as stream:
        for path in paths:
            model=onnx.load(path,load_external_data=True)
            for tensor in graph_tensors(model.graph):
                if not tensor.HasField('raw_data') or len(tensor.raw_data)<1024:continue
                raw=tensor.raw_data;key=(tensor.data_type,tuple(tensor.dims),hashlib.sha256(raw).hexdigest())
                if key not in offsets:
                    offset=stream.tell();stream.write(raw);offsets[key]=(offset,len(raw))
                offset,length=offsets[key]
                set_external_data(tensor,location='weights.bin',offset=offset,length=length)
                tensor.ClearField('raw_data')
            target=directory/path.name
            target.write_bytes(model.SerializeToString());outputs.append(target)
        stream.flush();os.fsync(stream.fileno())
    for path in outputs:onnx.checker.check_model(str(path))
    return outputs



def main():
    import argparse,onnx
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not Path('/content').is_dir():raise RuntimeError('Remote only')
    paths=[a.input/(x+'.onnx') for x in ['startup','steady']]
    new=externalize_shared(paths,a.out)
    for before,after in zip(paths,new):
        b=onnx.load(before,load_external_data=True);c=onnx.load(after,load_external_data=True)
        for graph in (b,c):
            for tensor in graph_tensors(graph.graph):
                if tensor.external_data:raise ValueError('Unresolved external data')
                tensor.ClearField('data_location') # Default-vs-unset protobuf metadata only.
        if b.SerializeToString(deterministic=True)!=c.SerializeToString(deterministic=True):raise ValueError('Loaded graphs differ after repack')
    files={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()} for f in a.out.iterdir()}
    result={'loaded_graphs_byte_equal':True,'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'input_total_bytes':sum(f.stat().st_size for f in a.input.iterdir()),'limitations':'Lossless storage repack only; no new numerical or browser qualification.'}
    (a.out/'repack.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
