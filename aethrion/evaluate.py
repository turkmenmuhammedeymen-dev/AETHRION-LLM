"""Held-out quality and inference-throughput evaluation."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import torch
from aethrion.chat import load_model
from aethrion.train import get_batch,load_tokens
from aethrion.tokenizer import ByteBPETokenizer

def summarize_metrics(losses:list[float],elapsed_seconds:float,tokens_processed:int,
                      parameter_count:int,peak_gpu_memory_bytes:int|None=None)->dict:
    if not losses or elapsed_seconds<=0 or tokens_processed<1:
        raise ValueError("Evaluation requires losses, positive elapsed time, and processed tokens.")
    mean_loss=sum(losses)/len(losses)
    result={"heldout_loss":mean_loss,"perplexity":math.exp(min(mean_loss,20)),
            "evaluation_batches":len(losses),"tokens_processed":tokens_processed,
            "elapsed_seconds":elapsed_seconds,"tokens_per_second":tokens_processed/elapsed_seconds,
            "parameter_count":parameter_count}
    if peak_gpu_memory_bytes is not None: result["peak_gpu_memory_bytes"]=peak_gpu_memory_bytes
    return result

def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="artifacts/checkpoint_last.pt")
    parser.add_argument("--tokenizer",default="artifacts/tokenizer.json")
    parser.add_argument("--data",required=True,help="Independent held-out UTF-8 text file")
    parser.add_argument("--batches",type=int,default=100)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--json-out",default=None)
    args=parser.parse_args()
    if args.batches<1 or args.batch_size<1: parser.error("--batches and --batch-size must be positive")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=load_model(args.checkpoint,device)
    tokenizer=ByteBPETokenizer.load(args.tokenizer)
    data=load_tokens(args.data,tokenizer)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    losses=[]; tokens=0
    if device.type=="cuda": torch.cuda.synchronize(device)
    started=time.perf_counter()
    with torch.no_grad():
        for _ in range(args.batches):
            x,y=get_batch(data,args.batch_size,model.config.block_size,device)
            _,loss=model(x,y); losses.append(float(loss.item())); tokens+=int(x.numel())
    if device.type=="cuda": torch.cuda.synchronize(device)
    elapsed=max(time.perf_counter()-started,1e-9)
    peak=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
    report=summarize_metrics(losses,elapsed,tokens,model.parameter_count(),peak)
    report["device"]=str(device)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.json_out:
        target=Path(args.json_out); target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
