#!/usr/bin/env python3
"""Real multiprocess Gloo + native receiver control broadcast, no model/TPU.

Two ranks bound CPU resource use. Tests real transport ordering, not TP16 safety.
"""
import json
import socket
import sys
import types
from unittest.mock import patch


def worker(rank, port):
    import torch.distributed as dist
    from benchmark.tpu.glm52_epic.run_reader_cpu_tests import unavailable
    boundary = types.ModuleType('sgl_kernel')
    for name in ('copy_all_layer_kv_cache_cpu','assign_req_to_token_pool_cpu',
                 'assign_extend_cache_locs_cpu','fill_accept_out_cache_loc_cpu',
                 'fill_bonus_tokens_cpu','build_tree_kernel_efficient_cpu','verify_tree_greedy_cpu'):
        setattr(boundary,name,unavailable)
    with patch.dict(sys.modules,{'sgl_kernel':boundary}):
        from sglang.srt.managers.scheduler_components import request_receiver as module
        from benchmark.tpu.glm52_epic.ordered_admission import typed_control
        from sglang.srt.managers.io_struct import PauseGenerationReqInput,ContinueGenerationReqInput
        from datetime import timedelta
        dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=timedelta(seconds=90))
        try:
            parallel=types.SimpleNamespace(attn_dp_enabled=True,attn_tp_rank=0,
                    attn_cp_rank=0,enable_dp_attention_local_control_broadcast=False,
                    tp_size=2,tp_group=types.SimpleNamespace(rank=rank,cpu_group=dist.group.WORLD,ranks=[0,1]))
            receiver=module.SchedulerRequestReceiver.__new__(module.SchedulerRequestReceiver)
            received=[]
            # Runtime config group handles are boundary supplied; native branch,
            # classification and broadcast_pyobj (real Gloo) remain unchanged.
            with patch.object(module,'get_parallel',return_value=parallel), \
                 patch.object(module,'get_exec',return_value=types.SimpleNamespace(moe=types.SimpleNamespace(is_ep_scale_joiner=False))), \
                 patch.object(module,'node_fanout_enabled',return_value=False), \
                 patch.object(module,'attn_cp_tp_broadcast_pyobj',side_effect=lambda x:x):
                for seq,action in enumerate(('pause','resume')):
                    command=dict(action=action,rid=f'glm52-control:run:measure:{seq}:{action}')
                    raw=[typed_control(command)] if rank==0 else []
                    result=receiver._broadcast_reqs_across_ranks(raw)
                    assert len(result)==1 and result[0].rid==command['rid']
                    assert isinstance(result[0],PauseGenerationReqInput if seq==0 else ContinueGenerationReqInput)
                    if seq==1:assert result[0].torch_empty_cache is False
                    received.append(result[0].rid)
            print('ORDERED_NATIVE_GLOO_RANK_PASS '+json.dumps(dict(rank=rank,received=received,scope='real2process nativecontrolbroadcast; attentionTP1 identityboundary; no16rank/admission/modelproof')),flush=True)
        finally:
            dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=2,join=True)
    print('ORDERED_NATIVE_GLOO_CPU_PASS ranks=2',flush=True)


if __name__=='__main__':
    main()
