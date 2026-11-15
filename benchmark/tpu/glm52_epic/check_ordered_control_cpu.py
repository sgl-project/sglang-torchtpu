#!/usr/bin/env python3
"""CPU integration: real IPC, classifier and pause handlers, no collectives."""
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from benchmark.tpu.glm52_epic.run_reader_cpu_tests import unavailable


def main():
    # Same narrow unused compiled extension boundary as accepted reader tests.
    boundary = types.ModuleType('sgl_kernel')
    for name in ('copy_all_layer_kv_cache_cpu','assign_req_to_token_pool_cpu',
                 'assign_extend_cache_locs_cpu','fill_accept_out_cache_loc_cpu',
                 'fill_bonus_tokens_cpu','build_tree_kernel_efficient_cpu','verify_tree_greedy_cpu'):
        setattr(boundary, name, unavailable)
    with patch.dict(sys.modules, {'sgl_kernel':boundary}):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
        from sglang.srt.managers.io_struct import PauseGenerationReqInput, ContinueGenerationReqInput
        from benchmark.tpu.glm52_epic.ordered_admission import typed_control
        from benchmark.tpu.glm52_epic.ordered_scheduler_ack import install

        pause = typed_control(dict(action='pause',rid='glm52-control:run:measure:0:pause'))
        resume = typed_control(dict(action='resume',rid='glm52-control:run:measure:1:resume'))
        assert type(pause) is PauseGenerationReqInput and type(resume) is ContinueGenerationReqInput
        work, controls = SchedulerRequestReceiver._split_work_and_control_reqs(None,[pause,resume])
        assert work == [] and controls == [pause,resume]
        config=dict(dp_size=1,tp_size=16,attn_dp_size=16,pp_size=1,attn_tp_size=1,
                    attn_cp_size=1,attn_dcp_size=1,local_control=False,elastic_join=False,
                    speculative=False,disaggregation=False,tp_rank=0,attn_dp_rank=0)

        class IngestBoundary:
            # Dispatch REAL handlers; incoming control already reflects broadcast.
            _engine_paused=False
            running_batch=chunked_req=last_batch=None
            result_queue=()
            _deferred_input_requests=()
            disaggregation_mode=types.SimpleNamespace(value='null')
            def __init__(self):
                from sglang.srt.disaggregation.utils import DisaggregationMode
                self.disaggregation_mode=DisaggregationMode.NULL
                self.waiting_queue=[]
                self.incoming=[]
            def ingest_requests(self):
                items,self.incoming=self.incoming,[]
                for item in items:
                    fn=Scheduler.pause_generation if isinstance(item,PauseGenerationReqInput) else Scheduler.continue_generation
                    assert fn(self,item) is None  # No syntheticRID frontend reply.
                return items

        with tempfile.TemporaryDirectory() as directory:
            undo=install(IngestBoundary,directory=directory,run_token='run',topology=lambda _:config)
            owner=IngestBoundary();owner.incoming=[pause]
            assert owner.ingest_requests()==[pause] and owner._engine_paused
            owner.waiting_queue=[types.SimpleNamespace(rid='request-a')]
            assert owner.ingest_requests()==[]
            record=json.loads(Path(directory,'ordered-rank-0.json').read_text())
            assert record['paused'] and record['waiting_rids']==['request-a'] and record['sequence']==0
            owner.incoming=[resume]
            assert owner.ingest_requests()==[resume] and not owner._engine_paused
            record=json.loads(Path(directory,'ordered-rank-0.json').read_text())
            assert record['action']=='resume' and record['sequence']==1
            undo()
        print('ORDERED_REAL_CONTROL_CPU_PASS '+json.dumps(dict(real='IPC types/classifier/pause/continue handlers',
              scope='single scheduler boundary; no receive transport/16rankcollective/hardware/admission/cadence proof')),flush=True)


if __name__ == '__main__':
    main()
