# SPDX-License-Identifier: Apache-2.0
"""Full graphs between host Engram boundaries, using standard TPU bucketing."""

from functools import wraps

from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy


class DSV41CompilePolicy(DecoderTpuCompilePolicy):
    def _compile(self, fn):
        model = self.owner.model
        compile_graph = super()._compile
        model.segments = tuple(compile_graph(segment) for segment in model.segments)
        self.owner.logits_processor.forward = compile_graph(
            self.owner.logits_processor.forward
        )

        # The coordinator stays eager so CPU lookup and transfer are outside
        # Dynamo. Every device segment is still required to be a full graph.
        # install() attaches signature metadata, which bound methods reject.
        @wraps(fn)
        def coordinator(*args, **kwargs):
            return fn(*args, **kwargs)

        return coordinator

    def _direct_compile(self, fn):
        raise ValueError("V4.1 segmented execution requires the Dynamo tracer")
