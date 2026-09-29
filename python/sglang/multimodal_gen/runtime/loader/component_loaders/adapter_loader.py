<<<<<<< HEAD
=======
from sglang.multimodal_gen.configs.models.adapter.anima import (
    AnimaTextConditionerConfig,
)
>>>>>>> upstream/main
from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class AdapterLoader(PlainStateDictComponentLoader):
<<<<<<< HEAD
    component_names = ["connectors", "duration_head"]

    config_classes = {
=======
    component_names = ["connectors", "duration_head", "text_conditioner"]

    config_classes = {
        "text_conditioner": AnimaTextConditionerConfig,
>>>>>>> upstream/main
        "connectors": LTX2ConnectorConfig,
        "duration_head": LTX2DurationHeadConfig,
    }
