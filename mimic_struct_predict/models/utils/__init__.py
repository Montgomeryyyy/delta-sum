"""Shared utilities for model training and inference."""

# from mimic_struct_predict.models.utils.dataset import (
#     discover_shards,
#     load_prompt_dataset,
# )
# from mimic_struct_predict.models.utils.io import (
#     write_latest_sentinel,
# )
from mimic_struct_predict.models.utils.model_loading import (
    get_string_id,
    load_model_and_tokenizer,
)
from mimic_struct_predict.models.utils.prompt_template import (
    create_prompt,
)

# from mimic_struct_predict.models.utils.shard_processor import (
#     process_shards_from_prompt_dataset,
# )

__all__ = [
    "get_string_id",
    "load_model_and_tokenizer",
    "create_prompt",
]
