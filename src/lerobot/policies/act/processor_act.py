#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register(name="act_text_processor")
@dataclass
class ACTTextProcessorStep(ProcessorStep):
    """
    Processor step to extract text/task description from complementary data and add it to observation.
    
    This is needed for text-conditioned ACT where the text encoder expects a "text" key in the batch.
    """
    
    task_key: str = "task"
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Extract text from complementary data and add it to the observation.
        
        Args:
            transition: The environment transition containing complementary data with task description.
            
        Returns:
            Modified transition with text added to observation (as raw strings, not tokenized).
        """
        # Get complementary data (contains task description)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            # No text data available, skip processing
            return transition
        
        # Extract task description
        task = complementary_data.get(self.task_key)
        if task is None:
            # No task in complementary data, skip
            return transition
        
        # Create a copy to avoid modifying the original
        transition = transition.copy()
        
        # Ensure observation dict exists
        if TransitionKey.OBSERVATION not in transition:
            transition[TransitionKey.OBSERVATION] = {}
        
        # Add text to observation (keep as string, not tokenize)
        # The text encoder in the model will handle the actual tokenization
        observation = transition[TransitionKey.OBSERVATION]
        
        # Standardize to list of strings for batch processing
        if isinstance(task, str):
            observation["text"] = task
        elif isinstance(task, list):
            # If it's already a list, just use it
            observation["text"] = task[0] if len(task) == 1 else task
        
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step doesn't add tensor features (text is handled separately),
        so we don't modify the features dict.
        """
        return features


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
    ]
    
    # Add text processor if text conditioning is enabled
    if getattr(config, 'use_text_conditioning', False):
        input_steps.append(ACTTextProcessorStep(task_key="task"))
    
    # Add standard processing steps
    input_steps.extend([
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ])
    
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
