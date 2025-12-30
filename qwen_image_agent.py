"""\nQwen Image Editing AI Agent\nAutomated agent for character consistency editing, outfit changes, pose modification,\nand background editing using ComfyUI Qwen workflow.\n"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EditType(Enum):
    """Types of image edits supported by the agent"""
    CHARACTER_CONSISTENCY = "character_consistency"
    OUTFIT_CHANGE = "outfit_change"
    POSE_CHANGE = "pose_change"
    BACKGROUND_CHANGE = "background_change"
    MULTI_CHARACTER = "multi_character"
    FULL_EDIT = "full_edit"


@dataclass
class ImageEditRequest:
    """Request structure for image editing tasks"""
    image_url: str
    edit_type: EditType
    prompt: str
    num_people: int = 1
    nsfw_allowed: bool = False
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    steps: int = 20
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class EditResult:
    """Result structure for image editing operations"""
    request_id: str
    status: str
    edited_image_url: str
    edit_type: EditType
    processing_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class QwenImageAgent:
    """AI Agent for managing Qwen image editing tasks"""

    def __init__(
        self,
        comfyui_endpoint: str = "http://localhost:8188",
        api_timeout: int = 300,
        max_retries: int = 3
    ):
        self.comfyui_endpoint = comfyui_endpoint
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=api_timeout)
        self.workflow_cache = {}

    async def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Load ComfyUI workflow from JSON file"""
        try:
            if workflow_path in self.workflow_cache:
                return self.workflow_cache[workflow_path]

            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
            self.workflow_cache[workflow_path] = workflow
            logger.info(f"Loaded workflow from {workflow_path}")
            return workflow
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            raise

    async def build_workflow_input(
        self,
        request: ImageEditRequest,
        base_workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build workflow input based on edit request"""
        workflow_input = base_workflow.copy()

        # Configure workflow nodes based on edit type
        if request.edit_type == EditType.CHARACTER_CONSISTENCY:
            workflow_input = self._configure_consistency_editing(workflow_input, request)
        elif request.edit_type == EditType.OUTFIT_CHANGE:
            workflow_input = self._configure_outfit_change(workflow_input, request)
        elif request.edit_type == EditType.POSE_CHANGE:
            workflow_input = self._configure_pose_change(workflow_input, request)
        elif request.edit_type == EditType.BACKGROUND_CHANGE:
            workflow_input = self._configure_background_change(workflow_input, request)
        elif request.edit_type == EditType.MULTI_CHARACTER:
            workflow_input = self._configure_multi_character(workflow_input, request)
        elif request.edit_type == EditType.FULL_EDIT:
            workflow_input = self._configure_full_edit(workflow_input, request)

        # Set common parameters
        self._set_common_params(workflow_input, request)

        return workflow_input

    def _configure_consistency_editing(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for character consistency editing"""
        # Update text encoding nodes with prompt
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = request.prompt
                if request.negative_prompt:
                    node['inputs']['negative'] = request.negative_prompt
        return workflow

    def _configure_outfit_change(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for outfit changing"""
        # Set outfit-specific parameters
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = f"Change outfit: {request.prompt}"
        return workflow

    def _configure_pose_change(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for pose changes"""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = f"Change pose: {request.prompt}"
        return workflow

    def _configure_background_change(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for background changes"""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = f"Change background: {request.prompt}"
        return workflow

    def _configure_multi_character(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for multi-character editing"""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = f"Edit {request.num_people} characters: {request.prompt}"
                node['inputs']['num_people'] = request.num_people
        return workflow

    def _configure_full_edit(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> Dict[str, Any]:
        """Configure workflow for full image editing"""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'TextEncodeQwenImageEditPlus':
                node['inputs']['text'] = request.prompt
                if request.negative_prompt:
                    node['inputs']['negative'] = request.negative_prompt
        return workflow

    def _set_common_params(
        self,
        workflow: Dict[str, Any],
        request: ImageEditRequest
    ) -> None:
        """Set common parameters across workflow nodes"""
        for node_id, node in workflow.items():
            if node.get('class_type') == 'KSampler':
                node['inputs']['seed'] = request.seed or 0
                node['inputs']['steps'] = request.steps
                node['inputs']['cfg'] = request.guidance_scale

    async def submit_task(
        self,
        workflow_input: Dict[str, Any]
    ) -> str:
        """Submit editing task to ComfyUI"""
        try:
            response = await self.client.post(
                f"{self.comfyui_endpoint}/api/prompt",
                json={"prompt": workflow_input}
            )
            result = response.json()
            prompt_id = result.get('prompt_id')
            logger.info(f"Task submitted with ID: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise

    async def poll_task_status(
        self,
        prompt_id: str,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """Poll task status until completion"""
        max_polls = self.api_timeout // poll_interval
        for _ in range(max_polls):
            try:
                response = await self.client.get(
                    f"{self.comfyui_endpoint}/api/history/{prompt_id}"
                )
                history = response.json()
                if prompt_id in history:
                    return history[prompt_id]
            except Exception as e:
                logger.warning(f"Error checking status: {e}")

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task {prompt_id} did not complete within timeout")

    async def process_edit_request(self, request: ImageEditRequest) -> EditResult:
        """Process a complete image editing request"""
        start_time = datetime.now()
        request_id = f"{datetime.now().timestamp()}"

        try:
            # Load workflow
            workflow = await self.load_workflow("qwen_workflow.json")

            # Build workflow input
            workflow_input = await self.build_workflow_input(request, workflow)

            # Submit task
            prompt_id = await self.submit_task(workflow_input)

            # Poll for completion
            result = await self.poll_task_status(prompt_id)

            processing_time = (datetime.now() - start_time).total_seconds()

            return EditResult(
                request_id=request_id,
                status="success",
                edited_image_url=f"{self.comfyui_endpoint}/output/{prompt_id}",
                edit_type=request.edit_type,
                processing_time=processing_time,
                metadata={
                    "prompt_id": prompt_id,
                    "num_people": request.num_people,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale
                }
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing edit request: {e}")
            return EditResult(
                request_id=request_id,
                status="error",
                edited_image_url="",
                edit_type=request.edit_type,
                processing_time=processing_time,
                metadata={},
                error_message=str(e)
            )

    async def batch_process(
        self,
        requests: List[ImageEditRequest]
    ) -> List[EditResult]:
        """Process multiple editing requests concurrently"""
        tasks = [self.process_edit_request(req) for req in requests]
        results = await asyncio.gather(*tasks)
        return results

    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    """Example usage of the QwenImageAgent"""
    agent = QwenImageAgent(comfyui_endpoint="http://localhost:8188")

    # Example request
    edit_request = ImageEditRequest(
        image_url="./input_image.png",
        edit_type=EditType.OUTFIT_CHANGE,
        prompt="Change the outfit to a blue dress with white shoes",
        num_people=1,
        steps=20,
        guidance_scale=7.5
    )

    try:
        result = await agent.process_edit_request(edit_request)
        print(f"Edit result: {result}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
