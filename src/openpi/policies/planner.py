from dataclasses import dataclass

from transformers import AutoProcessor, AutoModelForImageTextToText
from openpi.models.model import Observation
from openpi.training.config import pi05_config
from openpi.models.tokenizer import PaligemmaTokenizer
import numpy as np
import PIL.Image as Image
import PIL
import json
import torch 
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


print(f"Using device: {device}")
@dataclass
class PlanState:
    subtask: str
    state: bool = False
    confidence: float | None = None
    raw_response : str | None = None

@dataclass
class PlannerInput: 
    prompt: str
    image_ext : Image.Image | None = None
    image_gripper : Image.Image | None = None

@dataclass
class PlannerOutput: 
    subtask: str
    done: bool = False
    confidence: float | None = None
    raw_response = None


class HighLevelPlanner():

    def __init__(self, config: pi05_config.Pi05Config): 
        self.config = config
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=dtype, device_map="auto")
        self.model.eval()
        self.pi_tokenizer = PaligemmaTokenizer(max_len=48)

        self.current_task : str | None = None
        self.current_plan : list[str] | None = None

    @torch.inference_mode()
    def generate_subtask(self, observation: Observation) -> PlannerOutput:
        planner_input = self.process_input(observation)
        messages = [
            {
            "role": "user",
            "content": [
                {"type" : "image", "image": planner_input.image_ext},
                {"type" : "image", "image": planner_input.image_gripper},
                {"type" : "text", "text": planner_input.prompt}
                ]
            }
        ] 

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True, # Might need to be false?
        )

        inputs = self.processor(
            text=[text],
            images=[planner_input.image_ext, planner_input.image_gripper],
            padding=True, 
            return_tensors="pt"
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        timebf = time.monotonic_ns()
        generated_ids = self.model.generate(**inputs, max_new_tokens=40)
        print(f"Generation time: {(time.monotonic_ns() - timebf) / 1e9} seconds")
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        print(f"Raw model output: {output_text}")
        try: 
            response_obj = json.loads(output_text)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            response_obj = {
                "subtask": "",
                "done": False,
                "confidence": 0.0
            }
        subtask_string = response_obj["subtask"]
        subtask_done = response_obj["done"]
        confidence = response_obj["confidence"]
        print(subtask_string, subtask_done, confidence)

        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")

        return PlannerOutput(
            subtask = subtask_string,
            done = subtask_done,
            confidence = subtask_done,
            raw_response = response_obj
        )
        
         

    def check_plan_status(self, observation: Observation) -> PlanState:
        pass

    def process_input(self, observation: Observation) :
        task_prompt : str = self.pi_tokenizer.decode(observation.tokenized_prompt) 
        prompt : str = f"""
        Return a subtask to the given task that the robot should do in order to complete the task.
        The subtask should be precise and concise. 
        Use the images to generate the subtask. 
        Return exactly one JSON object and nothing else.
        Do not use markdown.
        Do not wrap the answer in triple backticks.

        Format:
        {{
        "subtask": "string",
        "done": true/false,
        "confidence": float between 0 and 1
        }}

        Task: {task_prompt}
        """
        
        base_image = (np.array(observation.images["base_0_rgb"])[0, :, :, :]).astype(np.uint8)
        # base_image = np.transpose(base_image, (2,0,1))
        gripper_image = (np.array(observation.images["left_wrist_0_rgb"])[0,:,:,:]).astype(np.uint8)
        # gripper_image = np.transpose(gripper_image, (2,0,1))
        print(f"Image shape: {base_image.shape}, {gripper_image.shape}")
        
        base_image = Image.fromarray(base_image)
        gripper_image = Image.fromarray(gripper_image)

        return PlannerInput(
            prompt=prompt, 
            image_ext=base_image, 
            image_gripper=gripper_image
            )
        
 

    def create(self) -> "HighLevelPlanner":
        return HighLevelPlanner()
    
    def warmup(self) : 
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    {"type": "text", "text": "What animal is on the candy?"}
                ]
            },
        ]




     