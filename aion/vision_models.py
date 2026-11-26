"""Optional local pretrained adapters. Weights are never downloaded implicitly."""
from pathlib import Path
from PIL import Image
import torch
from .vision import region


class RTDETR:
    """Transformers-compatible RT-DETR checkpoint (not raw upstream .pth)."""
    def __init__(self, model_dir, threshold=.5):
        from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
        path = str(Path(model_dir).resolve(strict=True))
        self.processor = RTDetrImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = RTDetrForObjectDetection.from_pretrained(path, local_files_only=True, use_safetensors=True).eval()
        self.threshold = threshold

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=Image.fromarray(image), return_tensors='pt')
        result = self.processor.post_process_object_detection(self.model(**inputs), threshold=self.threshold, target_sizes=[image.shape[:2]])[0]
        return [r for score,label,box in zip(result['scores'],result['labels'],result['boxes'])
                if (r := region('object', self.model.config.id2label[int(label)], float(score), box.tolist(), 'rt_detr', image)) is not None]


class GroundingDINO:
    def __init__(self, model_dir, labels):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        path = str(Path(model_dir).resolve(strict=True))
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
        self.labels = labels

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=Image.fromarray(image), text=self.labels, return_tensors='pt')
        result = self.processor.post_process_grounded_object_detection(self.model(**inputs), inputs.input_ids, threshold=.4, text_threshold=.3, target_sizes=[image.shape[:2]])[0]
        return [r for score,label,box in zip(result['scores'],result['text_labels'],result['boxes'])
                if (r := region('object', label, float(score), box.tolist(), 'grounding_dino', image)) is not None]


class VisualLanguage:
    """Read-only Qwen3-VL or compatible UI-TARS checkpoint; no action execution."""
    def __init__(self, model_dir):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        path = str(Path(model_dir).resolve(strict=True))
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()

    @torch.inference_mode()
    def describe(self, image):
        messages = [{'role':'user','content':[{'type':'image','image':Image.fromarray(image)}, {'type':'text','text':'Describe visible objects and text. Distinguish uncertainty. Do not propose actions.'}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
