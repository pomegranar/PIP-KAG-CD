from transformers import Trainer
import torch
from transformers import TrainerCallback


class SNIPCallback(TrainerCallback):
    def __init__(self, masks):
        self.masks = masks

    def apply_masks(self, model):
        for name, param in model.named_parameters():
            cur_name = '.'.join(name.split('.')[2:])
            if cur_name in self.masks:
                param.data *= self.masks[cur_name].to(param.device)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        print("Calculating SNIP masks...")
        self.apply_masks(model)
        print("SNIP masks applied!")

        cnt = 0
        for name, param in model.named_parameters():
            
            
            cur_name = '.'.join(name.split('.')[2:])
            if cur_name in self.masks:
                if cnt == 300:
                    break
                cnt += 1
                print(f'{name} mask values:\n{self.masks[cur_name]}')
                print(f"{name} masked values:\n{param.data}")
        
        mask_sum = 0
        for mask in self.masks.values():
            mask = mask.float()  #  float32
            mask_sum += mask.sum().item()
        mask_num = mask_sum / 1000000000

    def on_step_end(self, args, state, control, model=None, **kwargs):

        print("SNIP masks applied!")
        if model is None:
            raise ValueError("Model is not provided to SNIPCallback.")
        self.apply_masks(model)  


class InputContrastiveTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        total_steps = self.state.max_steps
        current_step = self.state.global_step
        inputs['cur_step'] = current_step
        inputs['total_step'] = total_steps
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if hasattr(outputs, 'metrics'):
            combined_metrics = {}
            for metric_name, metric_value in outputs.metrics.items():
                if metric_name == 'margin':
                    combined_metrics[metric_name] = metric_value
                else:
                    combined_metrics[metric_name] = metric_value.mean().item()
            self.log(combined_metrics)
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    

class SparsePrunedTrainer(Trainer):
    def __init__(self, *args, masks=None, **kwargs):
        """
        Custom Trainer that integrates pruning masks.
        """
        super().__init__(*args, **kwargs)
        self.masks = masks

    def training_step(self, model, inputs):
        """
        Custom training step to reapply pruning masks after each optimizer step.
        """
        loss = super().training_step(model, inputs)

        # Apply pruning masks to ensure sparsity
        if self.masks:
            for name, param in model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name]
        return loss