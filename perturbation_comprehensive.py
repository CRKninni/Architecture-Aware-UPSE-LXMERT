"""
LXMERT comprehensive perturbation — 4 AUCs in one pass.

Methods:
  rm_with_lrp  — Chefer baseline
  ours         — native UPSE (PCA + spectral seed expansion + grad/rollout)
  upse_e_final   — legacy METER-style hybrid
  upse_lxmert    — LXMERT architecture-aware (rank fuse, no LOST/PCA weights)
  upse_e_sota  — rank-fusion complement stack + light graph energy (image)
"""

import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import auc
from tqdm import tqdm
from transformers import LxmertTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _patch_transformers_compat():
    import transformers.file_utils as file_utils
    _orig = file_utils.add_code_sample_docstrings

    def _wrapper(*args, **kwargs):
        kwargs.pop("tokenizer_class", None)
        kwargs.pop("processor_class", None)
        return _orig(*args, **kwargs)

    file_utils.add_code_sample_docstrings = _wrapper


_patch_transformers_compat()

from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from lxmert.lxmert.src.ExplanationGenerator import GeneratorBaselines
from spectral.ExplanationGeneratorOurs import GeneratorOurs
from lxmert.lxmert.lxmert_upse_e import UPSE_E_FinalGenerator
from lxmert.lxmert.lxmert_upse_sota import UPSE_E_SOTAGenerator
from lxmert.lxmert.lxmert_adaptive import LXMERTAdaptiveGenerator
from lxmert.lxmert.decompx_lxmert import DecompXGenerator

VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


def _normalize_cam(x):
    x = x.detach().float().cpu() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
    mn, mx = x.min(), x.max()
    if (mx - mn) > 1e-12:
        x = (x - mn) / (mx - mn)
    return x


class ModelPertComprehensive:
    def __init__(self, coco_val_path, device="cuda"):
        self.COCO_VAL_PATH = coco_val_path if coco_val_path.endswith("/") else coco_val_path + "/"
        self.device = device
        self.vqa_answers = utils.get_data(VQA_URL)
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = device
        self.frcnn = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg, use_cdn=False
        )
        self.image_preprocess = Preprocess(self.frcnn_cfg)
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained(
            "unc-nlp/lxmert-vqa-uncased"
        ).to(device)
        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa
        self.pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    def forward(self, item):
        image_file_path = self.COCO_VAL_PATH + item["img_id"] + ".jpg"
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images, sizes, scales_yx=scales_yx,
            padding="max_detections", max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        inputs = self.lxmert_tokenizer(
            item["sent"], truncation=True, return_token_type_ids=True,
            return_attention_mask=True, add_special_tokens=True, return_tensors="pt",
        )
        self.text_len = len(self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten()))
        features = output_dict.get("roi_features")
        self.image_boxes_len = features.shape[1]
        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            visual_feats=features.to(self.device),
            visual_pos=output_dict.get("normalized_boxes").to(self.device),
            token_type_ids=inputs.token_type_ids.to(self.device),
            return_dict=True, output_attentions=False,
        )
        return self.output

    def _predict_acc(self, item, features, normalized_boxes, input_ids, attention_mask, token_type_ids):
        output = self.lxmert_vqa(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            visual_feats=features.to(self.device),
            visual_pos=normalized_boxes.to(self.device),
            token_type_ids=token_type_ids.to(self.device),
            return_dict=True, output_attentions=False,
        )
        answer = self.vqa_answers[int(output.question_answering_score.argmax().item())]
        return item["label"].get(answer, 0)

    def perturbation_comprehensive(self, item, cam_image, cam_text):
        image_file_path = self.COCO_VAL_PATH + item["img_id"] + ".jpg"
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images, sizes, scales_yx=scales_yx,
            padding="max_detections", max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        inputs = self.lxmert_tokenizer(
            item["sent"], truncation=True, return_token_type_ids=True,
            return_attention_mask=True, add_special_tokens=True, return_tensors="pt",
        )
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        cam_image = cam_image.detach().float().cpu()
        cam_text = cam_text.detach().float().cpu()
        text_len = cam_text.shape[0] - 2
        image_len = cam_image.shape[0]

        text_pos = [0] * len(self.pert_steps)
        text_neg = [0] * len(self.pert_steps)
        image_pos = [0] * len(self.pert_steps)
        image_neg = [0] * len(self.pert_steps)

        for step_idx, step in enumerate(self.pert_steps):
            curr_num_tokens = int((1 - step) * text_len)
            pure_text = cam_text[1:-1]

            _, idx = pure_text.topk(k=curr_num_tokens, dim=-1)
            idx_neg = sorted([0, cam_text.shape[0] - 1] + [int(i) + 1 for i in idx.cpu().numpy()])
            text_neg[step_idx] = self._predict_acc(
                item, features, normalized_boxes,
                inputs.input_ids[:, idx_neg], inputs.attention_mask[:, idx_neg],
                inputs.token_type_ids[:, idx_neg],
            )

            _, idx = (pure_text * (-1)).topk(k=curr_num_tokens, dim=-1)
            idx_pos = sorted([0, cam_text.shape[0] - 1] + [int(i) + 1 for i in idx.cpu().numpy()])
            text_pos[step_idx] = self._predict_acc(
                item, features, normalized_boxes,
                inputs.input_ids[:, idx_pos], inputs.attention_mask[:, idx_pos],
                inputs.token_type_ids[:, idx_pos],
            )

            curr_num_boxes = int((1 - step) * image_len)
            _, idx = cam_image.topk(k=curr_num_boxes, dim=-1)
            image_neg[step_idx] = self._predict_acc(
                item, features[:, idx, :], normalized_boxes[:, idx, :],
                inputs.input_ids, inputs.attention_mask, inputs.token_type_ids,
            )
            _, idx = (cam_image * (-1)).topk(k=curr_num_boxes, dim=-1)
            image_pos[step_idx] = self._predict_acc(
                item, features[:, idx, :], normalized_boxes[:, idx, :],
                inputs.input_ids, inputs.attention_mask, inputs.token_type_ids,
            )

        return {
            "text_positive": text_pos, "text_negative": text_neg,
            "image_positive": image_pos, "image_negative": image_neg,
        }


def generate_maps(method, baselines, upse_gen, upse_final, upse_sota, upse_lxmert, item,
                  decompx=None):
    if method == "decompx":
        cam_image, cam_text = decompx.generate(item)
    elif method == "rm_with_lrp":
        R_t_t, R_t_i = baselines.generate_relevance_maps(item, use_lrp=True)
        cam_image, cam_text = R_t_i[0], R_t_t[0]
    elif method == "ours":
        R_t_t, R_t_i = upse_gen.generate_ours(item)
        cam_image = torch.tensor(R_t_i[0], dtype=torch.float32)
        cam_text = torch.tensor(R_t_t[0], dtype=torch.float32)
    elif method in ("upse_e_final", "amf_mma"):
        cam_image, cam_text = upse_final.generate(item)
    elif method == "upse_e_sota":
        cam_image, cam_text = upse_sota.generate(item)
    elif method == "upse_lxmert":
        cam_image, cam_text = upse_lxmert.generate(item)
    else:
        raise ValueError(method)
    return _normalize_cam(cam_image), _normalize_cam(cam_text)


def run_eval(args):
    os.chdir(REPO_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    mp = ModelPertComprehensive(args.COCO_path, device=device)
    baselines = GeneratorBaselines(mp)
    upse_gen = GeneratorOurs(mp)
    upse_final = UPSE_E_FinalGenerator(mp)
    upse_sota = UPSE_E_SOTAGenerator(mp)
    upse_lxmert = LXMERTAdaptiveGenerator(mp)
    decompx = DecompXGenerator(mp)

    with open(os.path.join(REPO_ROOT, "data/vqa/valid.json")) as f:
        items = json.load(f)
    random.seed(args.seed)
    r = list(range(len(items)))
    random.shuffle(r)
    indices = r[: args.num_samples]

    accum = {k: np.zeros(len(mp.pert_steps)) for k in
             ("text_positive", "text_negative", "image_positive", "image_negative")}
    processed = 0
    x_values = mp.pert_steps
    iterator = tqdm([items[i] for i in indices], desc=args.method)

    for item in iterator:
        if not os.path.isfile(mp.COCO_VAL_PATH + item["img_id"] + ".jpg"):
            continue
        try:
            cam_i, cam_t = generate_maps(
                args.method, baselines, upse_gen, upse_final, upse_sota, upse_lxmert, item,
                decompx=decompx,
            )
            curr = mp.perturbation_comprehensive(item, cam_i, cam_t)
            processed += 1
            for k in accum:
                accum[k] += np.array(curr[k], dtype=np.float64)
            curves = {k: (accum[k] / processed * 100) for k in accum}
            aucs = {k: float(auc(x_values, curves[k].tolist())) for k in accum}
            iterator.set_description(
                f"{args.method} I+:{aucs['image_positive']:.1f} I-:{aucs['image_negative']:.1f} "
                f"T+:{aucs['text_positive']:.1f} T-:{aucs['text_negative']:.1f}"
            )
        except Exception as exc:
            print(f"Skip: {exc}")
            import traceback; traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean = {k: (accum[k] / max(processed, 1) * 100).tolist() for k in accum}
    return {
        "method": args.method, "n_processed": processed, "seed": args.seed,
        "auc_image_positive": float(auc(x_values, mean["image_positive"])),
        "auc_image_negative": float(auc(x_values, mean["image_negative"])),
        "auc_text_positive": float(auc(x_values, mean["text_positive"])),
        "auc_text_negative": float(auc(x_values, mean["text_negative"])),
        "values_image_positive": [round(v, 2) for v in mean["image_positive"]],
        "values_image_negative": [round(v, 2) for v in mean["image_negative"]],
        "values_text_positive": [round(v, 2) for v in mean["text_positive"]],
        "values_text_negative": [round(v, 2) for v in mean["text_negative"]],
        "x_values": x_values,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--COCO_path", required=True)
    p.add_argument("--method", default="upse_e_final",
                     choices=["rm_with_lrp", "ours", "upse_e_final", "amf_mma", "upse_e_sota",
                              "upse_lxmert", "decompx"])
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="")
    args = p.parse_args()
    if not args.output:
        args.output = f"energy_lxmert_{args.method}_n{args.num_samples}.json"

    print(f"LXMERT {args.method} n={args.num_samples} seed={args.seed}")
    res = run_eval(args)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nImg+ {res['auc_image_positive']:.2f}  Img- {res['auc_image_negative']:.2f}")
    print(f"Txt+ {res['auc_text_positive']:.2f}  Txt- {res['auc_text_negative']:.2f}")
    print(f"Saved {args.output}")
