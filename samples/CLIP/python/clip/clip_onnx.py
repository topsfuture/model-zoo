import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Union, List
from packaging import version
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize
)
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# Preprocess
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def create_clip_preprocess(pixels=224):
    return Compose([
        Resize(pixels, interpolation=BICUBIC),
        CenterCrop(pixels),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    _tokenizer = _Tokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)



def clip_infer(image_model, text_model, projection, image_path, classnames, topk):
    print(f"[INFO] Loading models...")
    image_sess = ort.InferenceSession(image_model, providers=["CPUExecutionProvider"])
    text_sess = ort.InferenceSession(text_model, providers=["CPUExecutionProvider"])
    text_proj = np.load(projection)

    print(f"[INFO] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    preprocess = create_clip_preprocess(224)
    image_input = preprocess(image).unsqueeze(0).numpy()

    prompts = [f"a photo of a {c}" for c in classnames]
    tokenized = torch.cat([tokenize(p) for p in prompts]).numpy()

    # Text ONNX inference
    text_out = text_sess.run(None, {"text": tokenized})[0]  # (N,77,512)

    eot_id = 49407
    eot_pos = (tokenized == eot_id).argmax(axis=1)  # (N,)
    eot_hidden = text_out[np.arange(text_out.shape[0]), eot_pos]  # (N,512)

    text_feat = eot_hidden @ text_proj
    text_feat = text_feat / np.linalg.norm(text_feat, axis=1, keepdims=True)

    img_feat = image_sess.run(None, {"image": image_input})[0]
    img_feat = img_feat / np.linalg.norm(img_feat, axis=1, keepdims=True)

    # Similarity
    logits = 100.0 * (img_feat @ text_feat.T)
    probs = softmax(logits, axis=-1).reshape(-1)

    # Top-k
    topk_idx = probs.argsort()[-topk:][::-1]

    print("\nTop Predictions:")
    for idx in topk_idx:
        print(f"{classnames[idx]:>20s}: {probs[idx]*100:.2f}%")

    return topk_idx, probs



def main():
    parser = argparse.ArgumentParser(description="CLIP ONNX Inference")

    parser.add_argument("--image_model", type=str, required=True,
                        help="Path to image encoder ONNX model")
    parser.add_argument("--text_model", type=str, required=True,
                        help="Path to text encoder ONNX model")
    parser.add_argument("--projection", type=str, required=True,
                        help="Path to text projection .npy file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")

    parser.add_argument("--classnames", type=str, required=False,
                        help="Comma-separated class names (default: CIFAR100 classes)")

    parser.add_argument("--topk", type=int, default=5,
                        help="Top-K predictions to show")

    args = parser.parse_args()

    # If no classnames use CIFAR100 by default
    if args.classnames is None:
        from torchvision.datasets import CIFAR100
        classnames = CIFAR100(root="~/.cache", download=True).classes
    else:
        classnames = [c.strip() for c in args.classnames.split(",")]

    clip_infer(
        image_model=args.image_model,
        text_model=args.text_model,
        projection=args.projection,
        image_path=args.image,
        classnames=classnames,
        topk=args.topk
    )


if __name__ == "__main__":
    main()
