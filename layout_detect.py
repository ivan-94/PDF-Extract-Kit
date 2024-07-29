import os
import json
from typing import Any
import torch

import cv2
import numpy as np
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.extract_pdf import load_pdf_fitz
from PIL import Image, ImageDraw, ImageFont


def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model


cwd = os.getcwd()
models_dir = os.path.join(cwd, "models")
input_dir = os.path.join(cwd, "input")
output_dir = os.path.join(cwd, "output")

os.makedirs(output_dir, exist_ok=True)


layout_model = layout_model_init(os.path.join(models_dir, "Layout/model_final.pth"))

inputs = os.listdir(input_dir)


def draw(img: Any, page_res, output_path: str):
    color_palette = [
        (255, 64, 255),
        (255, 78, 0),
        (78, 78, 255),
        (255, 215, 135),
        (215, 100, 95),
        (100, 0, 48),
        (0, 175, 0),
        (95, 0, 95),
        (175, 95, 0),
        (95, 95, 0),
        (95, 95, 255),
        (95, 175, 135),
        (215, 95, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 95, 215),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]
    id2names = [
        "title",
        "plain_text",
        "abandon",
        "figure",
        "figure_caption",
        "table",
        "table_caption",
        "table_footnote",
        "isolate_formula",
        "formula_caption",
        " ",
        " ",
        " ",
        "inline_formula",
        "isolated_formula",
        "ocr_text",
    ]

    vis_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_img)
    for res in page_res:
        label = int(res["category_id"])
        if label > 15:  # categories that do not need visualize
            continue
        label_name = id2names[label]
        x_min, y_min = int(res["poly"][0]), int(res["poly"][1])
        x_max, y_max = int(res["poly"][4]), int(res["poly"][5])

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            fill=None,
            outline=color_palette[label],
            width=1,
        )
        fontText = ImageFont.truetype("assets/fonts/simhei.ttf", 15, encoding="utf-8")
        draw.text((x_min, y_min), label_name, color_palette[label], font=fontText)
    with open(output_path, "wb") as f:
        vis_img.save(f)


for file in inputs:
    file_path = os.path.join(input_dir, file)
    base_name = os.path.basename(file_path).split(".")[0]
    if os.path.isdir(file_path) or file.startswith("_"):
        continue

    doc_layout_result = []

    if file.endswith(".pdf"):
        print(f"Processing {file}...")
        # 转换为图片
        img_list = load_pdf_fitz(file_path, dpi=200)
        if img_list is None:
            continue

        for idx, img in enumerate(img_list):
            # 识别图片
            res = layout_model(img, ignore_catids=[])
            doc_layout_result.append(res)

            # 渲染结果
            draw(
                img,
                res["layout_dets"],
                os.path.join(output_dir, f"{base_name}_{idx}.jpg"),
            )
    else:
        print(f"Processing {file}...")
        img = np.array(Image.open(file_path).convert("RGB"))[:, :, ::-1]
        res = layout_model(img, ignore_catids=[])
        doc_layout_result.append(res)
        draw(
            img,
            res["layout_dets"],
            os.path.join(output_dir, f"{base_name}.jpg"),
        )

    # 出书结果
    with open(os.path.join(output_dir, f"{base_name}.json"), "w") as f:
        json.dump(doc_layout_result, f)

    torch.cuda.empty_cache()
