# PaliGemma

This repository contains the implementation and inference pipeline for the Paligemma-3B model, reproduced using PyTorch. Pre-trained weights of the model are utilized to enable efficient and accurate inference.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)

---

## Introduction
Paligemma-3B is a Vision Language Model(VLM) by Google designed for Image-text to text tasks. This repository provides a PyTorch implementation of the model along with pre-trained weights to facilitate inference.

---

## Features

- **Model Implementation**: A faithful reproduction of the Paligemma-3B model in PyTorch.
- **Pre-trained Weights**: Utilizes pre-trained weights for accurate predictions.
- **Inference Pipeline**: Includes a streamlined pipeline for performing inference with minimal setup.

---

## Requirements

Before getting started, ensure you have the following:

- Python >= 3.10
- PyTorch >= 2.0
- Additional dependencies listed in `requirement.txt`

Install the required packages with:
```bash
pip install -r requirement.txt
```

---

## Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/ParikshitGehlaut/PaliGemma.git
cd PaliGemma
```

---

## Usage

### Inference
1. **Download Pre-trained Weights**:
   Download model weights from [Hugging face](https://huggingface.co/google/paligemma-3b-pt-224) and ensure the pre-trained weights for Paligemma-3B are downloaded and placed in the `weights/` directory.

2. **Run the Inference Script**:
   Use the provided `launch_inference.sh` script to perform inference on your input data.

   Example:
   ```bash
   chmod +x launch_inference.sh
   ./launch_inference.sh
   ```

   arguments:
   - `--model_path`: Path to the pre-trained weights.
   - `--prompt`: text prompt.
   - `--image_file_path`: Path to input image.
   - `--max_tokens_to_generate`: maximun length of text output.
   - `--temperature`: parameter that controls the randomness of predictions.
   - `--top_p`: variables used in the decoding strategies.
   - `--do_sample`: boolean variable.
   - `--only_cpu`: set to False if GPU is available.

---

## File Structure

```
.
├── test_images/              # Input image
├── weights/                  # Pre-trained weights
├── inference.py              # Script for inference
├── launch_inference.sh       # To run inference
├── modeling_gemma.py         # PyTorch implementation 
├── modeling_siglip.py        # PyTorch implementation
├── processing_paligemma.py   # PyTorch implementation
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── utils.py                  # loading hf model
```

---

## Acknowledgements

```
@article{beyer2024paligemma,
      title={{PaliGemma: A versatile 3B VLM for transfer}},
      author={Lucas Beyer and Andreas Steiner and André Susano Pinto and Alexander Kolesnikov and Xiao Wang and Daniel Salz and Maxim Neumann and Ibrahim Alabdulmohsin and Michael Tschannen and Emanuele Bugliarello and Thomas Unterthiner and Daniel Keysers and Skanda Koppula and Fangyu Liu and Adam Grycner and Alexey Gritsenko and Neil Houlsby and Manoj Kumar and Keran Rong and Julian Eisenschlos and Rishabh Kabra and Matthias Bauer and Matko Bošnjak and Xi Chen and Matthias Minderer and Paul Voigtlaender and Ioana Bica and Ivana Balazevic and Joan Puigcerver and Pinelopi Papalampidi and Olivier Henaff and Xi Xiong and Radu Soricut and Jeremiah Harmsen and Xiaohua Zhai},
      year={2024},
      journal={arXiv preprint arXiv:2407.07726}
}
```

---

Feel free to raise issues or contribute to this repository to improve its functionality.
