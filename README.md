# I2V-DETACH: Source Grounding Detachment for Unauthorized Image-to-Video Generation

Official PyTorch implementation of **I2V-DETACH: Source Grounding Detachment for Unauthorized Image-to-Video Generation**, accepted to **NeurIPS 2026**.

[[Paper]](https://arxiv.org/abs/2610.23287)

I2V-DETACH protects source images from unauthorized image-to-video generation by suppressing source-video grounding and strengthening intra-video interactions, reducing the propagation of source-image content into generated videos.

## Supported Models

- CogVideoX-5B
- LTX-Video-2B

The method optimizes attention logits captured from the transformer blocks. The attention modules must expose the query and key tensors as <code>temp_query</code> and <code>temp_key</code>.

## Installation

Install the tested dependencies:

~~~bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements_cog-ltx.txt
~~~

LTX-Video-2B additionally expects the <code>ltxv-2b-0.9.6-dev-04-25.safetensors</code> checkpoint at the path configured in <code>main_all.py</code>.

## Usage

### CogVideoX-5B

~~~bash
python main_all.py --image_path path/to/image.jpg --prompt "your prompt" --method cog --model_id THUDM/CogVideoX-5b-I2V --apply_ES --steps 50 --eps 8 --step_size 1.0 --fixed_t_index 5
~~~

### LTX-Video-2B

~~~bash
python main_all.py --image_path path/to/image.jpg --prompt "your prompt" --method ltx --apply_ES --steps 50 --eps 8 --step_size 1.0 --fixed_t_index 5
~~~

Generated adversarial images, perturbations, videos, and run metadata are saved under <code>Results_&lt;model&gt;/</code>.

## Citation

~~~bibtex
@inproceedings{lee2026i2vdetach,
  title={I2V-DETACH: Source Grounding Detachment for Unauthorized Image-to-Video Generation},
  author={Lee, Chanhui and Song, Yeonghwan and Kang, Yewon and Son, Jeany},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
~~~
