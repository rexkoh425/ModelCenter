# Video Upscale Tool

1. Install dependencies: `ffmpeg` on PATH and `pip install pyyaml`.
2. Edit `video_upscale_config.yaml` in this folder to point at the input file and tweak parameters.
3. Run the script from the repo root: `python VideoProcessingLab/video_upscale.py`.

You can override anything from the config on the CLI. For example, to keep the config defaults but change the resolution for one-off run:

```
python VideoProcessingLab/video_upscale.py --width 2560 --height 1440
```

Or supply a different config file entirely:

```
python VideoProcessingLab/video_upscale.py --config /path/to/4k.yaml
```
