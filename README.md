<p align="center"><img src="docs/logo.svg" alt="Haunted Window" width="560"></p>

Halloween scenes rear-projected onto windows that notice when people walk by. A camera finds visitors on the street, and whatever is "inside" reacts to them. It runs on a Raspberry Pi 5 with one projector per window.

## Scenes

| Scene | What happens |
| --- | --- |
| **Eyes** | Giant eyes follow visitors, wake up with a voice line, and turn demonic when someone comes close. |
| **Hands** | Bloody hands slam against the glass, slide down, and leave smeared prints behind. |
| **Figure** | A woman's silhouette in a nightgown wanders behind the glass, walks to your window and tilts her head. Come too close and she rushes the glass. |
| **Pumpkins** | Two talking jack-o'-lanterns that lip-sync to the voice clips and follow visitors with their eyes. Friendly enough for daytime. |

In `auto` mode it shows pumpkins in the daytime and rotates through the scary scenes after `--scary-from` (19:00 by default).

## Running

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --source mouse --windowed
```

`--source mouse` simulates a visitor: move the mouse in the "street" window. On the Pi, `--source picam` (the default) uses the camera module through `picamera2`, and `--source webcam` uses OpenCV.

A dashboard runs at `http://<host>:8080`. Use it to pick the scene, play sounds, trigger a scare and watch the tracker.

### Useful options

- `--eye-x A B`: window centres in metres from the camera (+ = right, as seen from the street)
- `--eye-height`, `--cam-height`, `--hfov`, `--vfov`: geometry used to place visitors
- `--window-width`: metres of window covered by one projector image
- `--hflip`: mirror the output for rear projection
- `--scene auto|eyes|hands|figure|pumpkins`

## Raspberry Pi service

```sh
rsync -av --exclude .venv --exclude settings.json --exclude __pycache__ ./ pi@raspberrypi.local:~/haunted-window/
ssh pi@raspberrypi.local 'cd ~/haunted-window && python -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt'
ssh pi@raspberrypi.local 'mkdir -p ~/.config/systemd/user && cp ~/haunted-window/haunted-window.service ~/.config/systemd/user/ && systemctl --user enable --now haunted-window'
```

## License

The code is MIT licensed. See [assets/CREDITS.md](assets/CREDITS.md) for the bundled images and sounds, which include one CC BY-SA 4.0 image.
