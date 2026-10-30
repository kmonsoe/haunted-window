<p align="center"><img src="docs/logo.svg" alt="Haunted Window" width="560"></p>

Halloween scenes rear-projected onto windows that notice when people walk by. A camera finds visitors on the street, and whatever is "inside" reacts to them. It runs on a Raspberry Pi 5 with one projector per window.

## Scenes

| Scene | What happens |
| --- | --- |
| **Eyes** | Giant eyes follow visitors, wake up with a voice line, and turn demonic when someone comes close. |
| **Hands** | Bloody hands slam against the glass, slide down, and leave smeared prints behind. |
| **Figure** | A woman's silhouette in a nightgown wanders behind the glass, walks to your window and tilts her head. Come too close and she rushes the glass. |
| **Pumpkins** | Two talking jack-o'-lanterns that lip-sync to the voice clips and follow visitors with their eyes. Friendly enough for daytime. |
| **Breach** | A biohazard containment cell holding a massive horned beast. Warning beacons sweep and flash, claws slam against the glass with screen trauma, and fractures spiderweb across the window. |
| **Spiders** | Cobweb-draped crypt windows where small spiders scurry across the glass, while a giant 8-eyed Matriarch stalks visitors and drops straight into their faces on a silk thread. |

In `auto` mode it rotates through the scary scenes around visitors, any time of day.

## Our setup

- **Computer:** a Raspberry Pi 5 (2 GB) running Raspberry Pi OS. It sits indoors between the two windows and starts the show through a systemd user service.
- **Projectors:** two [Minikino mini projectors](https://minikino.no/produkter/mini-projektor) (720p, 150 ANSI lumens), one per window. Each connects to one of the Pi's two micro-HDMI ports. They stand inside the house and project onto a frosted glass film on the windows. The film catches the image, so it's seen from the street. They can flip the image themselves; otherwise use `--hflip`.
- **Camera:** a Raspberry Pi Camera Module 3 Wide (about 102°×67°), mounted in the corner of a window and looking out at the street. It finds visitors by motion detection.
- **Sound:** a JBL Flip 3 Bluetooth speaker plays the voices, thuds and background music.
- **Control:** the web dashboard on port 8080, opened from a phone on the home network, or directly over the Pi's own hotspot when that wifi is flaky (see [Direct connection](#direct-connection)).

The default `--eye-x`, `--cam-height` and `--eye-height` values match our house. Measure your own windows and camera position, then set these options to match.

The Pi renders at 240p and 20 fps (`--render-height 240 --fps 20`). Both projectors are 720p, so the picture is scaled up. Because it's projected onto frosted film in the dark, the lower resolution isn't noticeable.

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
- `--scene auto|eyes|hands|figure|pumpkins|breach|spiders`

## Raspberry Pi service

```sh
rsync -av --exclude .venv --exclude settings.json --exclude __pycache__ ./ pi@raspberrypi.local:~/haunted-window/
ssh pi@raspberrypi.local 'cd ~/haunted-window && python -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt'
ssh pi@raspberrypi.local 'mkdir -p ~/.config/systemd/user && cp ~/haunted-window/haunted-window.service ~/.config/systemd/user/ && systemctl --user enable --now haunted-window'
```

## Direct connection

The Pi's wifi chip can't stay on the home network and run its own hotspot at the same time, so the two are separate NetworkManager connection profiles on `wlan0` and you switch between them.

To go direct (useful when the house wifi doesn't reach the garage): join the **HauntedWindow** wifi network from your phone, then open `http://10.42.0.1:8080`.

```sh
# switch the Pi to its own hotspot
ssh pi@raspberrypi.local 'sudo nmcli connection up Hotspot'

# switch back to the home network
ssh pi@raspberrypi.local 'sudo nmcli connection up "Rabben 28"'
```

Switching to the hotspot drops the Pi off the home network (and any SSH session over it), so do it last.

## License

The code is MIT licensed. See [assets/CREDITS.md](assets/CREDITS.md) for the bundled images and sounds, which include one CC BY-SA 4.0 image.
