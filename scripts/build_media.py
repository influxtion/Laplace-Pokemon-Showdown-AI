"""Render honest demo graphics from recorded turns. Requires Pillow.

python scripts/build_media.py --font C:/Windows/Fonts/consola.ttf
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
BG, PANEL, INK, MUTED, CYAN, GOLD = "#0b1420", "#132233", "#eef5fb", "#8fa7ba", "#59e3cf", "#ffd782"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default="DejaVuSansMono.ttf")
    args = parser.parse_args()

    def font(size):
        return ImageFont.truetype(args.font, size)

    def text(draw, xy, value, size=20, fill=INK):
        draw.text(xy, value, font=font(size), fill=fill)

    source = json.loads((ASSETS / "demo-trace.json").read_text())
    frames = []
    for turn in source["turns"]:
        for step in range(10):
            im = Image.new("RGB", (1000, 560), BG)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 8, 560), fill=CYAN)
            text(d, (40, 25), "LAPLACE", 28, CYAN)
            text(d, (740, 32), "RECORDED DEMO", 18, MUTED)
            text(d, (40, 82), f"TURN {turn['turn']:02d}  /  GEN 9 RANDOM BATTLE", 24)
            d.rounded_rectangle((40, 130, 960, 213), 12, fill=PANEL)
            text(d, (62, 145), "OUR ACTIVE", 14, MUTED)
            text(d, (62, 170), turn["me"], 22)
            text(d, (545, 145), "OPPONENT", 14, MUTED)
            text(d, (545, 170), turn["opp"], 22)
            text(d, (40, 238), "CANDIDATE", 15, MUTED)
            text(d, (730, 238), "SEARCH SHARE", 15, MUTED)
            for i, (move, share) in enumerate(turn["top"]):
                y = 275 + 53 * i
                color = CYAN if move == turn["choice"] else MUTED
                text(d, (40, y), move.replace("switch ", "switch: "), 21, color)
                d.rounded_rectangle((440, y + 4, 880, y + 28), 5, fill=PANEL)
                width = int(440 * share * min(1, (step + 1) / 5))
                if width:
                    d.rounded_rectangle((440, y + 4, 440 + width, y + 28), 5, fill=color)
                text(d, (899, y), f"{share:.0%}", 20, color)
            d.line((40, 444, 960, 444), fill="#284153", width=1)
            text(d, (40, 466), "PLAYS  " + turn["choice"], 25, GOLD)
            text(d, (40, 518), "Real archived decisions / shares are not win probabilities", 15, MUTED)
            frames.append(im.quantize(colors=96))
    frames[0].save(ASSETS / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=([100] * 9 + [1800]) * len(source["turns"]), loop=0, optimize=True)

    im = Image.new("RGB", (1280, 640), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, 1280, 40):
        d.line((x, 0, x, 640), fill="#102130")
    for y in range(0, 640, 40):
        d.line((0, y, 1280, y), fill="#102130")
    d.rectangle((0, 0, 12, 640), fill=CYAN)
    text(d, (65, 52), "OPEN SOURCE / POKEMON SHOWDOWN AI", 20, CYAN)
    text(d, (60, 127), "LAPLACE", 105)
    text(d, (65, 265), "Search the hidden game.", 36)
    text(d, (65, 323), "Gen 9 Random Battle", 25, MUTED)
    text(d, (65, 362), "Monte Carlo search + a trained value net", 21, MUTED)
    d.rounded_rectangle((902, 127, 1215, 405), 16, fill=PANEL)
    text(d, (932, 153), "HISTORICAL PEAK", 18, MUTED)
    text(d, (932, 206), "2231", 72, CYAN)
    text(d, (935, 301), "ELO", 27)
    text(d, (935, 353), "~83.3 GXE", 23, MUTED)
    d.line((65, 482, 1215, 482), fill="#284153", width=2)
    text(d, (65, 520), "Watch real games. Run it. Improve it.", 26, GOLD)
    text(d, (65, 577), "github.com/influxtion/Laplace-Pokemon-Showdown-AI", 19, MUTED)
    im.save(ASSETS / "social-preview.png", optimize=True)
    print("Generated demo.gif and 1280x640 social-preview.png")


if __name__ == "__main__":
    main()
