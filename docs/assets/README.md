# Demo assets

`demo.gif` visualizes the first four entries from an actual archived decision
trace. `demo-trace.json` retains the source replay URL and recorded values.
The bars show pooled search shares, not win probabilities; a mixed policy can
select a near-tied action rather than the largest bar.

`social-preview.png` is a 1280 × 640 repository card. Upload it in GitHub:
Settings → General → Social preview → Edit → Upload an image.
Committing the file alone does not configure GitHub's social preview.

To rebuild both assets, install Pillow and run:

```sh
python scripts/build_media.py --font /path/to/a/monospace.ttf
```

The committed renders use Windows Consolas. No Pokémon artwork is bundled in
these assets. The social card's rating is explicitly a historical peak.
