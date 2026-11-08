# README assets

`lapras.gif` is the animated pixel-art Lapras mascot, sourced from
[Pokémon Showdown's Gen 5 sprite collection](https://play.pokemonshowdown.com/sprites/gen5ani/lapras.gif).
The original 32-frame GIF is preserved unchanged. Pokémon artwork belongs to its
respective owners and is not covered by Laplace's MIT code license.

`social-preview.png` is a 1280 × 640 repository card. Upload it in GitHub:
Settings → General → Social preview → Edit → Upload an image.
Committing the file alone does not configure GitHub's social preview.

To rebuild the social-preview card, install Pillow and run:

```sh
python scripts/build_media.py --font /path/to/a/monospace.ttf
```

The social card uses Windows Consolas and labels its rating as a historical peak.
