# Lumen: search your photos by describing them

Lumen is a local app that reads your photo and video folders, builds an understanding of every
picture with on-device AI models, and lets you find anything by describing it. You can type a word
("dogs"), a phrase ("golden hour at the beach"), a feeling ("moody and dark") or a memory ("Maria
smiling in Crete last summer"). Results can be gathered into collections, turned into slideshows,
and shared.

Your photos never leave your computer. The models run locally after a one-time download. The only
things fetched from the internet are those models and, on the *Places* page, the map tiles.

## Quick start

**Desktop app:** download the build for Windows, macOS or Linux from the *Build Lumen desktop app*
GitHub Action, unzip it, and double-click **Lumen**. It sits in the menu bar or system tray.

**From source** (Python 3.10+):

```bash
cd lumen
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m lumen                    # opens http://127.0.0.1:8765 in your browser
```

1. Click **Choose photo folders** and add as many folders as you like (subfolders are included,
   external drives work too).
2. Lumen scans them and starts learning. The first run downloads the models (~1.7 GB, once). You can
   search while it works, and results grow as indexing continues. New photos are picked up
   automatically while Lumen runs.
3. Search, select photos, and add them to collections.

Options: `python -m lumen --port 9000 --no-browser --lan --tray`.

## On a Mac

1. **Python 3.10 or newer.** The `python3` built into macOS is too old. Install it from
   [python.org](https://www.python.org/downloads/macos/), or with `brew install python@3.12`.
2. **Double-click `Lumen.command`** in this folder. The first run sets everything up (a few minutes),
   then Lumen opens in your browser. Keep the Terminal window open while you use it. Close it to stop
   Lumen. If macOS says the file can't be opened, right-click it and choose **Open**.
3. **Your Apple Photos library:** add it as a folder. *Apple Photos* is a shortcut in the folder
   picker. Lumen reads only the original photos inside it, not Photos' own thumbnails and caches.
   - macOS protects the library. If Lumen says it can't read it, open **System Settings → Privacy &
     Security → Full Disk Access**, turn on **Terminal** (or **Lumen**, for the desktop app), and
     restart Lumen.
   - With **iCloud Photos → Optimize Mac Storage**, most originals live only in iCloud, and Lumen can
     index only what's on the Mac. Choose **Download Originals to this Mac** in Photos' settings if
     you have the space, or export the photos you care about to a folder.
4. **Speed:** on Apple Silicon, Lumen uses the Neural Engine/GPU through Core ML automatically. *Sources →
   This computer* shows what it's using.

## Searching

| Type | Finds |
|---|---|
| `golden hour at the beach` | anything you can describe, in ~100 languages (`ηλιοβασίλεμα στη θάλασσα`) |
| `moody and dark`, `cinematic street photography` | moods and styles |
| `Maria smiling`, `everyone laughing at dinner` | named people, and their actual expressions |
| `Luna in the snow` | named pets |
| `beach in Crete`, `dinner in Thessaloniki`, `Κρήτη` | places, from GPS, offline |
| `summer 2019`, `last Christmas`, `June`, `the 90s`, `πέρσι` | dates; `morning`/`at night`/`weekend` nudge the ranking |
| `"taverna"`, `receipt total` | text written *in* the photo (signs, menus, tickets, screenshots) |
| `#holidays`, `my favourites` | your own tags and favourites |
| `beach -people`, `-"red car"` | leave things out |

You can combine all of these: "Maria and Nikos at the beach in Crete summer 2019 -boats". You can
also search **by picture** (📷 next to the search box, or drop any image onto Lumen) and **by
colour** (🎨). If a search isn't quite right, press **👍 / 👎** on a few results and Lumen steers
towards what you mean.

Results come in *Spot on*, *Strong* and *Possible* tiers. *Best looking first* sorts each tier by
photo quality. Videos are searched frame by frame: a match opens at the moment that fits.

## Everything else

- **Memories.** Lumen cuts your library into events and joins days away from home into trips:
  "Trip to Chania · 12–15 Jul 2023", "Christmas in Paris", "Birthday with Nikos". Each one can
  be played as a slideshow, shared, or saved as a collection.
- **Slideshows.** Any collection, memory or search plays full-screen with slow zooms and crossfades.
- **Collections.** Named, with an icon and colour.
  - *Hand-picked*: your selection is kept as you move between searches.
  - *Smart*: they fill themselves from a description.
  - Pick a cover, remove photos with undo, **copy the originals to a folder**, or download a zip.
- **Share.** Turn a collection or memory into a **web gallery** (a zip with a page you can send or
  upload anywhere) or a **slideshow video** (1080p MP4 that plays on any phone or TV).
- **People & pets.** Faces, cats and dogs are recognised and grouped. Name them once, then use the
  names in searches. You can merge groups (*Same as…*), correct mistakes (*Not them*), or hide strangers.
- **What Lumen sees.** The viewer shows a short description and tags ("A dog on the beach at sunset ·
  golden light"). Any text found in the photo can be copied.
- **Best shots.** A quality score combines the image model's sense of a good photo with measured focus
  and exposure. Once you have about ten favourites or 4–5 star photos, it also learns **your taste**.
- **Favourites, ratings and tags.** ♥, ★1–5 and your own #tags, on one photo or a whole selection.
  Optionally they are written to **XMP sidecars** (`IMG_0001.xmp`) so Lightroom, Bridge, Capture One
  and digiKam see them. Sidecars those apps already wrote are read in.
- **Places.** A map with photo-thumbnail clusters, place names in the viewer, and shortcuts to your
  most-visited places.
- **Tidy up.** Finds duplicates (including resized copies) and bursts of similar shots, with the best
  one pre-selected. Removals go to the system Trash.
- **Phone access.** Turn it on in *Sources* to browse and search from phones and tablets on your Wi-Fi.
  Scan the QR code and enter the PIN. Phones can't browse your computer's folders, delete photos or
  change settings.
- **Start with my computer**, **GPU acceleration**, and live folder watching are switches in *Sources*.

Keyboard: `/` search · `←` `→` browse · `Space` pause a slideshow · `Esc` close or clear the selection ·
`Shift`+click to select a range · `Ctrl`/`⌘`+click to start selecting.

## Large libraries (10k–100k+)

- Indexing is incremental and resumable. Only new or changed files are processed, and deleted ones
  are dropped. Folder watching catches changes while Lumen runs.
- You can pause and resume. Progress, speed and time left are shown in the top bar.
- On a modern laptop CPU, SigLIP 2 manages about 5–15 photos/s, and CLIP ViT-B/32 is about twice as fast.
- **GPU acceleration** is used automatically when ONNX Runtime has one:
  - **Apple Silicon:** Core ML. It's included in the standard `onnxruntime` on macOS.
  - **NVIDIA:** run `pip uninstall onnxruntime && pip install onnxruntime-gpu`. Needs CUDA 12 and cuDNN 9.
  - **Windows, any GPU:** run `pip uninstall onnxruntime && pip install onnxruntime-directml`.

  If an accelerator fails, Lumen falls back to the CPU on its own.
- The slower extras only run where they're useful:
  - text recognition, on photos that look like they contain text (`LUMEN_OCR=all|off`). It reads
    Latin-script languages, Chinese and Japanese out of the box, and Greek once a ~10 MB model has
    downloaded on first use (`LUMEN_OCR_GREEK=0` skips it). Searches ignore case and accents, so
    "καλημερα" finds "Καλημέρα";
  - the pet detector, on photos that look like they show a cat or a dog (`LUMEN_PETS=all|off`).
- Libraries indexed by an older version get a quick catch-up pass rather than a full re-index.
- **Formats:**
  - photos: JPEG, PNG, WebP, HEIC/HEIF (iPhone), AVIF, TIFF, BMP, GIF;
  - videos: MP4, MOV, M4V, 3GP, WebM, MKV, AVI, MTS.

  Browsers play H.264/HEVC/VP9. Other formats open in your video player.

## Where things are stored

Everything Lumen writes goes to `~/.lumen` (override with `LUMEN_HOME`):

| Path | What |
|---|---|
| `lumen.db` | SQLite: folders, photo metadata, embeddings, faces, collections, text found in photos |
| `previews/` | 512 px WebP smart previews, and `previews/faces/` face and pet thumbnails |
| `models/` | the downloaded models: the image/text model, plus `faces/` (YuNet, SFace, FER+, YOLOX) |

Lumen never modifies or moves your photos. It removes one only when you choose *Move to Trash* in
*Tidy up*, and that goes to the system Trash. With XMP sidecars switched on, it writes `.xmp` files
next to photos, never into them. Set `LUMEN_FACES=0` to switch face and pet recognition off.

## How it works

```
folders ──scan/watch──▶ new/changed files ──▶ decode + EXIF/GPS + 512px preview ──▶ image embedding ──▶ SQLite
                                 │                       ├─ faces → fingerprint, expression, people
                                 │                       ├─ focus, exposure, palette, difference hash
                videos ──────────┘ frames every 5 s      └─ later: places, text (OCR), pets, XMP import

query ──▶ who / where / when / "text" / #tags taken out as filters
      ──▶ the rest → text embedding ──▶ cosine vs. photos and video frames
      ──▶ robust z-score vs. the library + quality, expression, text, colour and 👍/👎 nudges ──▶ tiers
```

| Module | Does |
|---|---|
| `indexer.py` | background worker: scan, previews, EXIF/GPS, batched embedding, catch-up passes |
| `embedder.py` | [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, no PyTorch) wrappers, GPU selection |
| `vectors.py` | the in-memory embedding matrix (100k photos ≈ 300 MB, a search takes milliseconds) |
| `search.py` | query understanding and ranking, relevance feedback, Explore themes, moods |
| `dates.py`, `places.py` | dates in queries; offline reverse geocoding ([GeoNames](https://www.geonames.org/) via `reverse_geocoder`) |
| `faces.py`, `pets.py` | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) faces, [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) fingerprints, [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) expressions, [YOLOX](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox) cats and dogs, incremental grouping |
| `quality.py`, `colors.py`, `describe.py` | best-shot score and personal taste, colour palettes, zero-shot tags and captions |
| `ocr.py` | [RapidOCR](https://github.com/RapidAI/RapidOCR) text recognition with SQLite FTS5 search |
| `events.py` | memories: events by time gaps, trips away from home, titles |
| `duplicates.py` | difference hashes + locality-sensitive hashing, confirmed by the image model; bursts |
| `marks.py` | favourites, ratings, tags, XMP sidecars |
| `share.py`, `jobs.py` | web gallery and H.264 slideshow export; background jobs |
| `lan.py` | phone access: second listener, PIN (PBKDF2), signed session cookies, rate limiting |
| `watcher.py`, `autostart.py`, `tray.py` | live folder watching, start with the computer, tray icon |
| `server.py`, `static/` | FastAPI JSON API (localhost only, unless phone access is on) and the no-build UI |

Bundled: [Leaflet](https://leafletjs.com) and Leaflet.markercluster (BSD-2 / MIT). Map tiles are ©
OpenStreetMap contributors, © CARTO.

**Known limits.**
- Text recognition reads print well; handwriting and decorative lettering only partly. If the
  Greek model can't be downloaded (offline), Lumen reads without it and tries again next start,
  then re-reads photos that already have text.
- Place names come from the nearest town, which can be a suburb. Searching for the city's name
  still works.
- Pets are told apart by appearance, so two similar-looking dogs may share a group until you split them.

## Desktop app

`packaging/lumen.spec` builds a double-click app with [PyInstaller](https://pyinstaller.org)
(`pyinstaller packaging/lumen.spec` from this folder). The GitHub workflow
`.github/workflows/lumen-desktop.yml` builds Windows, macOS and Linux versions. Run it by hand or push
a `lumen-v*` tag. The macOS app isn't notarised, so the first time, right-click it and choose
**Open**.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests use a deterministic stand-in embedder (`LUMEN_EMBEDDER=fake`), so they run offline without
downloading a model. The face, expression and pet tests need the four small models (YuNet, SFace,
FER+, YOLOX) in a folder: `LUMEN_TEST_FACE_MODELS=/path/to/folder python -m pytest tests`. Without
them, those tests are skipped.
