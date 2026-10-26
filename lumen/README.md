# Lumen: search your photos by describing them

Lumen is a local app that reads your photo folders, builds an understanding of every picture with an
on-device vision model, and lets you find photos by describing them. You can type a word ("dogs"), a
phrase ("golden hour at the beach") or a memory ("the morning we took the ferry to the island and it was
windy"). You can then gather photos into named collections of smart previews.

It also groups faces into **people**, puts photos on a **map**, finds **duplicates**, and copies
collections into a folder.

Your photos never leave your computer. The models run locally after a one-time download. The
only things fetched from the internet are those models and, on the *Places* page, the map tiles.

## Quick start

Requires Python 3.10+.

```bash
cd lumen
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m lumen                    # opens http://127.0.0.1:8765 in your browser
```

1. Click **Choose photo folders** and add as many folders as you like (subfolders are included,
   external drives work too).
2. Lumen scans them and starts learning. The first run downloads the models (~1.5 GB, once).
   You can search while it works, and results grow as indexing continues.
3. Search, select photos, and add them to collections.

Options: `python -m lumen --port 9000 --no-browser`.

## What you can do

- **Describe anything.** Natural-language search ranks photos into *Spot on*, *Strong* and
  *Possible* matches. The default model (SigLIP 2) is multilingual, so Greek queries work too
  ("ηλιοβασίλεμα στη θάλασσα").
- **Leave things out.** `beach -people`, or `-"red car"` for a phrase.
- **Find similar.** Hover any photo and press ✦, or select several photos and choose *More like these*.
- **Collections.** Create named collections with an icon and colour.
  - *Hand-picked*: add exactly the photos you select. Your selection is kept as you move between
    searches, so you can gather photos from several searches and add them all at once.
  - *Smart*: describe the collection ("sunsets over the sea") and it fills itself, including with
    photos you add later. Use *Keep as is* to turn it into a regular collection.
  - Pick a cover, remove photos (with undo), **copy the originals into a folder** (optionally a new
    subfolder named after the collection; existing files are never overwritten), or download a zip.
- **People.** Faces are detected and grouped by person. Name someone once, then search with their
  name: "Maria at the beach" looks only at Maria's photos and ranks them by "at the beach". "Photos
  of Maria and Nikos" finds the photos they're both in. Merge groups that are the same person
  (*Same as…*), take wrong photos out (*Not them*, which Lumen remembers), or hide strangers.
- **Places.** Photos with GPS data appear on a map with photo-thumbnail clusters. The grid under the
  map shows whatever area you're looking at. Every photo with a location has *View on map* in the viewer.
- **Tidy up.** *Duplicates* finds copies of the same photo, including resized and re-saved
  versions, and pre-selects the best one (highest resolution, then largest file) to keep. *Similar
  shots* finds bursts of near-identical photos taken within two minutes, so you can keep the best of
  each moment. Removed photos go to the system Trash / Recycle Bin, so they can always be restored.
- **Explore.** The home page groups your library into themes it spotted (beaches, food, pets,
  celebrations…) and shows *On this day* memories from earlier years.
- **Library.** Browse everything by month, filter by year.
- **Viewer.** Full-size photo with date, camera, folder, the collections it belongs to and look-alikes,
  plus *Show in folder* and *Download*.

Keyboard: `/` search · `←` `→` browse in the viewer · `Esc` close or clear the selection ·
`Shift`+click to select a range · `Ctrl`/`⌘`+click to start selecting.

## Large libraries (10k–100k+)

- Indexing is incremental. Re-scans (*Sources → Check for new photos*, and on every start) only
  process new or changed files, and drop deleted ones.
- You can pause and resume indexing. Progress, speed and the time left are shown in the top bar.
- Rough CPU speeds: SigLIP 2 manages about 5–15 photos/s on a modern laptop, and CLIP ViT-B/32 is
  about twice as fast. Switch models in *Sources*. Switching re-uses the existing previews, so it's
  much faster than the first run.
- Faces, places and duplicate fingerprints are collected in the same pass. Libraries indexed by an
  earlier version get a quick catch-up pass ("Faces & places" in the top bar) rather than a full re-index.
- Formats: JPEG, PNG, WebP, HEIC/HEIF (iPhone), AVIF, TIFF, BMP, GIF.

## Where things are stored

Everything Lumen writes goes to `~/.lumen` (override with `LUMEN_HOME`):

| Path | What |
|---|---|
| `lumen.db` | SQLite: folders, photo metadata, embeddings, collections |
| `previews/` | 512 px WebP smart previews, and `previews/faces/` face thumbnails |
| `models/` | the downloaded models (image/text model, plus `faces/` for the face models) |

Set `LUMEN_FACES=0` to switch face grouping off.

Lumen never modifies or moves your photos. The only way it removes one is when you choose *Move to
Trash* in *Tidy up*, and that goes to the system Trash. Removing a folder, collection or person only
removes Lumen's own records.

## How it works

```
folders ──scan──▶ new/changed files ──▶ decode + EXIF + 512px preview ──▶ image embedding ──▶ SQLite
                                                                                      │
query ──▶ text embedding (averaged over caption-style phrasings) ──▶ cosine vs. all ──┘
                                                                     ▼
                                       robust z-score vs. the whole library ──▶ tiers
```

- `lumen/indexer.py`: background worker for scanning, previews, EXIF and batched embedding.
- `lumen/embedder.py`: [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, no PyTorch)
  wrappers for the image and text halves of the model.
- `lumen/vectors.py`: the in-memory embedding matrix. 100k photos fit in about 300 MB of RAM, and a search takes milliseconds.
- `lumen/search.py`: query parsing (including people's names), ranking, similar photos, Explore themes, On this day.
- `lumen/faces.py`: face detection ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)),
  face fingerprints ([SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)),
  and grouping. New faces join the nearest existing person, or form new people by centroid
  clustering, so named people never get reshuffled.
- `lumen/duplicates.py`: 64-bit difference hashes with locality-sensitive bucketing, confirmed by the
  image model, plus time-window grouping for similar shots.
- `lumen/jobs.py`: background jobs (copying a collection to a folder).
- `lumen/server.py`: FastAPI JSON API. It binds to `127.0.0.1` only.
- `lumen/static/`: the UI, plain HTML/CSS/JS with no build step. [Leaflet](https://leafletjs.com) and
  Leaflet.markercluster are bundled under `static/vendor/` (BSD-2 / MIT). Map tiles are © OpenStreetMap
  contributors, © CARTO.

Match strength is a robust z-score (median/MAD) of each photo's similarity against the whole
library. This keeps the tiers meaningful for any model and any library, even when a big share of
your photos match.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests use a deterministic stand-in embedder (`LUMEN_EMBEDDER=fake`), so they run offline and
without downloading a model. The face test needs the two OpenCV face models. Put them in a folder
and run `LUMEN_TEST_FACE_MODELS=/path/to/folder python -m pytest tests`; without it, that test is
skipped. It also needs `scikit-image` for its sample photo.
