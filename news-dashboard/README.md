# Newsroom — Live News Dashboard

A self-contained, zero-dependency live news dashboard. One HTML file, no build
step, no API keys. It aggregates 30+ public RSS/Atom feeds from fifteen sources —
general news from **BBC News, The Guardian, The New York Times, Al Jazeera, NPR,
Sky News, DW, France 24, CNBC and ESPN**, plus academic and research coverage
from **arXiv (cs.AI / cs.LG papers), MIT News, ScienceDaily, Nature and MIT
Technology Review** — and presents them with dashboard UI/UX principles: a KPI
summary row, a single filter row, a lead story, a card grid, a stories-by-source
chart and a live wire. Categories span World, Business, Technology, Science,
AI Research, Sports and Health.

## Run it

Open `index.html` in any modern browser, or serve the folder:

```bash
cd news-dashboard
python3 -m http.server 8000
# → http://localhost:8000
```

It also works as a GitHub Pages site (`Settings → Pages`, point at this folder
or copy `index.html` to the Pages root).

## Intelligence features

**Client-side (no keys, computed in your browser):**

- **Story clustering** — stories about the same event are grouped across
  outlets by title/snippet similarity; each cluster renders once, with an
  "N sources" badge (hover it to see the other outlets' headlines).
- **Coverage-weighted Top tab** — Top ranks clusters by how many outlets are
  covering them, decayed by age, instead of raw recency.
- **Trending now** — terms gaining cross-source traction over the last 24
  hours, with an ↑ marker for accelerating ones; click a chip to filter.
- **For You tab** — learns from which stories you open (topic tokens, sources,
  categories, with gradual forgetting) and re-ranks the feed. The profile
  lives entirely in `localStorage`; nothing leaves the browser.

**Claude-powered (via the deploy workflow):**

- **Claude briefing** — every deploy (and on a 3-hour schedule) the workflow
  runs `scripts/generate-briefing.mjs`, which fetches all feeds server-side,
  sends the corpus to Claude (`claude-opus-5`, structured outputs), and writes
  `briefing.json` into the published site: the five most important stories,
  synthesized across outlets.
- **Signals** — the same call asks Claude for non-obvious patterns connecting
  seemingly unrelated stories, rendered under the briefing.
- **arXiv TL;DRs** — every arXiv paper gets a plain-English one-liner, shown
  inside its card on the AI Research tab.
- **Background context ("hidden knowledge")** — for each of the five briefing
  stories, Claude adds background the article text itself wouldn't tell you:
  who a named figure or organization is and their track record, what a
  technical/legal/financial term means, precedent from a similar past event,
  or whether a comparative claim ("worst since 2008") checks out. Shown as a
  collapsed "Background" disclosure under the story. This draws on Claude's
  training knowledge rather than the fetched articles — the prompt explicitly
  tells it to omit the note (`context: null`) rather than state anything it
  isn't confident is accurate, and the panel carries a visible disclaimer
  ("a starting point, not a citation") so it's never mistaken for sourced
  reporting.

To enable the Claude features, add an `ANTHROPIC_API_KEY` repository secret
(Settings → Secrets and variables → Actions). Without the key the workflow
still deploys cleanly and the dashboard simply hides the briefing panel.
Rough cost at current pricing: a few cents per generation, ~8 generations/day
on the default 3-hour schedule. Adjust the `cron` line in
`.github/workflows/deploy-pages.yml` to change the cadence.

## Features

- **Live aggregation** — 30+ RSS/Atom feeds fetched through a bounded
  concurrency pool (8 at a time, to stay friendly to the CORS relays),
  deduplicated and merged into one reverse-chronological stream. arXiv's
  daily paper batches are capped per feed so they can't drown the stream.
- **Auto-refresh** — every 5 minutes by default (configurable: 2/5/10 min or
  manual). Refreshing pauses while the tab is hidden and catches up on return;
  relative timestamps ("4m ago") tick every minute without refetching.
- **Filtering** — category tabs (Top, World, Business, Technology, Science,
  AI Research, Sports, Health), per-source toggle chips, and instant headline
  search, all in one row above the content.
- **Feed health** — a LIVE/OFFLINE pill, a sources-live stat tile, and a
  non-blocking notice when an outlet can't be reached. A dead feed degrades
  gracefully; the rest of the page stays current.
- **Mini-visualizations** — stat tiles (stories, sources live, past hour,
  publishing pace), a 12-hour publishing sparkline, and a stories-by-source
  bar chart with direct labels.
- **Design system** — validated colorblind-safe categorical palette with a
  fixed color per source (color follows the entity, never its rank; sources
  beyond the 8-hue token ceiling share a neutral swatch, with identity always
  carried by the visible name), selected
  light *and* dark themes (auto via `prefers-color-scheme`, manual toggle
  persisted to `localStorage`), thin marks with rounded data ends, text always
  in ink tokens.
- **Accessibility** — semantic landmarks, `aria-live` update announcements,
  visible focus rings, keyboard-navigable cards, source identity carried by
  label + swatch (never color alone), reduced-motion support.
- **Responsive** — three-column desktop layout collapses to a single column
  on tablets/phones; wide elements scroll within their own containers.

## How the feeds are fetched

Browsers block cross-origin RSS reads, so requests go through a chain of
public CORS relays (`allorigins` → `corsproxy.io` → `codetabs`); the first
relay that responds is remembered for the session and each feed has a 15 s
timeout. Public relays are convenient but rate-limited and best-effort — for
production use, put a tiny proxy of your own in front (a one-line
Cloudflare Worker or an nginx `proxy_pass` block) and replace the `PROXIES`
list at the top of the script in `index.html`.

Adding or removing outlets is a data change only: edit the `SOURCES` map
(name + palette slot) and the `FEEDS` list (source, category, URL) at the top
of the script.

All headlines link to, and remain the property of, their original publishers.
