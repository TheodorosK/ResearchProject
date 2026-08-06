# Newsroom — Live News Dashboard

A self-contained, zero-dependency live news dashboard. One HTML file, no build
step, no API keys. It aggregates the public RSS feeds of six major outlets —
**BBC News, The Guardian, The New York Times, Al Jazeera, NPR and Sky News** —
and presents them with dashboard UI/UX principles: a KPI summary row, a single
filter row, a lead story, a card grid, a stories-by-source chart and a live wire.

## Run it

Open `index.html` in any modern browser, or serve the folder:

```bash
cd news-dashboard
python3 -m http.server 8000
# → http://localhost:8000
```

It also works as a GitHub Pages site (`Settings → Pages`, point at this folder
or copy `index.html` to the Pages root).

## Features

- **Live aggregation** — 19 RSS/Atom feeds fetched in parallel, deduplicated
  and merged into one reverse-chronological stream.
- **Auto-refresh** — every 5 minutes by default (configurable: 2/5/10 min or
  manual). Refreshing pauses while the tab is hidden and catches up on return;
  relative timestamps ("4m ago") tick every minute without refetching.
- **Filtering** — category tabs (Top, World, Business, Technology, Science),
  per-source toggle chips, and instant headline search, all in one row above
  the content.
- **Feed health** — a LIVE/OFFLINE pill, a sources-live stat tile, and a
  non-blocking notice when an outlet can't be reached. A dead feed degrades
  gracefully; the rest of the page stays current.
- **Mini-visualizations** — stat tiles (stories, sources live, past hour,
  publishing pace), a 12-hour publishing sparkline, and a stories-by-source
  bar chart with direct labels.
- **Design system** — validated colorblind-safe categorical palette with a
  fixed color per source (color follows the entity, never its rank), selected
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
