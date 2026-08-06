# Newsroom — Live News Dashboard

A self-contained, zero-dependency live news dashboard. One HTML file, no build
step, no API keys. It aggregates 50+ public RSS/Atom feeds from thirty outlets
across nine regions and presents them with dashboard UI/UX principles: a KPI
summary row, a single filter row, a lead story, a card grid, a stories-by-source
chart, a live wire — and a **Connections** view that maps how separate stories
relate to one another.

| Region | Outlets |
| --- | --- |
| Global | BBC News, The Guardian, The New York Times, Al Jazeera |
| Europe | Sky News, DW, France 24, El País (EN) |
| Greece | News247, CNN Greece, in.gr, Kathimerini (EN), Greek Reporter, Keep Talking Greece, NewsNow Greece |
| Americas | NPR, CNBC, ESPN, CBC |
| Asia | Japan Times, SCMP, Times of India |
| Middle East | Times of Israel |
| Africa | Africanews |
| Oceania | ABC Australia |
| Research | arXiv (cs.AI / cs.LG), MIT News, ScienceDaily, Nature, MIT Technology Review |

Categories span World, Greece, Business, Technology, Science, AI Research,
Sports and Health, and a region filter narrows everything to one part of the
world at a time.

News247, CNN Greece and in.gr publish in Greek. The tokenizer, stopword list
and entity extractor are Unicode-aware and fold Greek accents and final sigma,
so Greek headlines cluster, trend and enter the connection graph exactly like
English ones rather than silently dropping out.

Several of these publishers document no feed URL, and feed paths move. Each
feed entry may therefore carry a **list of candidate URLs**; they are tried in
order and whichever one parses is pinned for the session. A source whose
candidates all fail degrades to an `!` on its chip and a notice — the rest of
the page stays current.

The last candidate for the Greek outlets (and Times of Israel) is a Google News
per-site search feed. Measured against the live web: `in.gr` and Keep Talking
Greece serve their native feeds fine; `news247.gr`, `greekreporter.com`,
`ekathimerini.com` and `timesofisrael.com` answer **403** to anything that is
not a browser, which includes the CORS relays and the deploy runner; `cnn.gr`
returns **404** at every conventional feed path. The Google News fallback keeps
those outlets contributing their own stories under their own chip — its items
redirect to the original publisher. Native feeds are always tried first, so a
publisher that unblocks (or starts publishing a feed) is picked up
automatically with no code change. The server-side generator also sends a
conventional browser user-agent, which clears some of the 403s on its own.

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
- **Connections tab** — see below.

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

- **Hidden connections** — the same call asks Claude for 3–6 links between
  *different* stories that term-overlap arithmetic cannot see: a shared
  upstream cause, one story being the downstream consequence of another, the
  same pressure surfacing in unrelated industries, two outlets framing the same
  facts revealingly differently, or a story the Greek outlets cover very
  differently from the English-language ones. Each carries a kind, a 1–5
  surprise score and the stories it was drawn from. They head the Connections
  tab, labelled as leads to check rather than findings.

## The Connections tab

Clustering answers *"which stories are the same story"*. This tab answers the
different question: which **separate** stories are quietly related — and which
of those relationships cross beats you would never expect to be linked.

It builds a graph over the most important clusters from the last 48 hours:

- **Nodes** are story clusters, sized by how many outlets are covering them and
  coloured by source.
- **Edges** are shared proper nouns (people, places, organizations, products)
  and shared distinctive vocabulary. Terms are weighted by inverse document
  frequency, so a word half the graph uses contributes nothing while a term
  shared by exactly two stories counts heavily; proper nouns are weighted
  higher than plain words. Pairs the clusterer nearly merged are excluded —
  an edge always means two genuinely different stories.
- **Dashed accent edges cross domain boundaries.** Categories are grouped
  (geo / econ / tech / sci / sport); a Business↔Technology link is ordinary and
  a Sports↔Science one is not, and the surprise score reflects that distance,
  the strength of the shared ground, and whether the link is carried by a named
  actor rather than loose vocabulary.

Layout is a force simulation run to completion once and then scaled to fill the
canvas — deterministic, no animation, and it does not reshuffle under your
cursor when the minute-tick re-render fires. Click any node to focus it: the
graph dims everything unattached and the side panel lists each connection with
the terms that produced it. Everything is keyboard-reachable.

Below the graph:

- **Surprising connections** — the top cross-domain pairs as cards, each naming
  both stories, their beats and the shared ground. Clicking a card's header
  jumps to that story in the graph.
- **Bridge terms** — single terms appearing across the most distinct beats, the
  vocabulary stitching unrelated sections together. Multi-word entities outrank
  and absorb their component words, so "taiwan semiconductor" does not appear
  three times.
- **Where the links run** — a category × category matrix of link counts. The
  off-diagonal cells are where a story in one section turns out to share ground
  with another.

The whole graph is computed in the browser in a couple of milliseconds; the
node count, time window, edge threshold and per-node degree cap are the
`GRAPH_*` constants at the top of the script.

To enable the Claude features, add an `ANTHROPIC_API_KEY` repository secret
(Settings → Secrets and variables → Actions). Without the key the workflow
still deploys cleanly and the dashboard simply hides the briefing panel.
Rough cost at current pricing: a few cents per generation, ~8 generations/day
on the default 3-hour schedule. Adjust the `cron` line in
`.github/workflows/deploy-pages.yml` to change the cadence.

## Features

- **Live aggregation** — 50+ RSS/Atom feeds fetched through a bounded
  concurrency pool (10 at a time, to stay friendly to the CORS relays),
  deduplicated and merged into one reverse-chronological stream. arXiv's
  daily paper batches are capped per feed so they can't drown the stream.
- **Auto-refresh** — every 5 minutes by default (configurable: 2/5/10 min or
  manual). Refreshing pauses while the tab is hidden and catches up on return;
  relative timestamps ("4m ago") tick every minute without refetching.
- **Filtering** — category tabs (Top, For You, Connections, World, Greece,
  Business, Technology, Science, AI Research, Sports, Health), a region filter,
  a collapsible source picker grouped by region, and instant headline search.
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
(name + region + optional palette slot) and the `FEEDS` list (source, category,
URL or list of candidate URLs) at the top of the script. Mirror the same change
in `scripts/generate-briefing.mjs`, which keeps its own copy of the feed list
for the server-side Claude pass.

All headlines link to, and remain the property of, their original publishers.
