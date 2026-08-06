// Generates news-dashboard/briefing.json for the dashboard's AI panel.
// Runs in the Pages deploy workflow with ANTHROPIC_API_KEY from repo secrets;
// on any failure (or missing key) it writes an "unavailable" stub and exits 0
// so the site deploy never breaks.
//
// Feeds mirror the FEEDS list in ../index.html — keep the two in sync.

import { writeFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT_PATH = join(dirname(fileURLToPath(import.meta.url)), "..", "briefing.json");

/* Last-resort candidate — see the matching comment in ../index.html. Google
   News' per-site search feed links back to the original outlet, so a publisher
   whose own feed is bot-blocked or absent still contributes its own stories. */
const GNEWS = (q, hl, gl) =>
  `https://news.google.com/rss/search?q=${encodeURIComponent(q + " when:2d")}&hl=${hl}&gl=${gl}&ceid=${gl}:${hl.split("-")[0]}`;
const GNEWS_EL = (q) => GNEWS(q, "el", "GR");
const GNEWS_EN = (q) => GNEWS(q, "en-US", "US");

const FEEDS = [
  { src: "BBC News",        cat: "World",       url: "https://feeds.bbci.co.uk/news/world/rss.xml" },
  { src: "BBC News",        cat: "Business",    url: "https://feeds.bbci.co.uk/news/business/rss.xml" },
  { src: "BBC News",        cat: "Technology",  url: "https://feeds.bbci.co.uk/news/technology/rss.xml" },
  { src: "BBC News",        cat: "Science",     url: "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml" },
  { src: "BBC News",        cat: "Health",      url: "https://feeds.bbci.co.uk/news/health/rss.xml" },
  { src: "BBC News",        cat: "Sports",      url: "https://feeds.bbci.co.uk/sport/rss.xml" },
  { src: "Guardian",        cat: "World",       url: "https://www.theguardian.com/world/rss" },
  { src: "Guardian",        cat: "Sports",      url: "https://www.theguardian.com/uk/sport/rss" },
  { src: "Guardian",        cat: "Business",    url: "https://www.theguardian.com/uk/business/rss" },
  { src: "Guardian",        cat: "Technology",  url: "https://www.theguardian.com/uk/technology/rss" },
  { src: "Guardian",        cat: "Science",     url: "https://www.theguardian.com/science/rss" },
  { src: "NY Times",        cat: "World",       url: "https://rss.nytimes.com/services/xml/rss/nyt/World.xml" },
  { src: "NY Times",        cat: "Business",    url: "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml" },
  { src: "NY Times",        cat: "Technology",  url: "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml" },
  { src: "NY Times",        cat: "Science",     url: "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml" },
  { src: "NY Times",        cat: "Health",      url: "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml" },
  { src: "NY Times",        cat: "Sports",      url: "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml" },
  { src: "Al Jazeera",      cat: "World",       url: "https://www.aljazeera.com/xml/rss/all.xml" },
  { src: "NPR",             cat: "World",       url: "https://feeds.npr.org/1004/rss.xml" },
  { src: "NPR",             cat: "Business",    url: "https://feeds.npr.org/1006/rss.xml" },
  { src: "NPR",             cat: "Science",     url: "https://feeds.npr.org/1007/rss.xml" },
  { src: "NPR",             cat: "Health",      url: "https://feeds.npr.org/1128/rss.xml" },
  { src: "Sky News",        cat: "World",       url: "https://feeds.skynews.com/feeds/rss/world.xml" },
  { src: "Sky News",        cat: "Business",    url: "https://feeds.skynews.com/feeds/rss/business.xml" },
  { src: "Sky News",        cat: "Technology",  url: "https://feeds.skynews.com/feeds/rss/technology.xml" },
  { src: "ESPN",            cat: "Sports",      url: "https://www.espn.com/espn/rss/news" },
  { src: "DW",              cat: "World",       url: "https://rss.dw.com/rdf/rss-en-all" },
  { src: "France 24",       cat: "World",       url: "https://www.france24.com/en/rss" },
  { src: "CNBC",            cat: "Business",    url: "https://www.cnbc.com/id/10001147/device/rss/rss.html" },
  { src: "MIT Tech Review", cat: "Technology",  url: "https://www.technologyreview.com/feed/" },
  { src: "Nature",          cat: "Science",     url: "https://www.nature.com/nature.rss" },
  { src: "ScienceDaily",    cat: "Science",     url: "https://www.sciencedaily.com/rss/top/science.xml" },
  { src: "arXiv",           cat: "AI Research", url: "https://rss.arxiv.org/rss/cs.AI", abstracts: true, max: 12 },
  { src: "arXiv",           cat: "AI Research", url: "https://rss.arxiv.org/rss/cs.LG", abstracts: true, max: 12 },
  { src: "MIT News",        cat: "AI Research", url: "https://news.mit.edu/rss/topic/artificial-intelligence2" },
  { src: "ScienceDaily",    cat: "AI Research", url: "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml" },

  /* Greece. `url` may be a list of candidates — same fallback behaviour as the
     browser side; the first one that yields items wins. */
  { src: "News247",            cat: "Greece", lang: "el", url: ["https://www.news247.gr/feed/", "https://www.news247.gr/rss", "https://www.news247.gr/feed", GNEWS_EL("site:news247.gr")] },
  { src: "CNN Greece",         cat: "Greece", lang: "el", url: ["https://www.cnn.gr/rss", "https://www.cnn.gr/feed", "https://www.cnn.gr/rss.xml", "https://www.cnn.gr/news/rss", GNEWS_EL("site:cnn.gr")] },
  { src: "in.gr",              cat: "Greece", lang: "el", url: ["https://www.in.gr/feed/", "https://www.in.gr/rss", "https://www.in.gr/feed", GNEWS_EL("site:in.gr")] },
  { src: "Greek Reporter",     cat: "Greece", url: ["https://greekreporter.com/feed/", "https://greekreporter.com/rss", GNEWS_EN("site:greekreporter.com")] },
  { src: "Keep Talking Greece",cat: "Greece", url: ["https://www.keeptalkinggreece.com/feed/", GNEWS_EN("site:keeptalkinggreece.com")] },
  { src: "NewsNow Greece",     cat: "Greece", url: ["https://www.newsnow.co.uk/h/World+News/Europe/Southern+Europe/Greece?type=rss", "https://www.newsnow.co.uk/h/World+News/Europe/Southern+Europe/Greece/rss", GNEWS_EN("Greece")] },
  { src: "Kathimerini EN",     cat: "Greece", url: ["https://www.ekathimerini.com/feed/", "https://www.ekathimerini.com/rss", "https://www.ekathimerini.com/news/feed/", GNEWS_EN("site:ekathimerini.com")] },

  /* Other countries */
  { src: "El País EN",      cat: "World",       url: ["https://feeds.elpais.com/mrss-s/pages/ep/site/english.elpais.com/portada", "https://english.elpais.com/rss/"] },
  { src: "Japan Times",     cat: "World",       url: ["https://www.japantimes.co.jp/feed/", "https://www.japantimes.co.jp/news/feed/"] },
  { src: "SCMP",            cat: "World",       url: ["https://www.scmp.com/rss/91/feed", "https://www.scmp.com/rss/5/feed"] },
  { src: "Times of India",  cat: "World",       url: ["https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"] },
  { src: "Times of Israel", cat: "World",       url: ["https://www.timesofisrael.com/feed/", GNEWS_EN("site:timesofisrael.com")] },
  { src: "Africanews",      cat: "World",       url: ["https://www.africanews.com/feed/rss", "https://www.africanews.com/api/en/rss"] },
  { src: "ABC Australia",   cat: "World",       url: ["https://www.abc.net.au/news/feed/51120/rss.xml", "https://www.abc.net.au/news/feed/2942460/rss.xml"] },
  { src: "CBC",             cat: "World",       url: ["https://www.cbc.ca/webfeed/rss/rss-world", "https://www.cbc.ca/cmlink/rss-world"] },
];

const decode = (s) =>
  s.replace(/<!\[CDATA\[(.*?)\]\]>/gs, "$1")
   .replace(/<[^>]+>/g, " ")
   .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
   .replace(/&quot;/g, '"').replace(/&#0?39;|&apos;/g, "'")
   .replace(/\s+/g, " ").trim();

function parseItems(xml, feed) {
  const items = [];
  const blocks = xml.match(/<(?:item|entry)[\s>][\s\S]*?<\/(?:item|entry)>/g) || [];
  for (const b of blocks.slice(0, feed.max || 15)) {
    const pick = (tag) => {
      const m = b.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`));
      return m ? decode(m[1]) : "";
    };
    const linkAttr = b.match(/<link[^>]*href="([^"]+)"/);
    const item = {
      source: feed.src,
      category: feed.cat,
      ...(feed.lang ? { language: feed.lang } : {}),
      title: pick("title"),
      link: pick("link") || (linkAttr ? linkAttr[1] : ""),
      published: pick("pubDate") || pick("dc:date") || pick("published") || pick("updated"),
    };
    if (feed.abstracts) item.abstract = pick("description").slice(0, 700);
    if (item.title && item.link) items.push(item);
  }
  return items;
}

async function fetchOne(url, feed) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      // Several publishers (notably the Cloudflare-fronted Greek ones) answer
      // 403 to a bare tool user-agent from a datacenter IP. A conventional
      // browser UA plus a feed Accept header is what every feed reader sends
      // and is what these hosts expect.
      headers: {
        "user-agent":
          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) " +
          "Chrome/124.0.0.0 Safari/537.36 newsroom-dashboard-briefing/1.0",
        accept: "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        "accept-language": "en,el;q=0.8",
      },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return parseItems(await res.text(), feed);
  } finally {
    clearTimeout(t);
  }
}

/* Google News titles carry a " - Publisher" suffix; strip it so the model sees
   the headline as the outlet wrote it. */
const tidyGoogleNews = (items, url) =>
  !url.includes("news.google.com") ? items
    : items.map((i) => ({ ...i, title: i.title.replace(/\s+-\s+[^-]{2,40}$/, "").trim() || i.title }));

async function fetchFeed(feed) {
  for (const url of Array.isArray(feed.url) ? feed.url : [feed.url]) {
    try {
      const items = tidyGoogleNews(await fetchOne(url, feed), url);
      if (items.length) return items;
      console.error(`feed empty: ${url}`);
    } catch (e) {
      console.error(`feed failed: ${url} (${e.message})`);
    }
  }
  return [];
}

const BRIEFING_SCHEMA = {
  type: "object",
  properties: {
    briefing: {
      type: "array",
      description: "The 5 most important stories right now, synthesized across outlets",
      items: {
        type: "object",
        properties: {
          headline: { type: "string", description: "Short neutral headline, max 12 words" },
          detail: { type: "string", description: "Two sentences: what happened and why it matters" },
          link: { type: "string", description: "The exact link (copied from the input) of the single input story this briefing item is primarily based on" },
          context: {
            type: ["string", "null"],
            description:
              "2-3 sentences of background the article text itself would not tell you and that you " +
              "are confident is accurate from general knowledge: who a named person/organization is " +
              "and relevant history, what a technical or legal term means, precedent for a similar " +
              "past event, or whether a comparative claim (e.g. 'worst since 2008') checks out. " +
              "Never speculate or invent specifics you are not confident about — output null if you " +
              "have nothing solid to add beyond what the headline and detail already say.",
          },
        },
        required: ["headline", "detail", "link", "context"],
        additionalProperties: false,
      },
    },
    signals: {
      type: "array",
      description: "2-4 non-obvious patterns connecting seemingly unrelated stories",
      items: {
        type: "object",
        properties: {
          title: { type: "string", description: "Name of the pattern, max 8 words" },
          insight: { type: "string", description: "Two or three sentences naming the specific stories that form the pattern and what it suggests" },
        },
        required: ["title", "insight"],
        additionalProperties: false,
      },
    },
    connections: {
      type: "array",
      description:
        "3-6 hidden connections between stories that are NOT about the same event and that a " +
        "reader scanning the feed would not notice. The dashboard already detects links through " +
        "shared names and shared vocabulary, so those are worthless here — find the links that " +
        "only knowledge and reasoning reveal: a shared upstream cause, one story being the " +
        "downstream consequence of another, the same pressure showing up in two unrelated " +
        "sectors, a pattern in how differently outlets frame the same facts, or the same event " +
        "covered in Greek and in English with materially different emphasis.",
      items: {
        type: "object",
        properties: {
          title: { type: "string", description: "Name the connection, max 9 words" },
          insight: {
            type: "string",
            description:
              "Three or four sentences: what the link actually is, why it is not obvious, and " +
              "what it would mean if it holds. Name the stories concretely.",
          },
          kind: {
            type: "string",
            enum: ["shared-cause", "consequence", "same-pressure", "framing-contrast", "cross-language", "other"],
            description: "What kind of link this is",
          },
          surprise: {
            type: "integer",
            description: "1 = an informed reader would already expect this; 5 = genuinely unexpected",
          },
          stories: {
            type: "array",
            description: "The 2-4 input stories this connection is drawn from",
            items: {
              type: "object",
              properties: {
                headline: { type: "string", description: "The story's headline, copied from the input" },
                link: { type: "string", description: "The story's link, copied exactly from the input" },
              },
              required: ["headline", "link"],
              additionalProperties: false,
            },
          },
        },
        required: ["title", "insight", "kind", "surprise", "stories"],
        additionalProperties: false,
      },
    },
    arxiv: {
      type: "array",
      description: "One entry per arXiv paper provided in the input",
      items: {
        type: "object",
        properties: {
          link: { type: "string", description: "The paper's link, copied exactly from the input" },
          tldr: { type: "string", description: "Plain-English: what the paper shows and why it matters, max 30 words, no jargon" },
        },
        required: ["link", "tldr"],
        additionalProperties: false,
      },
    },
  },
  required: ["briefing", "signals", "connections", "arxiv"],
  additionalProperties: false,
};

async function main() {
  const stub = (reason) => ({ generatedAt: new Date().toISOString(), unavailable: true, reason });

  if (!process.env.ANTHROPIC_API_KEY) {
    await write(stub("no ANTHROPIC_API_KEY configured"));
    console.log("No API key — wrote unavailable stub.");
    return;
  }

  const results = await Promise.all(FEEDS.map(fetchFeed));
  const items = results.flat();
  const news = items.filter((i) => !i.abstract);
  const papers = items.filter((i) => i.abstract);
  console.log(`Fetched ${news.length} headlines, ${papers.length} papers.`);
  if (news.length < 20) {
    await write(stub("too few headlines fetched"));
    return;
  }

  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic();

  const response = await client.beta.messages.create({
    model: "claude-opus-5",
    max_tokens: 16000,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { format: { type: "json_schema", schema: BRIEFING_SCHEMA } },
    system:
      "You are the analysis engine behind a live news dashboard. You receive the current " +
      "headlines from around thirty outlets across several countries — including Greek-language " +
      "outlets whose items are in Greek — plus today's arXiv AI papers, and produce five things: " +
      "a briefing of the five most important stories (judge importance by cross-outlet coverage " +
      "and real-world consequence, not recency alone; synthesize across outlets rather than " +
      "echoing one; for each, pick the single input story it is best matched to and copy that " +
      "story's link exactly), background context for each briefing story drawn from your own " +
      "knowledge rather than the fetched text — who a named figure or organization is and their " +
      "relevant track record, what a technical/legal/financial term actually means, precedent " +
      "from a genuinely similar past event, or whether a comparative claim in the story holds up " +
      "(err toward omitting a context note entirely, via null, rather than stating anything you " +
      "are not confident is factually correct), a set of signals — non-obvious patterns that " +
      "connect seemingly unrelated stories (a supply-chain thread behind separate business " +
      "stories, a policy shift visible across regions, a technology quietly appearing in several " +
      "fields; only report patterns genuinely supported by the given stories, never invent " +
      "connections), a set of hidden connections between individual stories, and a plain-English " +
      "TL;DR for every arXiv paper provided. " +
      "For the connections specifically: the dashboard already computes links from shared proper " +
      "nouns and shared vocabulary and shows them as a graph, so repeating those adds nothing. " +
      "Look instead for links that require knowing something — a common upstream cause behind two " +
      "stories filed under different sections, one story being a foreseeable consequence of " +
      "another, the same economic or regulatory pressure surfacing in unrelated industries, two " +
      "outlets reporting the same facts with revealingly different emphasis, or a story the Greek " +
      "outlets cover very differently from the English-language ones. Read the Greek items as " +
      "carefully as the English ones and write every output in English. Score each connection's " +
      "surprise honestly: if an informed reader would already expect the link, score it low or " +
      "leave it out. Never manufacture a connection to fill the list — returning three solid ones " +
      "is better than six thin ones. Be neutral and specific; name the stories a signal or " +
      "connection draws on and copy their links exactly.",
    messages: [
      {
        role: "user",
        content:
          `Current headlines (fetched ${new Date().toUTCString()}):\n` +
          JSON.stringify(news, null, 1) +
          `\n\narXiv papers (write one tldr per paper, copy each link exactly):\n` +
          JSON.stringify(papers, null, 1),
      },
    ],
  });

  if (response.stop_reason === "refusal") {
    await write(stub("model declined the request"));
    console.error("Refusal:", response.stop_details);
    return;
  }

  const text = response.content.find((b) => b.type === "text")?.text;
  const data = JSON.parse(text);
  await write({
    generatedAt: new Date().toISOString(),
    model: response.model,
    headlineCount: news.length,
    paperCount: papers.length,
    ...data,
  });
  const withContext = data.briefing.filter((b) => b.context).length;
  const conns = data.connections || [];
  console.log(
    `Wrote briefing.json: ${data.briefing.length} briefing items (${withContext} with background ` +
    `context), ${data.signals.length} signals, ${conns.length} hidden connections ` +
    `(surprise ${conns.map((c) => c.surprise).join("/") || "–"}), ${data.arxiv.length} paper TLDRs ` +
    `(${response.usage.input_tokens} in / ${response.usage.output_tokens} out tokens).`
  );
}

async function write(obj) {
  await mkdir(dirname(OUT_PATH), { recursive: true });
  await writeFile(OUT_PATH, JSON.stringify(obj, null, 2));
}

main().catch(async (e) => {
  console.error("Briefing generation failed:", e);
  await write({ generatedAt: new Date().toISOString(), unavailable: true, reason: String(e) });
});
