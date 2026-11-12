"""Score Lumen on your own searches, so changes can be judged by numbers rather than by feel.

Write a text file with one search per line and the photo(s) you'd want to see first:

    # my-searches.txt
    a dog on the beach => IMG_2887.jpeg
    golden hour => R0005726.JPG, IMG_3657.jpeg
    σκύλος στην παραλία => IMG_2887.jpeg

With Lumen running and your photos indexed:

    python tools/score_searches.py my-searches.txt
    python tools/score_searches.py my-searches.txt --deep          # also with the closer look
    python tools/score_searches.py my-searches.txt --compare scores-before.json

For each search it reports where the best of your expected photos landed (1 = first). The
summary gives how often it was first, in the top 5 and top 10, and the mean reciprocal rank
(1.0 = always first). Results are saved to a scores-*.json file to compare against later.
Needs nothing but Python; it only talks to the running app.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def call(base, path, body=None, params=None):
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("detail")
        except Exception:
            detail = e.reason
        raise SystemExit(f"Lumen said no ({e.code}): {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Can't reach Lumen at {base} ({e.reason}). Is it running?")


def read_cases(path):
    cases = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" not in line:
            raise SystemExit(f"{path} line {n}: expected 'search => filename[, filename]'")
        q, names = line.split("=>", 1)
        names = [x.strip() for x in names.split(",") if x.strip()]
        if not q.strip() or not names:
            raise SystemExit(f"{path} line {n}: expected 'search => filename[, filename]'")
        cases.append((q.strip(), names))
    if not cases:
        raise SystemExit(f"{path} has no searches in it")
    return cases


def rank_of(items, names):
    """1-based position of the best expected photo, or None if none of them came back."""
    want = {n.lower() for n in names}
    for i, it in enumerate(items, 1):
        if it["name"].lower() in want:
            return i
    return None


def look_closer(base, q, items, top):
    job = call(base, "/api/search/rerank", {"q": q, "ids": [i["id"] for i in items[:top]]})
    while True:
        j = call(base, f"/api/jobs/{job['job_id']}")
        if j["state"] == "failed":
            raise SystemExit(f"Closer look failed: {j['error']}")
        if j["state"] == "done":
            return
        time.sleep(0.5)


def summary(ranks):
    n = len(ranks)
    return {
        "first": sum(r == 1 for r in ranks) / n,
        "top5": sum(r is not None and r <= 5 for r in ranks) / n,
        "top10": sum(r is not None and r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--deep", action="store_true", help="also score with the closer look (must be on in Settings)")
    ap.add_argument("--compare", help="an earlier scores-*.json to compare against")
    ap.add_argument("--out", help="where to save the results (default scores-<time>.json)")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    cases = read_cases(a.file)
    status = call(base, "/api/status")
    settings = call(base, "/api/settings")
    print(f"{status['ready']} photos indexed, model: {settings['model']}"
          + (f" ({status['pending']} still waiting to be indexed)" if status["pending"] else ""))
    top = settings.get("rerank", {}).get("top", 12)
    if a.deep and not call(base, "/api/rerank")["available"]:
        raise SystemExit("The closer look isn't ready: " + call(base, "/api/rerank")["detail"])

    rows = []
    for q, names in cases:
        items = call(base, "/api/search", params={"q": q, "limit": 1000})["items"]
        row = {"query": q, "expected": names, "rank": rank_of(items, names)}
        if a.deep:
            t0 = time.time()
            look_closer(base, q, items, top)
            row["deep_seconds"] = round(time.time() - t0, 1)
            row["deep_rank"] = rank_of(call(base, "/api/search", params={"q": q, "limit": 1000, "deep": "true"})["items"], names)
        rows.append(row)

    previous = {}
    if a.compare:
        previous = {r["query"]: r for r in json.load(open(a.compare))["rows"]}
    width = min(48, max(len(r["query"]) for r in rows))
    head = f"{'search':{width}}  rank" + ("  closer" if a.deep else "") + ("  before" if previous else "")
    print("\n" + head + "\n" + "-" * len(head))
    show = lambda r: "  — " if r is None else f"{r:4d}"   # noqa: E731
    for r in rows:
        line = f"{r['query'][:width]:{width}}  {show(r['rank'])}"
        if a.deep:
            line += f"    {show(r['deep_rank'])}"
        if previous:
            old = previous.get(r["query"])
            line += f"    {show(old['rank']) if old else '   ?'}"
        print(line)
    print("\n(rank = position of your expected photo; — = not in the results at all, "
          "or the filename doesn't match an indexed photo)")

    out = {"file": a.file, "model": settings["model"], "photos": status["ready"], "rows": rows,
           "summary": summary([r["rank"] for r in rows])}
    if a.deep:
        out["deep_summary"] = summary([r["deep_rank"] for r in rows])
        out["deep_seconds_per_search"] = round(sum(r["deep_seconds"] for r in rows) / len(rows), 1)

    def line(label, s):
        print(f"{label:14} first {s['first']:4.0%}   top 5 {s['top5']:4.0%}   top 10 {s['top10']:4.0%}   MRR {s['mrr']:.3f}")

    print()
    line("Lumen", out["summary"])
    if a.deep:
        line("+ closer look", out["deep_summary"])
        print(f"{'':14} closer look took {out['deep_seconds_per_search']} s per search on average (first time; repeats are instant)")
    if previous:
        both = [r for r in rows if r["query"] in previous]
        if both:
            line("before", summary([previous[r["query"]]["rank"] for r in both]))
    path = a.out or time.strftime("scores-%Y%m%d-%H%M%S.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"\nSaved to {path}")
    if len(rows) < 15:
        print(f"Note: {len(rows)} searches is a small sample. One search moving changes 'first' by {1 / len(rows):.0%}; "
              "aim for 20-30 before trusting small differences.")


if __name__ == "__main__":
    sys.exit(main())
