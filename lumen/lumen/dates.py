"""Understanding time in queries: 'summer 2019', 'last Christmas', 'June', 'in the 90s', 'πέρσι'.

parse() takes the date words out of the query and returns a When, which turns into a mask over
photos' capture dates. Times of day ('morning', 'at night') only nudge the ranking, since 'night'
is also something the image model can see.
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from .places import fold

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    # Greek, accent-free (queries are folded): nominative and genitive
    "ιανουαριος": 1, "ιανουαριου": 1, "φεβρουαριος": 2, "φεβρουαριου": 2, "μαρτιος": 3, "μαρτιου": 3,
    "απριλιος": 4, "απριλιου": 4, "μαιος": 5, "μαιου": 5, "ιουνιος": 6, "ιουνιου": 6,
    "ιουλιος": 7, "ιουλιου": 7, "αυγουστος": 8, "αυγουστου": 8, "σεπτεμβριος": 9, "σεπτεμβριου": 9,
    "οκτωβριος": 10, "οκτωβριου": 10, "νοεμβριος": 11, "νοεμβριου": 11, "δεκεμβριος": 12, "δεκεμβριου": 12,
}
SEASONS = {
    "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11), "fall": (9, 10, 11), "winter": (12, 1, 2),
    "ανοιξη": (3, 4, 5), "καλοκαιρι": (6, 7, 8), "φθινοπωρο": (9, 10, 11), "χειμωνας": (12, 1, 2),
    "χειμωνα": (12, 1, 2),
}
# (month, day) windows, possibly wrapping the year end
HOLIDAYS = {
    "christmas": ((12, 20), (1, 6)), "xmas": ((12, 20), (1, 6)), "χριστουγεννα": ((12, 20), (1, 6)),
    "new year's eve": ((12, 31), (1, 1)), "new years eve": ((12, 31), (1, 1)), "new year": ((12, 31), (1, 1)),
    "πρωτοχρονια": ((12, 31), (1, 1)), "halloween": ((10, 31), (10, 31)), "valentine's day": ((2, 14), (2, 14)),
    "valentines": ((2, 14), (2, 14)), "valentine": ((2, 14), (2, 14)),
}
TIMES_OF_DAY = {
    "morning": range(5, 12), "mornings": range(5, 12), "πρωι": range(5, 12),
    "afternoon": range(12, 18), "απογευμα": range(12, 18),
    "evening": range(17, 22), "βραδυ": range(17, 23),
    "night": (21, 22, 23, 0, 1, 2, 3, 4), "nighttime": (21, 22, 23, 0, 1, 2, 3, 4), "νυχτα": (21, 22, 23, 0, 1, 2, 3, 4),
}
FILLER = r"(?:(?:in|on|during|from|at|of|the|this|στο|στα|στη|στην|το|τα|τη|την|του|της)\s+)*"


@dataclass
class When:
    labels: list = field(default_factory=list)
    ranges: list = field(default_factory=list)      # absolute (start ts, end ts) — any may match
    years: set | None = None
    months: set | None = None
    windows: list = field(default_factory=list)     # (month, day) -> (month, day), any year
    hours: set | None = None                        # soft
    weekend: bool = False                           # soft

    def __bool__(self):
        return bool(self.ranges or self.years or self.months or self.windows or self.hours or self.weekend)

    @property
    def is_filter(self):
        return bool(self.ranges or self.years or self.months or self.windows)

    @property
    def label(self):
        return " · ".join(self.labels)


def _period(k, y):
    """(start, end) dates of a season, holiday or month named k in year y."""
    if k in SEASONS:
        months = SEASONS[k]
        if months[0] == 12:  # winter runs into the next year
            return date(y, 12, 1), date(y + 1, 3, 1)
        return date(y, months[0], 1), date(y, months[-1] + 1, 1)
    if k in HOLIDAYS:
        (sm, sd), (em, ed) = HOLIDAYS[k]
        return date(y, sm, sd), date(y + ((em, ed) < (sm, sd)), em, ed) + timedelta(1)
    mo = MONTHS[k]
    return date(y, mo, 1), date(y + (mo == 12), mo % 12 + 1, 1)


def _title(k):
    if k in MONTHS:
        return date(2000, MONTHS[k], 1).strftime("%B")
    if k in ("xmas", "χριστουγεννα"):
        return "Christmas"
    return k[:1].upper() + k[1:]


def _ts(d):
    return datetime(d.year, d.month, d.day).timestamp()


def parse(text: str, today: date | None = None):
    """Returns (query without the date words, When)."""
    today = today or date.today()
    w = When()
    t = fold(text)

    def cut(pattern, fn):
        nonlocal t, text
        while True:
            m = re.search(rf"(?<!\w){FILLER}(?:{pattern})(?!\w)", t)
            if not m:
                return
            fn(m)
            t = t[: m.start()] + " " + t[m.end():]
            text = text[: m.start()] + " " + text[m.end():]

    def add_years(ys, label):
        w.years = (w.years or set()) | set(ys)
        w.labels.append(label)

    # Relative periods
    def rel(m):
        k = m.group("rel")
        if k in ("today", "σημερα"):
            s, e, lab = today, today + timedelta(1), "Today"
        elif k in ("yesterday", "χθες", "χτες"):
            s, e, lab = today - timedelta(1), today, "Yesterday"
        elif k in ("this week",):
            s = today - timedelta(today.weekday())
            e, lab = today + timedelta(1), "This week"
        elif k in ("last week",):
            s = today - timedelta(today.weekday() + 7)
            e, lab = s + timedelta(7), "Last week"
        elif k in ("this month",):
            s, e, lab = today.replace(day=1), today + timedelta(1), "This month"
        elif k in ("last month",):
            e = today.replace(day=1)
            s, lab = (e - timedelta(1)).replace(day=1), "Last month"
        elif k in ("this year", "φετος"):
            s, e, lab = date(today.year, 1, 1), date(today.year + 1, 1, 1), str(today.year)
        else:  # last year / πέρσι
            s, e, lab = date(today.year - 1, 1, 1), date(today.year, 1, 1), str(today.year - 1)
        w.ranges.append((_ts(s), _ts(e)))
        w.labels.append(lab)

    cut(r"(?P<rel>today|yesterday|this week|last week|this month|last month|this year|last year|"
        r"σημερα|χθες|χτες|φετος|περσι)", lambda m: rel(m) if m.group("rel") != "περσι" else
        (w.ranges.append((_ts(date(today.year - 1, 1, 1)), _ts(date(today.year, 1, 1)))),
         w.labels.append(str(today.year - 1))))

    # "last summer", "last christmas": the most recent one that is already over
    def last_season(m):
        k = m.group("s")
        for y in (today.year, today.year - 1, today.year - 2):
            s, e = _period(k, y)
            if e <= today:
                break
        w.ranges.append((_ts(s), _ts(e)))
        w.labels.append(f"{_title(k)} {y}")

    seasons_holidays = "|".join(sorted((re.escape(k) for k in [*SEASONS, *HOLIDAYS]), key=len, reverse=True))
    cut(rf"(?:last|previous|περσινα|περσινο|περσινη)\s+(?P<s>{seasons_holidays})", last_season)

    # "summer 2019", "christmas 2018", "june 2020", "august"
    def season_or_month(m):
        k, y = m.group("k"), m.group("y")
        if y:
            s, e = _period(k, int(y))
            w.ranges.append((_ts(s), _ts(e)))
        elif k in SEASONS:
            w.months = (w.months or set()) | set(SEASONS[k])
        elif k in HOLIDAYS:
            w.windows.append(HOLIDAYS[k])
        else:
            w.months = (w.months or set()) | {MONTHS[k]}
        w.labels.append(f"{_title(k)}{' ' + y if y else ''}")

    names = "|".join(sorted((re.escape(k) for k in [*HOLIDAYS, *SEASONS, *MONTHS]), key=len, reverse=True))
    cut(rf"(?P<k>{names})(?:\s+(?:of\s+|του\s+)?(?P<y>(?:19|20)\d\d))?", season_or_month)

    # Decades and year ranges
    def decade(m):
        d = m.group("d1") or m.group("d2")
        start = int(d) if len(d) == 4 else (1900 + int(d) if int(d) >= 30 else 2000 + int(d))
        add_years(range(start, start + 10), f"The {start}s")

    cut(r"(?:the\s+(?P<d1>\d0)'?s|(?P<d2>(?:19|20)\d0)'?s)", decade)

    def year_range(m):
        a, b = sorted((int(m.group("a")), int(m.group("b"))))
        add_years(range(a, b + 1), f"{a}–{b}")

    cut(r"(?:between\s+)?(?P<a>(?:19|20)\d\d)\s*(?:-|–|to|and|until|εως|μεχρι)\s*(?P<b>(?:19|20)\d\d)", year_range)
    cut(r"(?P<y>(?:19|20)\d\d)", lambda m: add_years([int(m.group("y"))], m.group("y")))

    # Soft hints (the words stay in the query: the image model can see night and mornings too)
    for word, hours in TIMES_OF_DAY.items():
        if re.search(rf"(?<!\w){word}(?!\w)", t):
            w.hours = (w.hours or set()) | set(hours)
            w.labels.append(word.title())
            break
    if re.search(r"(?<!\w)(weekend|weekends|σαββατοκυριακο)(?!\w)", t):
        w.weekend = True
        w.labels.append("Weekend")

    return " ".join(text.split()), w


_cal = {"key": None}


def _calendar(ids, taken):
    """Year/month/day/hour/weekday arrays for photos' capture times (local time)."""
    key = (len(ids), int(ids.sum()) if len(ids) else 0, float(np.nansum(taken)))
    if _cal["key"] != key:
        dts = [datetime.fromtimestamp(t) if t == t else None for t in taken]
        _cal.update(
            key=key,
            year=np.array([d.year if d else 0 for d in dts]),
            month=np.array([d.month if d else 0 for d in dts]),
            md=np.array([d.month * 100 + d.day if d else 0 for d in dts]),
            hour=np.array([d.hour if d else -1 for d in dts]),
            weekday=np.array([d.weekday() if d else -1 for d in dts]),
        )
    return _cal


def mask_and_bonus(when: When, ids, taken):
    """(boolean mask or None, z-score bonus array or None) for the photos `ids` with capture times `taken`."""
    if not when:
        return None, None
    taken = np.asarray(taken, dtype=float)
    cal = _calendar(ids, taken)
    mask = None
    if when.is_filter:
        m = np.ones(len(ids), dtype=bool)
        if when.ranges:
            r = np.zeros(len(ids), dtype=bool)
            for s, e in when.ranges:
                r |= (taken >= s) & (taken < e)
            m &= r
        if when.years:
            m &= np.isin(cal["year"], list(when.years))
        if when.months:
            m &= np.isin(cal["month"], list(when.months))
        if when.windows:
            r = np.zeros(len(ids), dtype=bool)
            for (sm, sd), (em, ed) in when.windows:
                a, b = sm * 100 + sd, em * 100 + ed
                r |= ((cal["md"] >= a) & (cal["md"] <= b)) if a <= b else ((cal["md"] >= a) | (cal["md"] <= b))
            m &= r
        mask = m
    bonus = None
    if when.hours or when.weekend:
        bonus = np.zeros(len(ids))
        if when.hours:
            bonus += np.where(np.isin(cal["hour"], list(when.hours)), 1.0, -0.3)
        if when.weekend:
            bonus += np.where(cal["weekday"] >= 5, 0.8, -0.2)
    return mask, bonus
