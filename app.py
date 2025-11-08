import os, sqlite3, datetime
from math import ceil
import streamlit as st

DB_PATH = os.path.join("data", "papers.db")
BREAKDOWN_MAX = {"Novelty": 3, "Impact": 4, "Results": 2, "Access": 1}
CARDS_PER_PAGE = 15
_CARD_CSS_FLAG = "_card_css"

# --- Quantized score slider + breakdown chips ---
def _score_color(s: int) -> str:
    if s >= 9: return "#0ea5e9"  # bright blue for unicorns
    if s >= 8: return "#16a34a"
    if s >= 7: return "#22c55e"
    if s >= 6: return "#84cc16"
    if s >= 5: return "#eab308"
    if s >= 4: return "#f59e0b"
    if s >= 3: return "#f97316"
    return "#ef4444"

def inject_slider_css():
    if not getattr(st.session_state, "_qslider_css", False):
        st.markdown("""
<style>
.qslider { position: relative; height: 32px; margin-top: 6px; }
.qslider-track {
  position: absolute; top: 13px; left: 0; right: 0; height: 6px;
  background: repeating-linear-gradient(to right, #e5e7eb, #e5e7eb 9%, #fff 9%, #fff 10%);
  border-radius: 9999px; box-shadow: inset 0 0 0 1px #e5e7eb;
}
.qslider-fill {
  position: absolute; top: 13px; left: 0; height: 6px; border-radius: 9999px;
}
.qslider-thumb {
  position: absolute; top: 7px; width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid white; box-shadow: 0 1px 2px rgba(0,0,0,.15);
  background: currentColor;
}
.qslider-ticks {
  position: absolute; top: 22px; left: 0; right: 0; height: 10px;
  display: flex; justify-content: space-between; pointer-events: none;
}
.qslider-ticks span { width: 2px; height: 6px; background: #cbd5e1; display: inline-block; }
.qslider-label {
  font-size: 12px; color: #475569; margin-top: 4px;
}
.qchip { background:#f3f4f6; color:#374151; padding:2px 8px; border-radius:9999px; margin-right:6px; display:inline-block;}
</style>
        """, unsafe_allow_html=True)
        st.session_state._qslider_css = True

def render_quant_slider(score: int):
    score = max(0, min(10, int(score or 0)))
    pct = f"{score*10}%"
    color = _score_color(score) if score else "#cbd5e1"
    inject_slider_css()
    ticks = "".join("<span></span>" for _ in range(10))
    html = f"""
<div class="qslider" role="img" aria-label="Score {score} out of 10">
  <div class="qslider-track"></div>
  <div class="qslider-fill" style="width:{pct}; background:{color};"></div>
  <div class="qslider-ticks">{ticks}</div>
  <div class="qslider-thumb" style="left: calc({pct} - 9px); color:{color};"></div>
</div>
<div class="qslider-label">{score if score else '—'}/10</div>
"""
    st.markdown(html, unsafe_allow_html=True)

def parse_breakdown(breakdown: str):
    parts = {}
    for kv in (breakdown or "").split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            parts[k.strip()] = v.strip()
    return parts


def inject_card_css():
    if getattr(st.session_state, _CARD_CSS_FLAG, False):
        return
    st.markdown(
        """
<style>
  .score-badge {
    display:inline-block;
    padding:2px 8px;
    border:1px solid #1f2937;
    border-radius:8px;
    font-weight:600;
    color:#e2e8f0;
  }
  .card-meta {
    color:#94a3b8;
    font-size:12.5px;
    margin-top:-6px;
  }
  .score-block {
    margin:10px 0 6px 0;
  }
  .score-block strong {
    font-size:15px;
  }
  .score-breakdown-label {
    color:#94a3b8;
    font-size:12.5px;
    line-height:1.2;
    margin:6px 0 12px 0;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CARD_CSS_FLAG] = True


def _format_timestamp(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return ts

@st.cache_data(ttl=3600, show_spinner=False)
def load_rows(search="", cats=None, only_summarized=False, min_score=0, only_scored=False, sort="newest", limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    sql = """
    SELECT
      id,
      COALESCE(arxiv_id, '') AS arxiv_id,
      title,
      authors,
      date,
      COALESCE(reasoning_category, '') AS reasoning_category,
      arxiv_link,
      COALESCE(tldr, '')       AS tldr,
      COALESCE(summary_md, '') AS summary_md,
      COALESCE(excitement_score, 0) AS excitement_score,
      COALESCE(excitement_reasoning, '') AS excitement_reasoning,
      COALESCE(score_breakdown, '') AS score_breakdown,
      COALESCE(last_scored_at, '') AS last_scored_at
    FROM papers
    WHERE 1=1
    """
    params = []

    # Category filter
    if cats:
        placeholders = ",".join(["?"] * len(cats))
        sql += f" AND reasoning_category IN ({placeholders})"
        params += list(cats)

    # Search across title/abstract/keywords + tldr/summary
    if search:
        like = f"%{search}%"
        # NOTE: this references 'abstract' and 'keywords' assuming they exist in your table.
        # If not, remove them from the WHERE below.
        sql += " AND (title LIKE ? OR abstract LIKE ? OR keywords LIKE ? OR tldr LIKE ? OR summary_md LIKE ?)"
        params += [like, like, like, like, like]

    # Only items that already have a TL;DR
    if only_summarized:
        sql += " AND tldr <> ''"

    if only_scored:
        sql += " AND COALESCE(excitement_score, 0) > 0"

    if min_score:
        sql += " AND COALESCE(excitement_score, 0) >= ?"
        params.append(min_score)

    order_clause = "date DESC"
    if sort == "score":
        order_clause = "COALESCE(excitement_score,0) DESC, date DESC"

    sql += f" ORDER BY {order_clause} LIMIT ?"
    params.append(limit)

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows

# ---------- UI ----------
st.set_page_config(page_title="Reasoning Hub", layout="wide")
st.title("Reasoning Hub")

with st.sidebar:
    st.subheader("Filters")
    search = st.text_input("Search")

    # Category list (hide blanks)
    conn = sqlite3.connect(DB_PATH)
    cats_all = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT reasoning_category FROM papers "
            "WHERE reasoning_category IS NOT NULL AND reasoning_category <> '' "
            "ORDER BY reasoning_category"
        )
    ]
    conn.close()

    sel = st.multiselect("Filter by category", cats_all)
    only_summarized = st.checkbox("Summarized", value=False)
    limit = st.slider("Max results", min_value=25, max_value=500, value=200, step=25)

    min_score = st.slider("Min excitement score", min_value=0, max_value=10, value=0, step=1)
    only_scored = st.checkbox("Only show scored papers", value=False)
    sort = st.radio("Sort by", options=["Newest", "Score"], horizontal=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh"):
            st.cache_data.clear()

if "page" not in st.session_state:
    st.session_state.page = 0
sort_key = "score" if sort == "Score" else "newest"
rows = load_rows(
    search=search,
    cats=sel,
    only_summarized=only_summarized,
    min_score=min_score,
    only_scored=only_scored,
    sort=sort_key,
    limit=limit,
)
current_signature = (search, tuple(sel), only_summarized, min_score, only_scored, sort_key, limit)
if st.session_state.get("_last_filter_signature") != current_signature:
    st.session_state.page = 0
    st.session_state._last_filter_signature = current_signature

total_pages = max(1, ceil(len(rows) / CARDS_PER_PAGE)) if rows else 1
st.session_state.page = max(0, min(st.session_state.page, total_pages - 1))
start = st.session_state.page * CARDS_PER_PAGE
end = start + CARDS_PER_PAGE
page_rows = rows[start:end]

st.caption(f"{len(rows)} results")
if rows:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        disabled = st.session_state.page <= 0
        if st.button("◀ Prev", key="prev_page", disabled=disabled):
            st.session_state.page -= 1
            st.rerun()
    with col3:
        disabled = st.session_state.page >= total_pages - 1
        if st.button("Next ▶", key="next_page", disabled=disabled):
            st.session_state.page += 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center;'>Page {st.session_state.page + 1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )

if not rows:
    st.info("No results. Try clearing filters or broaden your search.")
else:
    for r in page_rows:
        with st.container(border=True):
            inject_card_css()
            score = int(r.get("excitement_score") or 0)

            head_l, head_r = st.columns([0.82, 0.18], gap="small")
            with head_l:
                st.markdown(f"**[{r['title']}]({r['arxiv_link']})**")

                meta_bits = []
                if r.get("date"):
                    meta_bits.append(str(r["date"]))
                if r.get("reasoning_category"):
                    meta_bits.append(r["reasoning_category"])
                if r.get("authors"):
                    meta_bits.append(r["authors"])
                if meta_bits:
                    st.markdown(
                        f"<div class='card-meta'>{' • '.join(meta_bits)}</div>",
                        unsafe_allow_html=True,
                    )

            with head_r:
                badge_val = f"{score}/10" if score else "—/10"
                st.markdown(
                    f"<div style='text-align:right'><span class='score-badge'>{badge_val}</span></div>",
                    unsafe_allow_html=True,
                )

            if r.get("tldr"):
                st.markdown(f"> **TLDR:** {r['tldr']}")

            # Title + link
            if score:
                bd = parse_breakdown(r.get("score_breakdown") or "")
                labels = ["Novelty", "Impact", "Results", "Access"]

                def _safe_int(v):
                    try:
                        return int(float(v))
                    except (TypeError, ValueError):
                        return 0

                segment_bits = []
                for label in labels:
                    val = _safe_int(bd.get(label, 0) or 0)
                    max_val = BREAKDOWN_MAX.get(label, 0)
                    if max_val:
                        segment_bits.append(f"{label} {val}/{max_val}")
                    else:
                        segment_bits.append(f"{label} {val}")
                st.markdown(
                    "<div class='score-block'>"
                    f"<strong>Score: {score}/10</strong>"
                    f"<div class='score-breakdown-label'>Breakdown: {' | '.join(segment_bits)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                # tiny spacer between breakdown row and the assessment/expander
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                reasoning = (r.get("excitement_reasoning") or "").strip()
                if reasoning:
                    if len(reasoning) > 260:
                        with st.expander("Assessment"):
                            st.markdown(reasoning)
                    else:
                        st.markdown(f"**Assessment:** {reasoning}")

                # Removed fallback caption for no excitement score timestamp

                # Full Markdown summary (expander)
                if r.get("summary_md"):
                    with st.expander("Show full summary"):
                        st.markdown(r["summary_md"])
                        arxiv_id = (r.get("arxiv_id") or "").strip()
                        if arxiv_id and st.button("Load PDF", key=f"load_pdf_{r['id']}"):
                            st.markdown(
                                f'<iframe src="https://arxiv.org/pdf/{arxiv_id}" width="100%" height="600"></iframe>',
                                unsafe_allow_html=True,
                            )
