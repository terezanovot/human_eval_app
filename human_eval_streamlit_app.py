import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import html

st.markdown(
    """
    <style>
    .query-box-label {
        color: #000000 !important;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .query-box textarea,
    .query-box textarea:disabled,
    .query-box [data-baseweb="textarea"] textarea,
    .query-box [data-baseweb="textarea"] textarea:disabled,
    .query-box [data-baseweb="base-input"] textarea,
    .query-box [data-baseweb="base-input"] textarea:disabled {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        opacity: 1 !important;
        border: 1px solid #bdbdbd !important;
        caret-color: #000000 !important;
    }

    .query-box [data-baseweb="textarea"],
    .query-box [data-baseweb="base-input"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 1px solid #bdbdbd !important;
        border-radius: 0.5rem !important;
    }

    .query-box [data-baseweb="textarea"] > div,
    .query-box [data-baseweb="base-input"] > div {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Hodnocení vyhledávání judikatury", layout="wide")

DATA_DIR = Path("human_eval_outputs")
DB_PATH = DATA_DIR / "human_eval_app.db"

TASK1_POOL_PATH = DATA_DIR / "task1_website_pool.parquet"
TASK2_POOL_PATH = DATA_DIR / "task2_website_pool.parquet"
ASSIGNMENTS_PATH = DATA_DIR / "evaluator_assignments.parquet"

RELEVANCE_OPTIONS = {
    6: "6 = vysoce relevantní",
    5: "5 = velmi relevantní",
    4: "4 = spíše relevantní",
    3: "3 = částečně relevantní",
    2: "2 = málo relevantní",
    1: "1 = nerelevantní",
}

WHOLE_DECISION_OPTIONS = {
    6: "6 = vysoce relevantní jako celek",
    5: "5 = velmi relevantní jako celek",
    4: "4 = spíše relevantní jako celek",
    3: "3 = částečně relevantní jako celek",
    2: "2 = málo relevantní jako celek",
    1: "1 = nerelevantní jako celek",
}


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judgments (
                judgment_id TEXT PRIMARY KEY,
                evaluator_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                task TEXT NOT NULL,
                candidate_uid TEXT NOT NULL,
                candidate_decision_id TEXT NOT NULL,
                relevance_label INTEGER,
                whole_decision_label INTEGER,
                confidence INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@st.cache_data
def load_task_pool(task_name: str) -> pd.DataFrame:
    if task_name == "task1":
        df = pd.read_parquet(TASK1_POOL_PATH)
    elif task_name == "task2":
        df = pd.read_parquet(TASK2_POOL_PATH)
    else:
        raise ValueError(task_name)

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_assignments() -> pd.DataFrame:
    if ASSIGNMENTS_PATH.exists():
        df = pd.read_parquet(ASSIGNMENTS_PATH)
    else:
        csv_path = DATA_DIR / "evaluator_assignments.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=["evaluator_id", "task", "query_id"])
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["evaluator_id", "task", "query_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data
def load_evaluator_ids() -> list[str]:
    df = load_assignments()
    if df.empty or "evaluator_id" not in df.columns:
        return []
    return sorted(df["evaluator_id"].astype(str).str.strip().dropna().unique().tolist())


def read_saved_judgment(evaluator_id: str, candidate_uid: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT relevance_label, whole_decision_label, confidence, comment
            FROM judgments
            WHERE evaluator_id = ? AND candidate_uid = ?
            """,
            (evaluator_id, candidate_uid),
        ).fetchone()

    if row is None:
        return None

    return {
        "relevance_label": row[0],
        "whole_decision_label": row[1],
        "confidence": row[2],
        "comment": row[3] or "",
    }


def save_judgment(
    evaluator_id: str,
    query_id: str,
    task: str,
    candidate_uid: str,
    candidate_decision_id: str,
    relevance_label: int,
    whole_decision_label: int,
    confidence: int,
    comment: str,
) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    judgment_id = f"{evaluator_id}::{candidate_uid}"

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT created_at FROM judgments WHERE judgment_id = ?",
            (judgment_id,),
        ).fetchone()
        created_at = existing[0] if existing else now

        conn.execute(
            """
            INSERT OR REPLACE INTO judgments (
                judgment_id, evaluator_id, query_id, task,
                candidate_uid, candidate_decision_id,
                relevance_label, whole_decision_label, confidence, comment,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                judgment_id,
                evaluator_id,
                query_id,
                task,
                candidate_uid,
                candidate_decision_id,
                relevance_label,
                whole_decision_label,
                confidence,
                comment,
                created_at,
                now,
            ),
        )
        conn.commit()


@st.cache_data
def get_progress_snapshot() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["evaluator_id", "task", "n_saved"])

    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT evaluator_id, task, COUNT(*) AS n_saved
            FROM judgments
            GROUP BY evaluator_id, task
            """,
            conn,
        )
    return df


def render_query_panel(task: str, qdf: pd.DataFrame) -> None:
    st.subheader("Dotaz")
    qrow = qdf.iloc[0]

    if task == "task1":
        if str(qrow.get("query_docket_number", "")).strip():
            st.markdown(f"**Spisová značka:** {qrow['query_docket_number']}")
        if str(qrow.get("query_url", "")).strip():
            st.markdown(f"[Otevřít celé rozhodnutí dotazu]({qrow['query_url']})")

        query_text = html.escape(str(qrow.get("query_display_text", ""))).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="query-box-label">Skutkové okolnosti dotazovaného rozhodnutí</div>
            <div style="
                height: 900px;
                overflow-y: auto;
                background: #ffffff;
                border: 1px solid #bdbdbd;
                border-radius: 8px;
                padding: 14px 16px;
                color: #000000;
                line-height: 1.5;
                white-space: normal;
            ">{query_text}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        query_text = html.escape(str(qrow.get("query_display_text", ""))).replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="query-box-label">Text dotazu</div>
            <div style="
                min-height: 180px;
                overflow-y: auto;
                background: #ffffff;
                border: 1px solid #bdbdbd;
                border-radius: 8px;
                padding: 14px 16px;
                color: #000000;
                line-height: 1.5;
                white-space: normal;
            ">{query_text}</div>
            """,
            unsafe_allow_html=True,
        )


def render_candidate_card(task: str, row: pd.Series, evaluator_id: str) -> None:
    saved = read_saved_judgment(evaluator_id=evaluator_id, candidate_uid=str(row["candidate_uid"]))

    with st.expander(
        f"{int(row['display_order'])}. {row.get('candidate_docket_number', '') or row['candidate_decision_id']}",
        expanded=False,
    ):
        if str(row.get("candidate_url", "")).strip():
            st.markdown(f"[Otevřít celé kandidátní rozhodnutí]({row['candidate_url']})")

        display_label = (
            "Skutkové pozadí kandidátního rozhodnutí"
            if task == "task1"
            else "Zobrazený odstavec kandidátního rozhodnutí"
        )

        st.text_area(
            display_label,
            value=str(row.get("candidate_display_text", "")),
            height=220,
            disabled=True,
            key=f"candidate_text_{row['candidate_uid']}",
        )

        if task == "task2" and pd.notna(row.get("candidate_paragraph_index")):
            st.caption(f"Index odstavce: {row['candidate_paragraph_index']}")

        st.markdown("**Hodnocení**")

        saved_rel = saved["relevance_label"] if saved else 4
        saved_whole = saved["whole_decision_label"] if saved else 4
        saved_conf = saved["confidence"] if saved else 3
        saved_comment = saved["comment"] if saved else ""

        rel = st.radio(
            "Nakolik je zobrazený text relevantní k dotazu?",
            options=list(RELEVANCE_OPTIONS.keys()),
            format_func=lambda x: RELEVANCE_OPTIONS[x],
            horizontal=True,
            key=f"rel_{row['candidate_uid']}",
            index=list(RELEVANCE_OPTIONS.keys()).index(saved_rel),
        )

        whole = st.radio(
            "Nakolik je kandidátní rozhodnutí relevantní k dotazu jako celek?",
            options=list(WHOLE_DECISION_OPTIONS.keys()),
            format_func=lambda x: WHOLE_DECISION_OPTIONS[x],
            horizontal=True,
            key=f"whole_{row['candidate_uid']}",
            index=list(WHOLE_DECISION_OPTIONS.keys()).index(saved_whole),
        )

        confidence = st.slider(
            "Jistota hodnocení",
            min_value=1,
            max_value=5,
            value=saved_conf,
            key=f"conf_{row['candidate_uid']}",
        )

        comment = st.text_area(
            "Volitelný komentář",
            value=saved_comment,
            key=f"comment_{row['candidate_uid']}",
        )

        if st.button("Uložit hodnocení", key=f"save_{row['candidate_uid']}"):
            save_judgment(
                evaluator_id=evaluator_id,
                query_id=str(row["query_id"]),
                task=str(row["task"]),
                candidate_uid=str(row["candidate_uid"]),
                candidate_decision_id=str(row["candidate_decision_id"]),
                relevance_label=int(rel),
                whole_decision_label=int(whole),
                confidence=int(confidence),
                comment=str(comment),
            )
            st.success("Uloženo.")


init_db()

st.title("Lidské hodnocení vyhledávání judikatury Ústavního soudu")
st.caption("Lokální testovací verze")

with st.sidebar:
    st.header("Relace")

    available_evaluator_ids = load_evaluator_ids()
    if len(available_evaluator_ids) == 0:
        st.error("V souboru s přiřazením nebyla nalezena žádná ID hodnotitelů.")
        st.stop()

    evaluator_id = st.selectbox(
        "ID hodnotitele",
        options=available_evaluator_ids,
        index=0,
        key="selected_evaluator_id",
    )

    task = st.radio(
        "Úloha",
        options=["task1", "task2"],
        format_func=lambda x: "Úloha 1" if x == "task1" else "Úloha 2",
    )

    assignments_df = load_assignments()
    evaluator_assignments = assignments_df[
        (assignments_df["evaluator_id"] == str(evaluator_id).strip())
        & (assignments_df["task"] == task)
    ].copy()

    current_queries = evaluator_assignments["query_id"].astype(str).drop_duplicates().tolist()

    session_key = f"query_idx_{evaluator_id}_{task}"
    if session_key not in st.session_state:
        st.session_state[session_key] = 0

    if len(current_queries) == 0:
        st.warning("Pro tohoto hodnotitele a tuto úlohu nebyly nalezeny žádné přiřazené dotazy.")
        st.stop()

    query_idx = st.session_state[session_key]
    query_idx = min(query_idx, max(len(current_queries) - 1, 0))
    st.session_state[session_key] = query_idx

    st.write(f"Počet přiřazených dotazů v této úloze: {len(current_queries)}")

    progress_df = get_progress_snapshot()
    if not progress_df.empty and evaluator_id:
        person = progress_df[progress_df["evaluator_id"] == evaluator_id]
        if not person.empty:
            st.dataframe(person, use_container_width=True, hide_index=True)

pool_df = load_task_pool(task)
current_query_id = current_queries[st.session_state[session_key]]
qdf = pool_df[pool_df["query_id"].astype(str) == str(current_query_id)].copy()
qdf = qdf.sort_values("display_order").reset_index(drop=True)

col_a, col_b = st.columns([1, 2])
with col_a:
    render_query_panel(task, qdf)
with col_b:
    st.subheader(f"Kandidáti pro dotaz {st.session_state[session_key] + 1} / {len(current_queries)}")
    for _, row in qdf.iterrows():
        render_candidate_card(task, row, evaluator_id=evaluator_id)

nav1, nav2, nav3 = st.columns([1, 1, 2])
with nav1:
    if st.button("Předchozí dotaz") and st.session_state[session_key] > 0:
        st.session_state[session_key] -= 1
        st.rerun()
with nav2:
    if st.button("Další dotaz") and st.session_state[session_key] < len(current_queries) - 1:
        st.session_state[session_key] += 1
        st.rerun()
with nav3:
    st.write("")

if st.checkbox("Zobrazit tabulku uložených hodnocení"):
    with get_conn() as conn:
        all_judgments = pd.read_sql_query("SELECT * FROM judgments ORDER BY updated_at DESC", conn)
    st.dataframe(all_judgments, use_container_width=True, hide_index=True)

    csv_bytes = all_judgments.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Stáhnout uložená hodnocení jako CSV",
        data=csv_bytes,
        file_name="human_eval_judgments.csv",
        mime="text/csv",
    )
