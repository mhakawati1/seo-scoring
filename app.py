from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st


# ------------------------------
# Constants
# ------------------------------
REQUIRED_COLUMNS = ["title", "body", "url"]
SCORE_COLUMNS = ["title_seo_score", "body_seo_score", "overall_seo_score"]
TEXT_OUTPUT_COLUMNS = [
    "title_assessment",
    "body_assessment",
    "url_assessment",
    "strengths",
    "issues",
    "recommended_improvements",
    "suggested_seo_title",
    "processing_status",
]

SEO_DEVELOPER_PROMPT = """
أنت خبير محترف في SEO للمحتوى العربي، خصوصاً الأخبار والمقالات الرقمية.

ستستلم مقالاً يحتوي على:
- title
- body
- url

مهمتك هي تقييم مدى تحسين العنوان والمحتوى لمحركات البحث، مع التركيز على السياق العربي والتحريري.

قيّم العنوان من 0 إلى 10 بناءً على:
1. وضوح العنوان ودقته
2. احتوائه على الموضوع أو الكلمة المفتاحية بشكل طبيعي
3. ملاءمته لنية البحث
4. مناسبته للظهور في نتائج البحث
5. جاذبيته بدون تضليل
6. تطابقه مع محتوى المقال
7. تجنّب الغموض والعمومية الزائدة

قيّم المحتوى من 0 إلى 10 بناءً على:
1. وضوح الموضوع الرئيسي من البداية
2. إبراز الموضوع والكلمات المهمة بشكل طبيعي
3. ملاءمة المحتوى لنية البحث
4. عمق التغطية وكفايتها
5. سهولة القراءة والتنظيم
6. الحاجة أو عدم الحاجة لعناوين فرعية
7. تجنب الحشو والتكرار
8. القيمة المعلوماتية والأصالة
9. مناسبة النص لـ News SEO أو Editorial SEO العربي
10. وجود فرص تحسين عملية وواضحة

قيّم الرابط URL من ناحية الوضوح والارتباط بالموضوع فقط.

قواعد ملزمة:
- أعد النتيجة بصيغة JSON فقط
- أعط الدرجات كأرقام صحيحة من 0 إلى 10
- كن صارماً ولكن عادلاً
- لا تخترع معلومات غير موجودة
- لا تعد كتابة المقال كاملاً
- إذا كان النص جيداً تحريرياً لكنه أضعف في SEO، وضّح ذلك
- اجعل الملاحظات قصيرة وعملية
""".strip()

RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "arabic_seo_audit",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title_seo_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "body_seo_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "overall_seo_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "title_assessment": {"type": "string"},
            "body_assessment": {"type": "string"},
            "url_assessment": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array", "items": {"type": "string"}},
            "recommended_improvements": {"type": "array", "items": {"type": "string"}},
            "suggested_seo_title": {"type": "string"},
        },
        "required": [
            "title_seo_score",
            "body_seo_score",
            "overall_seo_score",
            "title_assessment",
            "body_assessment",
            "url_assessment",
            "strengths",
            "issues",
            "recommended_improvements",
            "suggested_seo_title",
        ],
    },
    "strict": True,
}


# ------------------------------
# Helpers
# ------------------------------
def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def safe_join(items: Optional[Iterable[str]]) -> str:
    if not items:
        return ""
    return " | ".join(clean_text(x) for x in items if clean_text(x))


def fix_header_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df.columns):
        return df

    if all(str(c).startswith("Unnamed") for c in df.columns):
        for i in range(min(5, len(df))):
            potential_header = [clean_text(x).lower() for x in df.iloc[i].tolist()]
            if any(x in potential_header for x in REQUIRED_COLUMNS):
                df = df.copy()
                df.columns = df.iloc[i]
                df = df.drop(index=list(range(i + 1))).reset_index(drop=True)
                break
    return df


def normalize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        lower = clean_text(col).lower()
        if lower in REQUIRED_COLUMNS:
            rename_map[col] = lower
    return df.rename(columns=rename_map)


def validate_columns(df: pd.DataFrame) -> List[str]:
    normalized = {clean_text(c).lower(): c for c in df.columns}
    return [col for col in REQUIRED_COLUMNS if col not in normalized]


def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df = fix_header_if_needed(df)
    df = normalize_required_columns(df)
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="seo_scored")
    buffer.seek(0)
    return buffer.getvalue()


def color_score(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 8:
        return "background-color: rgba(52,199,89,0.25)"
    if v >= 6:
        return "background-color: rgba(255,204,0,0.25)"
    return "background-color: rgba(255,69,58,0.25)"


def make_gemini_safe_schema(schema: Any) -> Any:
    """
    Remove JSON Schema fields Gemini may reject.
    """
    if isinstance(schema, dict):
        cleaned = {}
        for key, value in schema.items():
            if key in {"additionalProperties", "$schema"}:
                continue
            cleaned[key] = make_gemini_safe_schema(value)
        return cleaned
    if isinstance(schema, list):
        return [make_gemini_safe_schema(item) for item in schema]
    return schema


GEMINI_RESPONSE_SCHEMA = make_gemini_safe_schema(RESPONSE_JSON_SCHEMA["schema"])


# ------------------------------
# Provider clients
# ------------------------------
def get_openai_client(api_key: Optional[str] = None):
    from openai import OpenAI

    final_api_key = clean_text(api_key) or os.getenv("OPENAI_API_KEY", "")
    if not final_api_key:
        raise RuntimeError("OpenAI API key is not set. Add it in the UI or as OPENAI_API_KEY.")
    return OpenAI(api_key=final_api_key)


def get_gemini_client(api_key: Optional[str] = None):
    from google import genai

    final_api_key = clean_text(api_key) or os.getenv("GEMINI_API_KEY", "")
    if not final_api_key:
        raise RuntimeError("Gemini API key is not set. Add it in the UI or as GEMINI_API_KEY.")
    return genai.Client(api_key=final_api_key)


# ------------------------------
# Provider scoring
# ------------------------------
def parse_openai_response_text(response: Any) -> Dict[str, Any]:
    text = getattr(response, "output_text", "")
    if text:
        return json.loads(text)

    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            maybe_text = getattr(content_item, "text", None)
            if maybe_text:
                return json.loads(maybe_text)

    raise ValueError("Could not find JSON text in OpenAI response.")


def score_article_openai(client: Any, model: str, title: str, body: str, url: str) -> Dict[str, Any]:
    user_prompt = f"""
قيّم هذا المقال:

title: {title}
body: {body}
url: {url}
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": SEO_DEVELOPER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": RESPONSE_JSON_SCHEMA["name"],
                "schema": RESPONSE_JSON_SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    return parse_openai_response_text(response)


def score_article_gemini(client: Any, model: str, title: str, body: str, url: str) -> Dict[str, Any]:
    from google.genai import types

    user_prompt = f"""
قيّم هذا المقال:

title: {title}
body: {body}
url: {url}
""".strip()

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SEO_DEVELOPER_PROMPT,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini response did not include text.")
    return json.loads(text)


def score_article(
    provider: str,
    client: Any,
    model: str,
    title: str,
    body: str,
    url: str,
) -> Dict[str, Any]:
    if provider == "OpenAI":
        return score_article_openai(client, model, title, body, url)
    if provider == "Gemini":
        return score_article_gemini(client, model, title, body, url)
    raise ValueError(f"Unsupported provider: {provider}")


# ------------------------------
# Processing
# ------------------------------
def process_dataframe(
    df: pd.DataFrame,
    provider: str,
    model: str,
    max_rows: Optional[int] = None,
    api_key: Optional[str] = None,
    max_workers: int = 5,
) -> pd.DataFrame:
    working_df = df.copy()

    if provider == "OpenAI":
        client = get_openai_client(api_key=api_key)
    elif provider == "Gemini":
        client = get_gemini_client(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    for col in SCORE_COLUMNS:
        if col not in working_df.columns:
            working_df[col] = pd.Series([pd.NA] * len(working_df), dtype="object")
        else:
            working_df[col] = working_df[col].astype("object")

    for col in TEXT_OUTPUT_COLUMNS:
        if col not in working_df.columns:
            working_df[col] = pd.Series([""] * len(working_df), dtype="object")
        else:
            working_df[col] = working_df[col].astype("object")

    rows_to_process = list(working_df.index)
    if max_rows is not None:
        rows_to_process = rows_to_process[:max_rows]

    total = max(len(rows_to_process), 1)

    def process_row(idx: int):
        title = clean_text(working_df.at[idx, "title"])
        body = clean_text(working_df.at[idx, "body"])
        url = clean_text(working_df.at[idx, "url"])

        if not title and not body:
            return idx, {"processing_status": "Skipped: empty title and body"}

        try:
            result = score_article(
                provider=provider,
                client=client,
                model=model,
                title=title,
                body=body,
                url=url,
            )
            return idx, {
                "title_seo_score": result.get("title_seo_score", ""),
                "body_seo_score": result.get("body_seo_score", ""),
                "overall_seo_score": result.get("overall_seo_score", ""),
                "title_assessment": result.get("title_assessment", ""),
                "body_assessment": result.get("body_assessment", ""),
                "url_assessment": result.get("url_assessment", ""),
                "strengths": safe_join(result.get("strengths")),
                "issues": safe_join(result.get("issues")),
                "recommended_improvements": safe_join(result.get("recommended_improvements")),
                "suggested_seo_title": result.get("suggested_seo_title", ""),
                "processing_status": "Processed",
            }
        except Exception as exc:
            return idx, {"processing_status": f"Error: {str(exc)[:250]}"}

    progress = st.progress(0)
    status_box = st.empty()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, idx) for idx in rows_to_process]

        for future in as_completed(futures):
            idx, result = future.result()

            for key, value in result.items():
                working_df.at[idx, key] = value

            completed += 1
            status_box.info(f"Processing row {completed} of {total}")
            progress.progress(completed / total)

    status_box.success("Processing complete.")
    return working_df


# ------------------------------
# Streamlit app
# ------------------------------
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
if "processed_filename" not in st.session_state:
    st.session_state["processed_filename"] = None

st.set_page_config(page_title="Arabic SEO Excel Scorer", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.10), rgba(0, 200, 150, 0.08));
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
    }
    </style>
    <div class="hero">
        <h1>🧭 Arabic SEO Excel Scorer</h1>
        <p>Upload an Excel sheet, review the articles, run Arabic SEO scoring, and download the enriched file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=["xlsx", "xls"],
            help="The sheet should include columns named title, body, and url.",
        )

    with right_col:
        provider = st.selectbox("Provider", ["OpenAI", "Gemini"])
        default_model = "gpt-4.1-mini" if provider == "OpenAI" else "gemini-3.1-flash-lite-preview"
        model_name = st.text_input("Model", value=default_model)

        api_key_input = st.text_input(
            "API key",
            type="password",
            placeholder="sk-..." if provider == "OpenAI" else "AIza...",
            help="Used only for this session in the running app.",
        )

        process_limit = st.number_input("Rows to process", min_value=1, value=20, step=1)
        max_workers = st.slider("Parallel requests", min_value=1, max_value=8, value=5, step=1)

with st.expander("Expected format", expanded=False):
    st.markdown("Required columns:")
    st.code("title\nbody\nurl")
    st.markdown("The app will add these columns:")
    st.code(
        "title_seo_score\n"
        "body_seo_score\n"
        "overall_seo_score\n"
        "processing_status\n"
        "suggested_seo_title"
    )
    st.markdown("API key:")
    st.code("Paste your OpenAI or Gemini API key in the UI field above.")

if not uploaded_file:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    df = load_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.exception(exc)
    st.stop()

file_name_stem = Path(uploaded_file.name).stem
st.success(f"Loaded file successfully: {uploaded_file.name}")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Rows detected", len(df))
metric_col2.metric("Columns detected", len(df.columns))
metric_col3.metric("Ready to process", "Yes")

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(10), width="stretch")

if st.button("Run SEO scoring", type="primary", width="stretch"):
    try:
        processed_df = process_dataframe(
            df,
            provider=provider,
            model=model_name,
            max_rows=process_limit,
            api_key=api_key_input,
            max_workers=max_workers,
        )
        st.session_state["processed_df"] = processed_df
        st.session_state["processed_filename"] = f"{file_name_stem}_seo_scored.xlsx"
    except Exception as exc:
        st.session_state["processed_df"] = None
        st.error(f"Processing failed: {exc}")
        st.exception(exc)

if st.session_state["processed_df"] is not None:
    processed_df = st.session_state["processed_df"]

    score_series = pd.to_numeric(processed_df.get("overall_seo_score"), errors="coerce")
    avg_score = round(float(score_series.dropna().mean()), 2) if score_series.notna().any() else 0.0
    processed_count = int((processed_df.get("processing_status") == "Processed").sum()) if "processing_status" in processed_df.columns else 0
    error_count = int(
        processed_df.get("processing_status", pd.Series(dtype="object"))
        .astype(str)
        .str.startswith("Error:")
        .sum()
    ) if "processing_status" in processed_df.columns else 0

    st.markdown("### Results dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Average overall score", avg_score)
    m2.metric("Processed rows", processed_count)
    m3.metric("Rows with errors", error_count)

    st.markdown("### Processed results")
    preview_columns = [
        c for c in [
            "title",
            "title_seo_score",
            "body_seo_score",
            "overall_seo_score",
            "suggested_seo_title",
            "processing_status",
        ] if c in processed_df.columns
    ]

    styled = processed_df[preview_columns].head(20).style
    for col in [c for c in ["title_seo_score", "body_seo_score", "overall_seo_score"] if c in preview_columns]:
        styled = styled.map(color_score, subset=[col])

    st.dataframe(styled, width="stretch")

    excel_bytes = dataframe_to_excel_bytes(processed_df)
    st.download_button(
        label="Download updated Excel",
        data=excel_bytes,
        file_name=st.session_state["processed_filename"] or "seo_scored.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )