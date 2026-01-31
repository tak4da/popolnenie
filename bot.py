import os
import json
import re
import sqlite3
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from rapidfuzz import process, fuzz
from openai import OpenAI


# =========================
# CONFIG
# =========================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("Нет BOT_TOKEN в .env")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Нет DEEPSEEK_API_KEY в .env")

FAQ_SEGMENTED_PATH = "faq_segmented.json"
TERMS_SEGMENTED_PATH = "terms_segmented.json"
DB_PATH = "stats.db"

FAQ_PAGE_SIZE = 5
TERMS_PAGE_SIZE = 8

TOP_K = 10
FUZZY_MIN = 55
LLM_MIN_CONF = 0.55

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# =========================
# HELPERS
# =========================
def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def chunk_page(items: List[Any], page: int, page_size: int) -> Tuple[List[Any], int]:
    total_pages = (len(items) - 1) // page_size + 1 if items else 1
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], total_pages


# =========================
# LOAD DATA (SEGMENTED)
# =========================
FAQ_SEG = load_json(FAQ_SEGMENTED_PATH)
TERMS_SEG = load_json(TERMS_SEGMENTED_PATH)

# FAQ: category -> groups -> items
FAQ_CATEGORIES: List[Dict[str, Any]] = FAQ_SEG.get("faq", [])
FAQ_BY_ID: Dict[str, Dict[str, Any]] = {}
FAQ_QUESTIONS_NORM: List[str] = []
FAQ_NORM_TO_ID: Dict[str, str] = {}
FAQ_ANSWERS_NORM: List[str] = []
FAQ_ANSWER_NORM_TO_ID: Dict[str, str] = {}


for cat in FAQ_CATEGORIES:
    for grp in cat.get("groups", []):
        for it in grp.get("items", []):
            qid = it["id"]
            FAQ_BY_ID[qid] = it
            qn = normalize(it.get("q", ""))
            FAQ_QUESTIONS_NORM.append(qn)
            # если вдруг дубль формулировки — оставим первый, это ок
            FAQ_NORM_TO_ID.setdefault(qn, qid)
# нормализованные ответы тоже добавим в поиск
for qid, it in FAQ_BY_ID.items():
    an = normalize(it.get("a", ""))
    if an:
        FAQ_ANSWERS_NORM.append(an)
        # если вдруг дубль ответов — оставим первый
        FAQ_ANSWER_NORM_TO_ID.setdefault(an, qid)


# TERMS: dict kind -> list[{term, definition}]
TERM_KINDS: List[str] = sorted(list(TERMS_SEG.keys()))
TERMS_BY_KIND: Dict[str, List[Dict[str, str]]] = TERMS_SEG

# быстрый доступ по термину
TERM_MAP: Dict[str, str] = {}
for kind, arr in TERMS_BY_KIND.items():
    for t in arr:
        TERM_MAP[normalize(t["term"])] = t["definition"]


# =========================
# DB (TOP)
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faq_stats (
            qid TEXT PRIMARY KEY,
            cnt INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def inc_stat(qid: str) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO faq_stats(qid, cnt) VALUES(?, 1) "
        "ON CONFLICT(qid) DO UPDATE SET cnt = cnt + 1",
        (qid,),
    )
    conn.commit()
    conn.close()


def get_top_ids(limit: int = 10) -> List[str]:
    conn = db()
    cur = conn.execute(
        "SELECT qid FROM faq_stats ORDER BY cnt DESC LIMIT ?",
        (limit,),
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return [qid for qid in rows if qid in FAQ_BY_ID]


# =========================
# DeepSeek client
# =========================
ds_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


def fuzzy_candidates(user_text: str, top_k: int) -> List[Tuple[str, int]]:
    user_norm = normalize(user_text)
    results = process.extract(user_norm, FAQ_QUESTIONS_NORM, scorer=fuzz.WRatio, limit=top_k)
    out: List[Tuple[str, int]] = []
    for match, score, _ in results:
        if score >= FUZZY_MIN:
            out.append((match, int(score)))
    return out




def fuzzy_candidates_all(user_text: str, top_k: int) -> List[str]:
    """Кандидаты FAQ по вопросам + по ответам, сразу списком id."""
    user_norm = normalize(user_text)

    q_hits = process.extract(user_norm, FAQ_QUESTIONS_NORM, scorer=fuzz.WRatio, limit=top_k)
    a_hits = process.extract(user_norm, FAQ_ANSWERS_NORM, scorer=fuzz.WRatio, limit=top_k)

    ids: List[str] = []

    for match, score, _ in q_hits:
        if score >= FUZZY_MIN:
            qid = FAQ_NORM_TO_ID.get(match)
            if qid and qid not in ids:
                ids.append(qid)

    for match, score, _ in a_hits:
        if score >= FUZZY_MIN:
            qid = FAQ_ANSWER_NORM_TO_ID.get(match)
            if qid and qid not in ids:
                ids.append(qid)

    return ids[:10]


def deepseek_answer_from_context(user_text: str, ids: List[str]) -> Dict[str, Any]:
    """Собирает ответ строго по найденным пунктам базы (grounded)."""
    ctx: List[Dict[str, Any]] = []
    for qid in ids[:8]:
        it = FAQ_BY_ID.get(qid)
        if it:
            ctx.append({"id": qid, "q": it.get("q", ""), "a": it.get("a", "")})

    if not ctx:
        return {
            "answer": None,
            "used_ids": [],
            "confidence": 0.0,
            "need_clarify": True,
            "clarify_question": "Я не нашёл в базе подходящий пункт. Попробуй другими словами или добавь ключевое слово.",
        }

    system = (
        "Ты умный помощник FAQ-бота для внутренних процессов. "
        "Твоя главная задача: дать точный и короткий ответ по ситуации. "
        "У тебя есть КОНТЕКСТ (пункты базы). "
        "Правила: "
        "1) Отвечай ТОЛЬКО на основе контекста. Не выдумывай. "
        "2) Если контекста не хватает, скажи, что в базе нет точного ответа, и задай 1 уточняющий вопрос. "
        "3) Если подходят несколько пунктов, аккуратно объедини их, но без воды. "
        "Верни строго JSON без лишнего текста."
    )

    user = {
        "user_query": user_text,
        "context": ctx,
        "output_format": {
            "answer": "string|null",
            "used_ids": "array of string",
            "confidence": "number 0..1",
            "need_clarify": "boolean",
            "clarify_question": "string|null",
        },
    }

    resp = ds_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.0,
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return {
            "answer": None,
            "used_ids": [],
            "confidence": 0.0,
            "need_clarify": True,
            "clarify_question": "Не смог разобрать ответ модели. Спроси чуть проще или выбери вариант из поиска.",
        }

def deepseek_pick_id(user_text: str, candidates: List[Tuple[str, int]]) -> Dict[str, Any]:
    cand_payload = []
    for q_norm, score in candidates:
        qid = FAQ_NORM_TO_ID.get(q_norm)
        if qid and qid in FAQ_BY_ID:
            cand_payload.append({"id": qid, "q": FAQ_BY_ID[qid]["q"], "score": score})

    if not cand_payload:
        return {"id": None, "confidence": 0.0, "reason": "no_candidates"}

    system = (
        "Ты классификатор запросов для FAQ-бота. "
        "Выбери ОДИН наиболее подходящий id из списка кандидатов. "
        "Если ни один не подходит, верни id=null. "
        "Ответ строго JSON без лишнего текста."
    )

    user = {
        "user_query": user_text,
        "candidates": cand_payload,
        "output_format": {"id": "string|null", "confidence": "number 0..1", "reason": "string"},
        "rules": [
            "Выбирай только id из candidates",
            "Если совпадение слабое или запрос не про это, верни id=null",
            "confidence 0.9+ только если почти идеально",
        ],
    }

    resp = ds_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.0,
    )

    text = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return {"id": None, "confidence": 0.0, "reason": "bad_json"}


# =========================
# FSM
# =========================
class SearchFlow(StatesGroup):
    waiting_query = State()


class TermSearchFlow(StatesGroup):
    waiting_query = State()


class ChoiceFlow(StatesGroup):
    waiting_choice = State()



# =========================
# UI: KEYBOARDS
# =========================
def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📚 Вопросы", callback_data="menu:faq_cats:0")],
            [InlineKeyboardButton(text="🔎 Поиск по базе", callback_data="menu:search")],
            [InlineKeyboardButton(text="⭐ Топ-вопросы", callback_data="menu:top")],
            [InlineKeyboardButton(text="📖 Термины", callback_data="menu:term_kinds:0")],
        ]
    )


def nav_row(prev_cb: Optional[str], page: int, total_pages: int, next_cb: Optional[str]) -> List[InlineKeyboardButton]:
    row: List[InlineKeyboardButton] = []
    if prev_cb:
        row.append(InlineKeyboardButton(text="⬅️", callback_data=prev_cb))
    row.append(InlineKeyboardButton(text=f"📄 {page+1}/{total_pages}", callback_data="noop"))
    if next_cb:
        row.append(InlineKeyboardButton(text="➡️", callback_data=next_cb))
    return row


# ---- FAQ categories ----
def faq_categories_kb(page: int) -> InlineKeyboardMarkup:
    items, total_pages = chunk_page(FAQ_CATEGORIES, page, page_size=7)

    kb: List[List[InlineKeyboardButton]] = []
    for idx, cat in enumerate(items):
        cat_index = page * 7 + idx
        title = cat.get("category", "Категория")
        count = cat.get("count", 0)
        kb.append([InlineKeyboardButton(text=f"{title} ({count})", callback_data=f"faq_cat:{cat_index}:0")])

    prev_cb = f"menu:faq_cats:{page-1}" if page > 0 else None
    next_cb = f"menu:faq_cats:{page+1}" if page < total_pages - 1 else None
    kb.append(nav_row(prev_cb, page, total_pages, next_cb))

    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---- FAQ groups inside category ----
def faq_groups_kb(cat_index: int, page: int) -> InlineKeyboardMarkup:
    cat = FAQ_CATEGORIES[cat_index]
    groups = cat.get("groups", [])
    items, total_pages = chunk_page(groups, page, page_size=7)

    kb: List[List[InlineKeyboardButton]] = []
    for idx, grp in enumerate(items):
        grp_index = page * 7 + idx
        title = grp.get("title", "Группа")
        count = grp.get("count", 0)
        kb.append([InlineKeyboardButton(text=f"{title} ({count})", callback_data=f"faq_grp:{cat_index}:{grp_index}:0")])

    prev_cb = f"faq_cat:{cat_index}:{page-1}" if page > 0 else None
    next_cb = f"faq_cat:{cat_index}:{page+1}" if page < total_pages - 1 else None
    kb.append(nav_row(prev_cb, page, total_pages, next_cb))

    kb.append([InlineKeyboardButton(text="⬅️ К категориям", callback_data="menu:faq_cats:0")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---- Questions inside group (5 per page) ----
def faq_questions_kb(cat_index: int, grp_index: int, page: int) -> InlineKeyboardMarkup:
    grp = FAQ_CATEGORIES[cat_index]["groups"][grp_index]
    q_items = grp.get("items", [])
    items, total_pages = chunk_page(q_items, page, page_size=FAQ_PAGE_SIZE)

    kb: List[List[InlineKeyboardButton]] = []
    for it in items:
        kb.append([InlineKeyboardButton(text=it["q"][:80], callback_data=f"faq_q:{it['id']}")])

    prev_cb = f"faq_grp:{cat_index}:{grp_index}:{page-1}" if page > 0 else None
    next_cb = f"faq_grp:{cat_index}:{grp_index}:{page+1}" if page < total_pages - 1 else None
    kb.append(nav_row(prev_cb, page, total_pages, next_cb))

    kb.append([InlineKeyboardButton(text="⬅️ К группам", callback_data=f"faq_cat:{cat_index}:0")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---- Search results (pick question) ----
def search_results_kb(ids: List[str]) -> InlineKeyboardMarkup:
    kb: List[List[InlineKeyboardButton]] = []
    for qid in ids[:10]:
        it = FAQ_BY_ID.get(qid)
        if it:
            kb.append([InlineKeyboardButton(text=it["q"][:80], callback_data=f"faq_q:{qid}")])
    kb.append([InlineKeyboardButton(text="🔎 Новый поиск", callback_data="menu:search")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)

def format_top_options(ids: List[str], limit: int = 3) -> Tuple[List[str], str]:
    """Return (ids_top, text_block) where text_block is numbered options."""
    ids_top = [str(x) for x in ids[:limit]]
    lines = []
    for i, qid in enumerate(ids_top, 1):
        it = FAQ_BY_ID.get(qid) or {}
        q = (it.get('q') or '').strip()
        if not q:
            q = f"Вариант {i}"
        lines.append(f"{i}) {q[:120]}")
    return ids_top, "\n".join(lines)



# ---- TOP ----
def top_kb() -> InlineKeyboardMarkup:
    top_ids = get_top_ids(10)
    kb: List[List[InlineKeyboardButton]] = []
    if not top_ids:
        kb.append([InlineKeyboardButton(text="Пока пусто. Открой вопросы и покликай 🙂", callback_data="noop")])
    else:
        for qid in top_ids:
            it = FAQ_BY_ID[qid]
            kb.append([InlineKeyboardButton(text=it["q"][:80], callback_data=f"faq_q:{qid}")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---- Terms: kinds ----
def term_kinds_kb(page: int) -> InlineKeyboardMarkup:
    items, total_pages = chunk_page(TERM_KINDS, page, page_size=7)

    kb: List[List[InlineKeyboardButton]] = []
    for idx, kind in enumerate(items):
        kind_index = page * 7 + idx
        kb.append([InlineKeyboardButton(text=f"{kind} ({len(TERMS_BY_KIND[kind])})", callback_data=f"term_kind:{kind_index}:0")])

    prev_cb = f"menu:term_kinds:{page-1}" if page > 0 else None
    next_cb = f"menu:term_kinds:{page+1}" if page < total_pages - 1 else None
    kb.append(nav_row(prev_cb, page, total_pages, next_cb))

    kb.append([InlineKeyboardButton(text="🔎 Поиск термина", callback_data="menu:term_search")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# ---- Terms: inside kind ----
def term_list_kb(kind_index: int, page: int) -> InlineKeyboardMarkup:
    kind = TERM_KINDS[kind_index]
    items_all = TERMS_BY_KIND[kind]
    items, total_pages = chunk_page(items_all, page, page_size=TERMS_PAGE_SIZE)

    kb: List[List[InlineKeyboardButton]] = []
    for t in items:
        kb.append([InlineKeyboardButton(text=t["term"][:60], callback_data=f"term_show:{kind_index}:{normalize(t['term'])}")])

    prev_cb = f"term_kind:{kind_index}:{page-1}" if page > 0 else None
    next_cb = f"term_kind:{kind_index}:{page+1}" if page < total_pages - 1 else None
    kb.append(nav_row(prev_cb, page, total_pages, next_cb))

    kb.append([InlineKeyboardButton(text="⬅️ К разделам терминов", callback_data="menu:term_kinds:0")])
    kb.append([InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(inline_keyboard=kb)


# =========================
# HANDLERS
# =========================
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Привет! Напиши вопрос обычным текстом. Я постараюсь найти точный ответ в базе."
    )


async def noop(call: CallbackQuery) -> None:
    await call.answer()


# ---- MENU router ----
async def menu_router(call: CallbackQuery, state: FSMContext) -> None:
    parts = call.data.split(":")
    if parts[0] != "menu":
        await call.answer()
        return

    action = parts[1]

    if action == "home":
        await state.clear()
        await call.message.edit_text("Выбирай, как искать ответ:", reply_markup=main_menu_kb())
        await call.answer()
        return

    if action == "faq_cats":
        await state.clear()
        page = safe_int(parts[2], 0) if len(parts) > 2 else 0
        await call.message.edit_text("Выбери категорию:", reply_markup=faq_categories_kb(page))
        await call.answer()
        return

    if action == "search":
        await state.set_state(SearchFlow.waiting_query)
        await call.message.edit_text(
            "Напиши запрос для поиска по базе.\nМожно с ошибками, как получится 🙂",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")]]
            ),
        )
        await call.answer()
        return

    if action == "top":
        await state.clear()
        await call.message.edit_text("Топ-вопросы:", reply_markup=top_kb())
        await call.answer()
        return

    if action == "term_kinds":
        await state.clear()
        page = safe_int(parts[2], 0) if len(parts) > 2 else 0
        await call.message.edit_text("Разделы терминов:", reply_markup=term_kinds_kb(page))
        await call.answer()
        return

    if action == "term_search":
        await state.set_state(TermSearchFlow.waiting_query)
        await call.message.edit_text(
            "Напиши термин или кусок слова (например: sscc, адрес, гmv):",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:home")]]
            ),
        )
        await call.answer()
        return

    await call.answer()


# ---- FAQ navigation ----
async def faq_cat_handler(call: CallbackQuery) -> None:
    # faq_cat:<cat_index>:<page>
    _, cat_index_s, page_s = call.data.split(":")
    cat_index = safe_int(cat_index_s, 0)
    page = safe_int(page_s, 0)

    if cat_index < 0 or cat_index >= len(FAQ_CATEGORIES):
        await call.answer("Категория не найдена", show_alert=True)
        return

    title = FAQ_CATEGORIES[cat_index].get("category", "Категория")
    await call.message.edit_text(f"Категория: {title}\nВыбери группу:", reply_markup=faq_groups_kb(cat_index, page))
    await call.answer()


async def faq_group_handler(call: CallbackQuery) -> None:
    # faq_grp:<cat_index>:<grp_index>:<page>
    parts = call.data.split(":")
    cat_index = safe_int(parts[1], 0)
    grp_index = safe_int(parts[2], 0)
    page = safe_int(parts[3], 0)

    if cat_index < 0 or cat_index >= len(FAQ_CATEGORIES):
        await call.answer("Категория не найдена", show_alert=True)
        return
    groups = FAQ_CATEGORIES[cat_index].get("groups", [])
    if grp_index < 0 or grp_index >= len(groups):
        await call.answer("Группа не найдена", show_alert=True)
        return

    gtitle = groups[grp_index].get("title", "Группа")
    await call.message.edit_text(f"Группа: {gtitle}\nВыбери вопрос:", reply_markup=faq_questions_kb(cat_index, grp_index, page))
    await call.answer()


async def faq_question_handler(call: CallbackQuery) -> None:
    # faq_q:<id>
    _, qid = call.data.split(":", 1)
    it = FAQ_BY_ID.get(qid)
    if not it:
        await call.answer("Вопрос не найден", show_alert=True)
        return

    inc_stat(qid)
    await call.message.answer(it["a"])
    await call.answer()


# ---- SEARCH ----
async def search_query_handler(message: Message, state: FSMContext) -> None:
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer("Напиши текстом, что ищем 🙂")
        return

    # быстрый ответ термином — только если это реально запрос термина
    nt = normalize(user_text)

    best = process.extractOne(nt, list(TERM_MAP.keys()), scorer=fuzz.WRatio)
    if best:
        term_norm, score, _ = best
        if score >= 92:
            await message.answer(f"{term_norm.upper()}: {TERM_MAP[term_norm]}")
            return

    m = re.match(r"^(что такое|что значит|расшифруй|определи)\s+(.+)$", nt)
    if m:
        q = m.group(2).strip()
        best2 = process.extractOne(q, list(TERM_MAP.keys()), scorer=fuzz.WRatio)
        if best2:
            term_norm2, score2, _ = best2
            if score2 >= 80:
                await message.answer(f"{term_norm2.upper()}: {TERM_MAP[term_norm2]}")
                return

    ids = fuzzy_candidates_all(user_text, TOP_K)
    result = deepseek_answer_from_context(user_text, ids)

    ans = result.get("answer")
    conf = float(result.get("confidence", 0.0) or 0.0)
    used_ids = result.get("used_ids") or []

    if ans and conf >= LLM_MIN_CONF:
        if used_ids:
            inc_stat(str(used_ids[0]))
        await message.answer(str(ans))
        await state.clear()
        return

    # если нужна уточнялка — спросим, и предложим до 3 вариантов
    if bool(result.get("need_clarify")):
        clarify = result.get("clarify_question") or "Уточни вопрос, пожалуйста."
        await message.answer(str(clarify))
        if ids:
            ids_top, opts = format_top_options(ids, limit=3)
            await state.set_state(ChoiceFlow.waiting_choice)
            await state.update_data(choice_ids=ids_top)
            await message.answer("Если хочешь, выбери вариант (1-3):\n" + opts)
        return

    if ids:
        ids_top, opts = format_top_options(ids, limit=3)
        await state.set_state(ChoiceFlow.waiting_choice)
        await state.update_data(choice_ids=ids_top)
        await message.answer("Похоже на несколько вариантов. Напиши номер (1-3):\n" + opts)
    else:
        await message.answer("По базе пока не попал. Попробуй другими словами.")


# ---- TERMS ----
async def term_kind_handler(call: CallbackQuery) -> None:
    # term_kind:<kind_index>:<page>
    _, kind_index_s, page_s = call.data.split(":")
    kind_index = safe_int(kind_index_s, 0)
    page = safe_int(page_s, 0)

    if kind_index < 0 or kind_index >= len(TERM_KINDS):
        await call.answer("Раздел не найден", show_alert=True)
        return

    kind = TERM_KINDS[kind_index]
    await call.message.edit_text(f"Термины: {kind}", reply_markup=term_list_kb(kind_index, page))
    await call.answer()


async def term_show_handler(call: CallbackQuery) -> None:
    # term_show:<kind_index>:<term_norm>
    parts = call.data.split(":", 2)
    kind_index = safe_int(parts[1], 0)
    term_norm = parts[2] if len(parts) > 2 else ""

    defin = TERM_MAP.get(term_norm)
    if not defin:
        await call.answer("Термин не найден", show_alert=True)
        return

    # показываем определение + оставляем клаву на месте (чтобы можно было дальше листать)
    await call.message.answer(f"{term_norm.upper()}: {defin}")
    await call.answer()


async def term_search_handler(message: Message, state: FSMContext) -> None:
    q = normalize(message.text or "")
    if not q:
        await message.answer("Напиши термин текстом 🙂")
        return

    # простая выдача топ-10 совпадений по подстроке и fuzzy
    matches = []
    for term_norm, defin in TERM_MAP.items():
        if q in term_norm:
            matches.append((term_norm, 100))
    if len(matches) < 10:
        # добиваем fuzzy по терминам
        all_terms = list(TERM_MAP.keys())
        for match, score, _ in process.extract(q, all_terms, scorer=fuzz.WRatio, limit=10):
            matches.append((match, int(score)))

    # уникализируем, сортируем
    uniq: Dict[str, int] = {}
    for t, s in matches:
        uniq[t] = max(uniq.get(t, 0), s)
    best = sorted(uniq.items(), key=lambda x: -x[1])[:10]

    if not best:
        await message.answer("Не нашёл термин. Попробуй по-другому.")
        return

    text_lines = ["Нашёл вот что:"]
    for t, _ in best[:5]:
        text_lines.append(f"- {t.upper()}: {TERM_MAP[t]}")
    await message.answer("\n".join(text_lines))


# ---- DEFAULT TEXT (optional smart assist) ----
async def default_text_handler(message: Message, state: FSMContext) -> None:
    # если ждём выбор варианта — не вмешиваемся
    st = await state.get_state()
    if st == ChoiceFlow.waiting_choice.state:
        return
    if st in {SearchFlow.waiting_query.state, TermSearchFlow.waiting_query.state}:
        return

    user_text = (message.text or "").strip()
    if not user_text:
        return

    # 1) термины — только если это реально запрос термина
    nt = normalize(user_text)

    best = process.extractOne(nt, list(TERM_MAP.keys()), scorer=fuzz.WRatio)
    if best:
        term_norm, score, _ = best
        if score >= 92:
            await message.answer(f"{term_norm.upper()}: {TERM_MAP[term_norm]}")
            return

    m = re.match(r"^(что такое|что значит|расшифруй|определи)\s+(.+)$", nt)
    if m:
        q = m.group(2).strip()
        best2 = process.extractOne(q, list(TERM_MAP.keys()), scorer=fuzz.WRatio)
        if best2:
            term_norm2, score2, _ = best2
            if score2 >= 80:
                await message.answer(f"{term_norm2.upper()}: {TERM_MAP[term_norm2]}")
                return

    # 2) умный ответ по базе: кандидаты (вопросы+ответы) -> дипсик собирает ответ строго по контексту
    ids = fuzzy_candidates_all(user_text, TOP_K)
    result = deepseek_answer_from_context(user_text, ids)

    ans = result.get("answer")
    conf = float(result.get("confidence", 0.0) or 0.0)
    used_ids = result.get("used_ids") or []

    if ans and conf >= LLM_MIN_CONF:
        if used_ids:
            inc_stat(str(used_ids[0]))
        await message.answer(str(ans))
        return

    if bool(result.get("need_clarify")):
        clarify = result.get("clarify_question") or "Уточни вопрос, пожалуйста."
        await message.answer(str(clarify))
        if ids:
            ids_top, opts = format_top_options(ids, limit=3)
            await state.set_state(ChoiceFlow.waiting_choice)
            await state.update_data(choice_ids=ids_top)
            await message.answer("Если хочешь, выбери вариант (1-3):\n" + opts)
        return

    if ids:
        ids_top, opts = format_top_options(ids, limit=3)
        await state.set_state(ChoiceFlow.waiting_choice)
        await state.update_data(choice_ids=ids_top)
        await message.answer("Похоже на несколько вариантов. Напиши номер (1-3):\n" + opts)
    else:
        await message.answer("Не нашёл точного ответа. Попробуй переформулировать вопрос чуть проще.")


# ---- CHOICE (1..3) ----
async def choice_handler(message: Message, state: FSMContext) -> None:
    st = await state.get_state()
    if st != ChoiceFlow.waiting_choice.state:
        return

    txt = (message.text or "").strip()
    if not txt:
        return

    m = re.match(r"^(?:вариант\s*)?(\d)$", txt.lower())
    if not m:
        await message.answer("Напиши номер 1, 2 или 3 (или просто переформулируй вопрос).")
        return

    n = int(m.group(1))
    data = await state.get_data()
    ids = data.get("choice_ids") or []
    if not (1 <= n <= len(ids)):
        await message.answer("Такого варианта нет. Напиши 1, 2 или 3.")
        return

    qid = str(ids[n - 1])
    it = FAQ_BY_ID.get(qid)
    if not it:
        await message.answer("Не нашёл этот пункт в базе. Спроси по-другому.")
        await state.clear()
        return

    inc_stat(qid)
    await message.answer(str(it.get("a") or ""))
    await state.clear()


# =========================
# MAIN
# =========================
def main() -> None:
    conn = db()
    conn.close()

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, F.text.in_({"/start", "/help"}))

    # выбор варианта (1..3), если бот предложил несколько трактовок
    dp.message.register(choice_handler, ChoiceFlow.waiting_choice, F.text)

    # основной умный режим
    dp.message.register(default_text_handler, F.text)


    dp.run_polling(bot)


if __name__ == "__main__":
    main()
