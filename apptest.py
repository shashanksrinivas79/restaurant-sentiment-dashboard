import os
from typing import Dict, Tuple
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, session, url_for
from flask_bcrypt import Bcrypt

# Dash / plotting
from dash import Dash, dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Optional: for LDA topic modeling fallback
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flask import Flask, render_template, request, redirect, session, url_for, make_response, jsonify

# NLP helpers
import nltk
from nltk.corpus import stopwords
from collections import Counter
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOPWORDS
    STOPWORDS = set(SKLEARN_STOPWORDS)
except Exception:
    STOPWORDS = set([
        "the","and","for","that","this","with","have","not","but","are","was","you","your",
        "what","when","where","which","from","they","their","them","there","been","were",
        "will","would","should","about","could","than","then","one","all","can","get"
    ])


# ---------- Config ----------
DATA_DIR = os.environ.get("DATA_DIR", os.getcwd())
business_df = pd.read_csv("/Users/shashankmidididdi/Downloads/business.csv")
reviews_df = pd.read_csv("reviews_with_sentiment.csv")
daily_df = pd.read_csv("agg_sentiment_daily.csv")
topwords_df = pd.read_csv("agg_top_words_by_category.csv")
users_df = pd.read_csv("user_access.csv")

# ---------- Flask server ----------
server = Flask(__name__, template_folder="templates", static_folder="assets")
server.secret_key = os.environ.get("APP_SECRET_KEY", "dev_secret_change_me")
bcrypt = Bcrypt(server)

# ---------- Load data (once at startup) ----------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# Normalize columns (safe checks)
if "sentiment_label" not in reviews_df.columns and "sentiment_compound" in reviews_df.columns:
    reviews_df["sentiment_label"] = reviews_df["sentiment_compound"].apply(
        lambda v: "Positive" if v >= 0.05 else ("Negative" if v <= -0.05 else "Neutral")
    )

def verify_user(username: str, password: str):
    row = users_df[users_df["username"] == username]
    if row.empty:
        return None
    pw_hash = str(row.iloc[0]["password_hash"])
    try: 
        ok = bcrypt.check_password_hash(pw_hash, password)
    except Exception:
        ok = (pw_hash == password)
    if not ok:
        return None
    return {
        "username": username,
        "role": row.iloc[0]["role"],
        "business_id": row.iloc[0]["business_id"]
    }

def current_user():
    return session.get("user")

def require_login():
    if not current_user():
        return redirect(url_for("login"))

# ---------- Routes ----------
@server.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = verify_user(username, password)
        if user:
            session["user"] = user
            return redirect("/dashboard/")
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)

@server.route("/logout")
def logout():
    # Completely clear the Flask session
    session.pop("user", None)
    session.clear()
    session.modified = True

    # Create response and disable caching
    response = redirect(url_for("login"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

# ---------- Dash app ----------
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, server=server, url_base_pathname="/dashboard/", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Dropdown options (admin)
business_options = [{"label": n, "value": bid} for bid, n in zip(business_df.get("business_id", []), business_df.get("name", []))]

def header():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="40"), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Restaurant Sentiment Analytics", className="ms-2")),
                ], align="center", className="g-2"),
                href="/dashboard/"
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Logout", href="/logout", external_link=True))
            ], className="ms-auto", navbar=True),
        ]),
        color="primary", dark=True, className="mb-3"
    )

def kpi_card(title, value, color="info"):
    return dbc.Card(
        dbc.CardBody([html.H6(title, className="mb-1"), html.H3(value, className="mb-0")]),
        className="shadow-sm", color=color, inverse=True
    )

app.layout = dbc.Container([
    dcc.Location(id="url"),
    header(),
    html.Div(id="user-context", style={"display":"none"}),

    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Select a Restaurant", className="card-title"),
                dcc.Dropdown(id="business-dropdown", options=business_options, placeholder="Choose a restaurant"),
                html.Div(id="active-business-name", className="mt-2 text-muted")
            ]))
        ], md=4),
        dbc.Col([
            dbc.Row([
                dbc.Col(id="kpi1", md=4),
                dbc.Col(id="kpi2", md=4),
                dbc.Col(id="kpi3", md=4),
            ], id="kpi-row")
        ], md=8),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="trend"), md=7),
        dbc.Col(dcc.Graph(id="pie"), md=5),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="aspect_bar"), md=6),
        dbc.Col(dcc.Graph(id="topic_bar"), md=6),
    ], className="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="keyword_comp"), md=12),
    ], className="mt-3")
], fluid=True)

# ---------- Utility analysis functions ----------
def tokenize(text: str):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [w for w in text.split() if w and w not in STOPWORDS and len(w) > 2]
    return toks

# Simple aspect keyword matching (adjust keywords as needed)
ASPECT_KEYWORDS = {
    "Food": ["food","taste","dish","flavor","menu","meal","bread","steak","cake","coffee","sushi","pizza"],
    "Service": ["service","waiter","server","staff","host","friendly","rude","manager","attentive","helpful"],
    "Ambience": ["ambience","atmosphere","music","noise","decor","environment","cozy","bright","clean","crowded"],
    "Pricing": ["price","cost","expensive","cheap","value","pricey","affordable","bill","charge"]
}

def compute_aspect_stats(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Given reviews for a business, returns DataFrame with aspect, avg_sentiment, review_count"""
    records = []
    for aspect, keys in ASPECT_KEYWORDS.items():
        mask = df_reviews["text"].fillna("").str.lower().apply(lambda t: any(k in t for k in keys))
        subset = df_reviews[mask]
        if len(subset) == 0:
            records.append({"aspect": aspect, "avg_sentiment": np.nan, "review_count": 0})
        else:
            avg = subset["sentiment_compound"].mean() if "sentiment_compound" in subset.columns else subset["sentiment_label"].map({"Positive":1,"Neutral":0,"Negative":-1}).mean()
            records.append({"aspect": aspect, "avg_sentiment": float(avg), "review_count": int(mask.sum())})
    return pd.DataFrame(records)

def top_keywords_by_sentiment(df_reviews: pd.DataFrame, topn=15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two DataFrames: positive keywords and negative keywords (keyword,count)"""
    pos = df_reviews[df_reviews["sentiment_label"]=="Positive"]["text"].fillna("")
    neg = df_reviews[df_reviews["sentiment_label"]=="Negative"]["text"].fillna("")
    def top_from_series(s):
        words = []
        for t in s:
            words += tokenize(t)
        c = Counter(words)
        return pd.DataFrame(c.most_common(topn), columns=["keyword","count"])
    return top_from_series(pos), top_from_series(neg)

def compute_keyword_sentiment(df_reviews: pd.DataFrame, topn: int = 12) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      - keyword: word
      - count: number of reviews that mention the word
      - avg_sentiment: average sentiment_compound for reviews that mention it (float)
      - sample_review: one short example review that contains the word
    """
    texts = df_reviews["text"].fillna("").astype(str)
    if texts.empty:
        return pd.DataFrame(columns=["keyword", "count", "avg_sentiment", "sample_review"])

    # Build token counts across corpus
    word_counts = Counter()
    docs_with_word = {}
    for idx, t in texts.items():
        toks = tokenize(t)
        uniq = set(toks)
        for w in uniq:
            word_counts[w] += 1
            docs_with_word.setdefault(w, []).append(idx)

    if not word_counts:
        return pd.DataFrame(columns=["keyword", "count", "avg_sentiment", "sample_review"])

    # Take top N by document frequency
    top_words = [w for w, c in word_counts.most_common(topn)]

    rows = []
    for w in top_words:
        indices = docs_with_word.get(w, [])
        count = len(indices)
        if count == 0:
            avg_sent = None
            sample = ""
        else:
            if "sentiment_compound" in df_reviews.columns:
                avg_sent = float(df_reviews.loc[indices, "sentiment_compound"].mean())
            else:
                avg_sent = float(df_reviews.loc[indices, "sentiment_label"].map({"Positive":1,"Neutral":0,"Negative":-1}).mean())
            sample_text = df_reviews.loc[indices, "text"].dropna().iloc[0]
            sample = (sample_text[:180] + "...") if len(sample_text) > 180 else sample_text

        rows.append({
            "keyword": w,
            "count": int(count),
            "avg_sentiment": float(avg_sent) if avg_sent is not None else None,
            "sample_review": sample
        })

    df_kw = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df_kw

# ---------- Callbacks ----------
@app.callback(
    Output("user-context", "children"),
    Input("url", "pathname"),
    prevent_initial_call=False
)
def load_user_ctx(_):
    user = current_user()
    if not user:
        return ""
    return f"{user['role']}|{user.get('business_id') or ''}|{user['username']}"

@app.callback(
    [Output("business-dropdown", "options"),
     Output("business-dropdown", "value"),
     Output("active-business-name", "children")],
    Input("user-context", "children")
)
def set_scope(ctx):
    if not ctx:
        return business_options, None, "Please log in"
    role, bid, username = ctx.split("|")
    if role == "admin" or role == "Admin":
        return business_options, (business_options[0]["value"] if business_options else None), "Admin mode"
    if bid and bid in business_df.get("business_id", []).tolist():
        name = business_df.loc[business_df["business_id"]==bid, "name"].iloc[0]
        return [{"label": name, "value": bid}], bid, f"Logged in as: {name}"
    return business_options, None, "Please select"

@app.callback(
    [Output("trend", "figure"),
     Output("pie", "figure"),
     Output("aspect_bar", "figure"),
     Output("topic_bar", "figure"),
     Output("keyword_comp", "figure"),
     Output("kpi1", "children"),
     Output("kpi2", "children"),
     Output("kpi3", "children")],
    [Input("business-dropdown", "value"),
     Input("user-context", "children")]
)
def update_charts(selected_bid, ctx):
    if not ctx:
        empty = px.scatter(title="Please log in.")
        return empty, empty, empty, empty, empty, kpi_card("Avg Sentiment (30d)", "—"), kpi_card("Avg Stars (30d)","—"), kpi_card("Reviews (30d)","—")

    role, user_bid, _ = ctx.split("|")
    bid = selected_bid if role.lower()=="admin" else (user_bid or selected_bid)

    if not bid:
        empty = px.scatter(title="No business selected.")
        return empty, empty, empty, empty, empty, kpi_card("Avg Sentiment (30d)", "—"), kpi_card("Avg Stars (30d)","—"), kpi_card("Reviews (30d)","—")

    # Filter data
    trend = daily_df[daily_df["business_id"] == bid].copy() if not daily_df.empty else pd.DataFrame()
    if not trend.empty and "date" in trend.columns:
        trend["date"] = pd.to_datetime(trend["date"], errors="coerce")
    reviews = reviews_df[reviews_df["business_id"] == bid].copy() if not reviews_df.empty else pd.DataFrame()

    # KPIs (last 30 days)
    if not trend.empty and "date" in trend.columns:
        maxd = trend["date"].max()
        last30 = trend[trend["date"] >= (maxd - pd.Timedelta(days=30))]
        k1 = f"{last30['avg_sentiment'].mean():.2f}" if not last30.empty else "—"
        k2 = f"{last30['avg_stars'].mean():.2f}" if not last30.empty else "—"
        k3 = str(int(last30['review_count'].sum())) if not last30.empty else "0"
    else:
        k1=k2="—"; k3="0"

    # Trend figure
    if not trend.empty and "date" in trend.columns:
        fig_trend = px.line(trend, x="date", y="avg_sentiment", title=f"Average Sentiment Over Time", markers=True)
        if "avg_stars" in trend.columns:
            fig_trend.add_scatter(x=trend["date"], y=trend["avg_stars"], mode="lines+markers", name="Average Stars", line=dict(color="orange"))
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Score", legend=dict(x=0.85,y=0.95))
    else:
        fig_trend = px.scatter(title="No trend data found.")
    
    fig_trend.update_traces(line_shape="spline")

    # Pie chart
    if not reviews.empty and "sentiment_label" in reviews.columns:
        counts = reviews["sentiment_label"].value_counts().reset_index()
        counts.columns = ["sentiment","count"]
        fig_pie = px.pie(counts, names="sentiment", values="count", title="Sentiment Distribution",
                         color="sentiment", color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"})
    else:
        fig_pie = px.scatter(title="No sentiment labels found.")

    # Aspect-based sentiment
    if not reviews.empty:
        aspect_df = compute_aspect_stats(reviews)
        if aspect_df["review_count"].sum() == 0:
            fig_aspect = px.scatter(title="No aspect matches found in reviews.")
        else:
            fig_aspect = px.bar(aspect_df, x="aspect", y="avg_sentiment", color="aspect", text="review_count",
                                title="Aspect-Based Sentiment")
            fig_aspect.update_traces(textposition='outside')
            fig_aspect.update_layout(yaxis_title="Average Sentiment", showlegend=False)
    else:
        fig_aspect = px.scatter(title="No reviews for this business.")

    if not reviews.empty:
        kw_df = compute_keyword_sentiment(reviews, topn=12)
        if not kw_df.empty:
            kw_df_plot = kw_df.sort_values("count", ascending=False)

            fig_topic = px.bar(
                kw_df_plot,
                x="keyword",
                y="avg_sentiment",
                text="count",
                title="Keyword-level Sentiment",
                labels={"avg_sentiment": "Avg Sentiment", "keyword": "Keyword"}
            )

            fig_topic.update_traces(textposition="outside",
                                    hovertemplate="<b>%{x}</b><br>Avg sentiment: %{y:.2f}<br>Mentions: %{text}<br>Example: %{customdata[0]}",
                                    customdata=kw_df_plot[["sample_review"]].values)

            fig_topic.update_yaxes(range=[-0.25, 1], title_text="Avg Sentiment")

            fig_topic.update_layout(
                xaxis_tickangle=-30,
                uniformtext_minsize=10,
                margin=dict(t=70, b=140),
                showlegend=False
            )
        else:
            fig_topic = px.scatter(title="No repeated keywords found.")
    else:
        fig_topic = px.scatter(title="No reviews for this business.")


    # Keyword comparison
    if not reviews.empty and "sentiment_label" in reviews.columns:
        pos_df, neg_df = top_keywords_by_sentiment(reviews, topn=12)
        all_keys = list(dict.fromkeys(list(pos_df["keyword"]) + list(neg_df["keyword"])))
        pos_map = dict(zip(pos_df["keyword"], pos_df["count"]))
        neg_map = dict(zip(neg_df["keyword"], neg_df["count"]))
        pos_vals = [pos_map.get(k, 0) for k in all_keys]
        neg_vals = [neg_map.get(k, 0) for k in all_keys]
        fig_kw = go.Figure()
        fig_kw.add_trace(go.Bar(x=all_keys, y=pos_vals, name="Positive"))
        fig_kw.add_trace(go.Bar(x=all_keys, y=neg_vals, name="Negative"))
        fig_kw.update_layout(barmode='group', title="Top Keywords: Positive vs Negative Reviews",
                             xaxis_title="Keyword", yaxis_title="Count")
    else:
        fig_kw = px.scatter(title="Insufficient review sentiment data for keyword comparison.")

    # KPIs as cards
    k1_card = kpi_card("Avg Sentiment (30d)", k1, "success")
    k2_card = kpi_card("Avg Stars (30d)", k2, "warning")
    k3_card = kpi_card("Reviews (30d)", k3, "secondary")

    return fig_trend, fig_pie, fig_aspect, fig_topic, fig_kw, k1_card, k2_card, k3_card

# ---------- Protect dash: redirect to login when necessary ----------
@server.before_request
def protect_dash():
    path = request.path
    if path.startswith("/dashboard"):
        if not current_user():
            return redirect(url_for("login"))

# ---------- Run locally ----------
if __name__ == "__main__":

    @server.before_request
    def protect_dash():
        path = request.path
        if path.startswith("/dashboard") or path.startswith("/_dash"):
            if not current_user():
                return redirect(url_for("login"))

    # Disable all caching after each request
    @server.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # Auto-open the login page in browser
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050/")
    app.run(host="127.0.0.1", port=8050, debug=True)
