import os
import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for
from flask_bcrypt import Bcrypt
from dash import Dash, dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

# ---------- Flask base ----------
server = Flask(__name__, template_folder="templates", static_folder="assets")
server.secret_key = os.environ.get("APP_SECRET_KEY", "dev_secret_change_me")
bcrypt = Bcrypt(server)

# ---------- Load data once ----------
business_df = pd.read_csv("/Users/shashankmidididdi/Downloads/business.csv")
reviews_df = pd.read_csv("reviews_with_sentiment.csv")   # must have sentiment_label, sentiment_compound
daily_df = pd.read_csv("agg_sentiment_daily.csv")
topwords_df = pd.read_csv("agg_top_words_by_category.csv")
users_df = pd.read_csv("user_access.csv")

# ---------- Helpers ----------
def verify_user(username: str, password: str):
    row = users_df[users_df["username"] == username]
    if row.empty:
        return None
    pw_hash = str(row.iloc[0]["password_hash"])
    try: 
        ok = bcrypt.check_password_hash(pw_hash, password)
    except Exception:
        ok = (pw_hash == password)  # fallback if you temporarily use plaintext (not recommended)
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
external_stylesheets = [dbc.themes.FLATLY]  # try CYBORG, MINTY, etc.
app = Dash(
    __name__,
    server=server,
    url_base_pathname="/dashboard/",
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

# Dropdown options (for admins)
admin_options = [
    {"label": name, "value": bid}
    for bid, name in zip(business_df["business_id"], business_df["name"])
]

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

# Layout
app.layout = dbc.Container([
    dcc.Location(id="url"),
    header(),

    # This invisible div holds the current user's business_id / role pulled from the Flask session
    html.Div(id="user-context", style={"display": "none"}),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5("Select a Restaurant", className="card-title"),
                    dcc.Dropdown(id="business-dropdown", options=admin_options, placeholder="Choose a restaurant"),
                    html.Div(id="active-business-name", className="mt-2 text-muted")
                ])
            )
        ], md=4),
        dbc.Col([
            dbc.Row([
                dbc.Col(kpi_card("Avg Sentiment (30d)", "—", "success"), md=4),
                dbc.Col(kpi_card("Avg Stars (30d)", "—", "warning"), md=4),
                dbc.Col(kpi_card("Reviews (30d)", "—", "secondary"), md=4),
            ], id="kpi-row")
        ], md=8),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="trend"), md=7),
        dbc.Col(dcc.Graph(id="pie"), md=5),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="topwords"), md=12),
    ], className="mt-3"),

], fluid=True)

# ---------- Callbacks ----------
@app.callback(
    Output("user-context", "children"),
    Input("url", "pathname"),
    prevent_initial_call=False
)
def load_user(_):
    # transfer Flask session -> Dash
    user = current_user()
    if not user:
        # if not logged in, send redirect (Dash can't redirect directly; Flask route handles)
        return ""
    return f"{user['role']}|{user['business_id']}|{user['username']}"

@app.callback(
    [Output("business-dropdown", "options"),
     Output("business-dropdown", "value"),
     Output("active-business-name", "children")],
    Input("user-context", "children")
)
def set_business_scope(ctx):
    if not ctx:
        return no_update, no_update, "Please log in."
    role, bid, _ = ctx.split("|")
    if role == "admin":
        # admin can choose any business
        return admin_options, (admin_options[0]["value"] if admin_options else None), "Admin mode"
    # regular user binds to their business; hide dropdown choices
    name = business_df.loc[business_df["business_id"] == bid, "name"].iloc[0]
    return [{"label": name, "value": bid}], bid, f"Logged in as: {name}"

@app.callback(
    [Output("trend", "figure"),
     Output("pie", "figure"),
     Output("topwords", "figure"),
     Output("kpi-row", "children")],
    [Input("business-dropdown", "value"),
     Input("user-context", "children")]
)
def update_charts(selected_bid, ctx):
    # If not logged in / no context, return placeholders
    if not ctx:
        empty = px.scatter(title="Please log in.")
        kpis = [
            dbc.Col(kpi_card("Avg Sentiment (30d)", "—"), md=4),
            dbc.Col(kpi_card("Avg Stars (30d)", "—"), md=4),
            dbc.Col(kpi_card("Reviews (30d)", "—"), md=4),
        ]
        return empty, empty, empty, kpis

    role, user_bid, _ = ctx.split("|")
    bid = selected_bid if role == "admin" else user_bid
    if not bid:
        empty = px.scatter(title="No business selected.")
        kpis = [
            dbc.Col(kpi_card("Avg Sentiment (30d)", "—"), md=4),
            dbc.Col(kpi_card("Avg Stars (30d)", "—"), md=4),
            dbc.Col(kpi_card("Reviews (30d)", "—"), md=4),
        ]
        return empty, empty, empty, kpis

    # Data slices (trend from daily_df, reviews from reviews_df)
    trend = daily_df[daily_df["business_id"] == bid].copy() if "business_id" in daily_df.columns else daily_df.copy()
    if "date" in trend.columns:
        trend["date"] = pd.to_datetime(trend["date"], errors="coerce")

    revs = reviews_df[reviews_df["business_id"] == bid].copy() if "business_id" in reviews_df.columns else reviews_df.copy()
    name = business_df.loc[business_df["business_id"] == bid, "name"].iloc[0] if ('business_id' in business_df.columns and (business_df["business_id"] == bid).any()) else str(bid)

    # KPIs (last 30 days)
    if not trend.empty and "date" in trend.columns:
        last30 = trend[trend["date"] >= (trend["date"].max() - pd.Timedelta(days=30))]
        k1 = f"{last30['avg_sentiment'].mean():.2f}" if not last30.empty and "avg_sentiment" in last30.columns else "—"
        k2 = f"{last30['avg_stars'].mean():.2f}" if not last30.empty and "avg_stars" in last30.columns else "—"
        k3 = f"{int(last30['review_count'].sum())}" if not last30.empty and "review_count" in last30.columns else "0"
    else:
        k1 = k2 = "—"; k3 = "0"

    kpi_children = [
        dbc.Col(kpi_card("Avg Sentiment (30d)", k1, "success"), md=4),
        dbc.Col(kpi_card("Avg Stars (30d)", k2, "warning"), md=4),
        dbc.Col(kpi_card("Reviews (30d)", k3, "secondary"), md=4),
    ]

    # ---------------- Trend chart (smoothed) ----------------
    if not trend.empty and "date" in trend.columns:
        # Ensure sorted by date
        df_ts = trend.sort_values("date")

        # Base line for sentiment compound
        fig_trend = px.line(
            df_ts,
            x="date",
            y="avg_sentiment" if "avg_sentiment" in df_ts.columns else "sentiment_compound",
            title=f"Average Sentiment Over Time — {name}"
        )

        # Apply spline to all existing traces (this will smooth the first trace)
        fig_trend.update_traces(line_shape="spline")

        # If avg_stars exists, add it as a smoothed trace
        if "avg_stars" in df_ts.columns:
            fig_trend.add_scatter(
                x=df_ts["date"],
                y=df_ts["avg_stars"],
                mode="lines",
                name="Average Stars",
                line=dict(shape="spline")
            )

        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Score", hovermode="x unified")
    else:
        fig_trend = px.scatter(title="No trend data found.")

    # ---------------- Pie chart ----------------
    if "sentiment_label" in revs.columns and not revs.empty:
        counts = revs["sentiment_label"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(counts, names="sentiment", values="count",
                         color="sentiment",
                         color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"},
                         title="Sentiment Distribution")
    else:
        fig_pie = px.scatter(title="No sentiment labels found.")

    # ---------------- Top words ----------------
    cat = business_df.loc[business_df["business_id"] == bid, "categories"] if ('business_id' in business_df.columns and (business_df["business_id"] == bid).any()) else pd.Series([], dtype=object)
    cat = str(cat.iloc[0]).split(",")[0].strip() if not cat.empty else ""
    words = topwords_df[topwords_df["category"] == cat] if "category" in topwords_df.columns else pd.DataFrame()
    if not words.empty:
        fig_words = px.bar(words.sort_values("count", ascending=False),
                           x="keyword", y="count",
                           title=f"Top Keywords in Category ({cat})", text_auto=True)
    else:
        fig_words = px.scatter(title=f"No keywords found for category '{cat}'")

    # Return all outputs (must be inside the function)
    return fig_trend, fig_pie, fig_words, kpi_children


# ---------- Run locally ----------
if __name__ == "__main__":

    # Protect /dashboard from unauthenticated users
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
