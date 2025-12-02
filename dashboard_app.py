import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


reviews_df = pd.read_csv("/Users/shashankmidididdi/Downloads/Applied_Project/reviews_with_sentiment.csv")
business_df = pd.read_csv("/Users/shashankmidididdi/Downloads/business.csv")
daily_df = pd.read_csv("/Users/shashankmidididdi/Downloads/Applied_Project/agg_sentiment_daily.csv")
topwords_df = pd.read_csv("/Users/shashankmidididdi/Downloads/Applied_Project/agg_top_words_by_category.csv")


business_options = [
    {"label": name, "value": bid}
    for bid, name in zip(business_df["business_id"], business_df["name"])
]

app = Dash(__name__)

app.layout = html.Div([
    html.H1("🍽️ Restaurant Sentiment Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select a Restaurant:"),
        dcc.Dropdown(
            id="business_dropdown",
            options=business_options,
            value=business_df["business_id"].iloc[0],
            style={"width": "60%"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div([
        dcc.Graph(id="sentiment_trend"),
        dcc.Graph(id="sentiment_pie"),
        dcc.Graph(id="top_words_bar")
    ])
])

@app.callback(
    [Output("sentiment_trend", "figure"),
     Output("sentiment_pie", "figure"),
     Output("top_words_bar", "figure")],
    [Input("business_dropdown", "value")]
)
def update_dashboard(selected_business_id):

    df_trend = daily_df[daily_df["business_id"] == selected_business_id]
    if df_trend.empty:
        fig_empty = px.scatter(title="⚠️ No data available for this restaurant")
        return fig_empty, fig_empty, fig_empty


    df_reviews = reviews_df[reviews_df["business_id"] == selected_business_id]


    business_row = business_df[business_df["business_id"] == selected_business_id]
    category_name = ""
    if not business_row.empty:
        cats = str(business_row["categories"].iloc[0])
        category_name = cats.split(",")[0].strip() if cats else "Unknown"
    df_words = topwords_df[topwords_df["category"] == category_name]


    fig_trend = px.line(df_trend, x="date", y="avg_sentiment",
                        title=f"Average Sentiment Over Time — {business_row['name'].iloc[0]}",
                        markers=True)
    fig_trend.add_scatter(x=df_trend["date"], y=df_trend["avg_stars"],
                          mode="lines+markers", name="Average Stars",
                          line=dict(color="orange"))
    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Score")


    if "sentiment_label" in df_reviews.columns:
        sentiment_counts = df_reviews["sentiment_label"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(sentiment_counts, names="sentiment", values="count",
                         color="sentiment",
                         color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
                         title="Sentiment Distribution")
    else:
        fig_pie = px.scatter(title="⚠️ Sentiment label not found in reviews data")


    if not df_words.empty:
        fig_bar = px.bar(df_words.sort_values(by="count", ascending=False),
                         x="keyword", y="count",
                         title=f"Top Keywords in Category ({category_name})",
                         text_auto=True)
    else:
        fig_bar = px.scatter(title=f"⚠️ No top words for category '{category_name}'")

    return fig_trend, fig_pie, fig_bar



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
