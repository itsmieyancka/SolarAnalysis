import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ============================================================
# 1. SETTINGS AND COLOURS
# ============================================================

CSV_FILE = "sunvolt_clean.csv"

NAVY = "#172B4D"
BLUE = "#3975B9"
GOLD = "#D6A743"
BACKGROUND = "#F3F5F8"
WHITE = "#FFFFFF"
TEXT = "#344054"
MUTED = "#7B8794"
BORDER = "#E5EAF0"

# ============================================================
# 2. LOAD AND CLEAN THE DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

required_columns = [
    "Customer ID", "Region", "Age Group",
    "Distance from Nearest Installer", "Income Band",
    "Marital Status", "Household Dependents", "Purchase Flag"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Remove duplicate customers
df = df.drop_duplicates(subset="Customer ID").copy()

# Purchase flag must be 0 or 1
df["Purchase Flag"] = pd.to_numeric(df["Purchase Flag"], errors="coerce")
df = df.dropna(subset=["Purchase Flag"]).copy()

if not df["Purchase Flag"].isin([0, 1]).all():
    raise ValueError("Purchase Flag must contain only 0 and 1.")

# Dependents must be a whole number
df["Household Dependents"] = pd.to_numeric(df["Household Dependents"], errors="coerce")
df = df.dropna(subset=["Household Dependents"]).copy()
df["Household Dependents"] = df["Household Dependents"].astype(int)

# ============================================================
# 3. CALCULATIONS
# ============================================================

def purchase_rate(data):
    if len(data) == 0:
        return 0
    return data["Purchase Flag"].mean() * 100


def group_summary(data, column):
    summary = (
        data.groupby(column, observed=True)
        .agg(
            Customers=("Purchase Flag", "count"),
            Purchases=("Purchase Flag", "sum"),
            Purchase_Rate=("Purchase Flag", "mean")
        )
        .reset_index()
    )
    summary["Purchase_Rate"] *= 100
    return summary


def get_highest_group(data, column):
    summary = group_summary(data, column)
    best = summary.loc[summary["Purchase_Rate"].idxmax()]
    return str(best[column]), best["Purchase_Rate"]


def get_lowest_group(data, column):
    summary = group_summary(data, column)
    worst = summary.loc[summary["Purchase_Rate"].idxmin()]
    return str(worst[column]), worst["Purchase_Rate"]


company_rate = purchase_rate(df)
total_customers = len(df)
total_purchases = int(df["Purchase Flag"].sum())
total_non_purchasers = total_customers - total_purchases

# ============================================================
# 4. CATEGORY ORDER
# ============================================================

PREFERRED_ORDER = {
    "Region": ["North America", "Europe", "Pacific"],
    "Age Group": ["Under 30", "30-39", "30–39", "40-49", "40–49",
                  "50-59", "50–59", "60+"],
    "Distance from Nearest Installer": ["0-1 km", "0–1 km", "1-2 km", "1–2 km",
                                        "2-5 km", "2–5 km", "5-10 km", "5–10 km",
                                        "10+ km"],
    "Marital Status": ["Single", "Married"],
    "Income Band": ["Under 30k", "Below 30k", "30k-49k", "30k–49k",
                    "50k-69k", "50k–69k", "70k-99k", "70k–99k", "100k+"]
}


def get_category_order(data, column):
    values = data[column].dropna().unique().tolist()

    if column == "Household Dependents":
        return sorted(values)

    if column in PREFERRED_ORDER:
        ordered = [v for v in PREFERRED_ORDER[column] if v in values]
        # Add any categories that are not in the preferred list
        for v in values:
            if v not in ordered:
                ordered.append(v)
        return ordered

    return sorted(values)

# ============================================================
# 5. BAR CHART
# ============================================================

def create_bar_chart(data, column, title):
    summary = group_summary(data, column)

    if summary.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[{"text": "No data available", "showarrow": False}]
        )
        return fig

    order = get_category_order(data, column)
    summary[column] = pd.Categorical(summary[column], categories=order, ordered=True)
    summary = summary.sort_values(column)

    highest_rate = summary["Purchase_Rate"].max()

    bar_colours = [
        GOLD if value == highest_rate else BLUE
        for value in summary["Purchase_Rate"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=summary["Purchase_Rate"],
            y=summary[column].astype(str),
            orientation="h",
            marker=dict(color=bar_colours, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in summary["Purchase_Rate"]],
            textposition="outside",
            cliponaxis=False,
            customdata=summary[["Customers", "Purchases"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Purchase rate: %{x:.1f}%<br>"
                "Customers: %{customdata[0]:,.0f}<br>"
                "Purchases: %{customdata[1]:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY), x=0.02, xanchor="left"),
        template="plotly_white",
        height=310,
        margin=dict(l=10, r=55, t=55, b=35),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Arial", color=TEXT, size=11),
        xaxis=dict(
            title="Purchase Rate (%)",
            range=[0, min(110, max(65, highest_rate + 15))],
            showgrid=False,
            zeroline=False,
            ticksuffix="%"
        ),
        yaxis=dict(title="", showgrid=False, autorange="reversed"),
        showlegend=False,
        bargap=0.35
    )

    return fig

# ============================================================
# 6. RECOMMENDATIONS
# ============================================================

region_name, region_rate = get_highest_group(df, "Region")
age_name, age_rate = get_highest_group(df, "Age Group")
distance_name, distance_rate = get_lowest_group(df, "Distance from Nearest Installer")
income_name, income_rate = get_lowest_group(df, "Income Band")
marital_name, marital_rate = get_highest_group(df, "Marital Status")
dependent_name, dependent_rate = get_lowest_group(df, "Household Dependents")

recommendations = [
    {
        "heading": "Strengthen regional conversion",
        "finding": f"{region_name} has the highest regional purchase rate at {region_rate:.1f}%.",
        "recommendation": "Use regional customer insights to improve conversion, "
                          "particularly in markets with lower purchase rates.",
        "actions": [
            "Review barriers affecting lower-performing regions.",
            "Adapt marketing campaigns to local customer needs."
        ]
    },
    {
        "heading": "Target high-performing age groups",
        "finding": f"{age_name} customers have the highest purchase rate at {age_rate:.1f}%.",
        "recommendation": "Prioritise customer segments showing stronger interest "
                          "while continuing to engage other age groups.",
        "actions": [
            "Tailor campaigns to the strongest age segment.",
            "Use targeted information to address customer concerns."
        ]
    },
    {
        "heading": "Improve installer accessibility",
        "finding": f"Customers in the {distance_name} category have the lowest "
                   f"purchase rate at {distance_rate:.1f}%.",
        "recommendation": "Investigate whether installer distance is creating "
                          "access or convenience barriers.",
        "actions": [
            "Explore partnerships with installers in underserved areas.",
            "Consider remote consultations and flexible installation options."
        ]
    },
    {
        "heading": "Explore more affordable options",
        "finding": f"The {income_name} income band has the lowest purchase rate "
                   f"at {income_rate:.1f}%.",
        "recommendation": "Explore pricing and financing options that could make "
                          "solar backup systems more accessible.",
        "actions": [
            "Consider suitable instalment or financing plans.",
            "Communicate product value and long-term savings clearly."
        ]
    },
    {
        "heading": "Adapt customer marketing messages",
        "finding": f"{marital_name} customers have the highest purchase rate "
                   f"among marital-status groups at {marital_rate:.1f}%.",
        "recommendation": "Test different marketing messages for household groups "
                          "rather than relying on one general campaign.",
        "actions": [
            "Test household-focused product messaging.",
            "Compare campaign performance across customer groups."
        ]
    },
    {
        "heading": "Develop flexible household packages",
        "finding": f"Households with {dependent_name} dependents have the lowest "
                   f"purchase rate at {dependent_rate:.1f}%.",
        "recommendation": "Explore whether flexible packages and clearer pricing "
                          "can better meet different household needs.",
        "actions": [
            "Review package options for different household sizes.",
            "Test flexible payment and system configurations."
        ]
    }
]

chart_details = [
    ("Region", "Purchase Rate by Region"),
    ("Age Group", "Purchase Rate by Age Group"),
    ("Distance from Nearest Installer", "Purchase Rate by Installer Distance"),
    ("Income Band", "Purchase Rate by Income Band"),
    ("Marital Status", "Purchase Rate by Marital Status"),
    ("Household Dependents", "Purchase Rate by Number of Dependents")
]

# ============================================================
# 7. STYLES
# ============================================================

card_style = {
    "backgroundColor": WHITE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "18px",
}

label_style = {
    "color": MUTED,
    "fontSize": "11px",
    "fontWeight": "700",
    "letterSpacing": "0.5px",
    "marginTop": "10px",
    "marginBottom": "4px",
}

text_style = {"color": TEXT, "fontSize": "13px", "lineHeight": "1.5"}

# ============================================================
# 8. LAYOUT HELPERS
# ============================================================

def make_kpi_card(label, value, note):
    return html.Div(
        style={**card_style, "flex": "1", "minHeight": "112px"},
        children=[
            html.Div(label, style={"color": MUTED, "fontSize": "13px", "marginBottom": "8px"}),
            html.Div(value, style={"color": NAVY, "fontSize": "27px", "fontWeight": "700"}),
            html.Div(note, style={"color": MUTED, "fontSize": "11px", "marginTop": "4px"}),
        ]
    )


def make_insight_card(rec):
    return html.Div(
        style={**card_style, "borderLeft": f"5px solid {GOLD}", "minHeight": "330px"},
        children=[
            html.Div(rec["heading"], style={"color": NAVY, "fontSize": "16px",
                                            "fontWeight": "700", "marginBottom": "13px"}),
            html.Div("DATA FINDING", style=label_style),
            html.Div(rec["finding"], style=text_style),
            html.Div("RECOMMENDATION", style=label_style),
            html.Div(rec["recommendation"], style=text_style),
            html.Div("SUGGESTED ACTIONS", style=label_style),
            html.Ul([html.Li(a) for a in rec["actions"]],
                    style={**text_style, "paddingLeft": "18px", "marginTop": "5px"}),
        ]
    )


# One row per recommendation: card on the left, chart on the right
rows = []
for i, rec in enumerate(recommendations):
    rows.append(
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
            children=[
                html.Div(make_insight_card(rec), style={"flex": "0.95"}),
                html.Div(
                    dcc.Graph(id=f"chart-{i}", config={"displayModeBar": False}),
                    style={**card_style, "flex": "1.15", "padding": "12px 12px 2px 12px"}
                ),
            ]
        )
    )

kpi_data = [
    ("CUSTOMERS ANALYSED", f"{total_customers:,}", "Unique customer records"),
    ("PURCHASES", f"{total_purchases:,}", "Customers who purchased"),
    ("NON-PURCHASERS", f"{total_non_purchasers:,}", "Potential conversion opportunities"),
    ("COMPANY PURCHASE RATE", f"{company_rate:.1f}%", "Overall purchase rate"),
]

regions = ["All Regions"] + sorted(df["Region"].dropna().unique().tolist())

# ============================================================
# 9. APP LAYOUT
# ============================================================

app = Dash(__name__)
app.title = "SunVolt Energy Dashboard"

app.layout = html.Div(
    style={"display": "flex", "minHeight": "100vh", "backgroundColor": BACKGROUND,
           "fontFamily": "Arial, sans-serif"},
    children=[

        # ---------- Sidebar ----------
        html.Div(
            style={"width": "260px", "backgroundColor": WHITE, "padding": "25px",
                   "borderRight": f"1px solid {BORDER}"},
            children=[
                html.H2("Dashboard Filters", style={"color": NAVY, "marginTop": "0"}),
                html.P("Select a region to explore customer purchase behaviour.",
                       style={"color": TEXT, "fontSize": "14px"}),
                dcc.Dropdown(id="region-dropdown", options=regions,
                             value="All Regions", clearable=False),
                html.Hr(style={"margin": "25px 0"}),
                html.Div(id="sidebar-stats", style={"color": TEXT, "fontSize": "14px"}),
                html.P("The region filter updates all six charts. The KPI cards "
                       "and recommendations show overall company findings.",
                       style={"color": MUTED, "fontSize": "12px", "marginTop": "20px"}),
            ]
        ),

        # ---------- Main content ----------
        html.Div(
            style={"flex": "1", "padding": "30px", "maxWidth": "1500px"},
            children=[

                # Header
                html.Div(
                    style={"backgroundColor": NAVY, "padding": "25px 30px",
                           "borderRadius": "12px", "marginBottom": "22px"},
                    children=[
                        html.Div("SunVolt Energy | Customer Purchase Analysis",
                                 style={"color": "white", "fontSize": "28px",
                                        "fontWeight": "700", "marginBottom": "5px"}),
                        html.Div("International Home Solar Backup System Analysis",
                                 style={"color": "#DCE5F0", "fontSize": "14px"}),
                        html.Div("Customer insights, purchase behaviour and business recommendations",
                                 style={"color": "#DCE5F0", "fontSize": "14px"}),
                    ]
                ),

                # KPI cards
                html.Div("Business Overview",
                         style={"color": NAVY, "fontSize": "20px", "fontWeight": "700"}),
                html.Div("Overall customer and purchase performance",
                         style={"color": MUTED, "fontSize": "13px", "marginBottom": "18px"}),
                html.Div(
                    [make_kpi_card(*k) for k in kpi_data],
                    style={"display": "flex", "gap": "16px"}
                ),

                # Insights and charts
                html.Div("Customer Insights & Business Actions",
                         style={"color": NAVY, "fontSize": "20px", "fontWeight": "700",
                                "marginTop": "28px"}),
                html.Div("Each recommendation is paired with the chart supporting the finding. "
                         "Use the region filter to explore the selected market.",
                         style={"color": MUTED, "fontSize": "13px", "marginBottom": "18px"}),
                html.Div(rows),

                # Footer
                html.Div(
                    [
                        html.Div("SunVolt Energy | Customer Purchase Analysis"),
                        html.Div("Data-driven insights to support strategic business decisions"),
                    ],
                    style={"color": MUTED, "fontSize": "11px", "textAlign": "center",
                           "marginTop": "30px"}
                ),
            ]
        ),
    ]
)

# ============================================================
# 10. CALLBACK (region filter updates the six charts + sidebar)
# ============================================================

@app.callback(
    [Output(f"chart-{i}", "figure") for i in range(6)]
    + [Output("sidebar-stats", "children")],
    Input("region-dropdown", "value")
)
def update_dashboard(selected_region):

    if selected_region == "All Regions":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["Region"] == selected_region].copy()

    figures = [
        create_bar_chart(filtered_df, column, title)
        for column, title in chart_details
    ]

    stats = [
        html.P([html.B("Customers displayed: "), f"{len(filtered_df):,}"]),
        html.P([html.B("Purchase rate: "), f"{purchase_rate(filtered_df):.1f}%"]),
    ]

    return figures + [stats]


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=False)