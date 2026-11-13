import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ============================================================
# 1. LOAD AND CHECK THE DATA
# ============================================================

CSV_FILE = "sunvolt_clean.csv"
df = pd.read_csv(CSV_FILE)

required_columns = [
    "Customer ID",
    "Region",
    "Age Group",
    "Distance from Nearest Installer",
    "Income Band",
    "Marital Status",
    "Household Dependents",
    "Purchase Flag"
]

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The CSV file is missing these columns: "
        + ", ".join(missing_columns)
    )

# Keep one record per customer.
df = df.drop_duplicates(subset="Customer ID").copy()

# Make sure the purchase flag is numeric.
df["Purchase Flag"] = pd.to_numeric(
    df["Purchase Flag"], errors="coerce"
)

df = df.dropna(subset=["Purchase Flag"]).copy()
df["Purchase Flag"] = df["Purchase Flag"].astype(int)

# Check that the purchase flag contains only 0 and 1.
if not df["Purchase Flag"].isin([0, 1]).all():
    raise ValueError(
        "Purchase Flag must contain only 0 (No) and 1 (Yes)."
    )

# Convert dependents to numbers.
# Do not remove customers with missing dependent values.
df["Household Dependents"] = pd.to_numeric(
    df["Household Dependents"], errors="coerce"
)

if df.empty:
    raise ValueError(
        "There are no usable customer records in the CSV file."
    )


def purchase_rate(data):
    """Return the percentage of customers who purchased."""
    if len(data) == 0:
        return 0.0

    return data["Purchase Flag"].mean() * 100


def estimated_extra_purchases(
    group_size, group_rate, comparison_rate
):
    """Estimate the difference if a group reached another group's rate."""
    return max(
        0,
        group_size * (comparison_rate - group_rate) / 100
    )


# Company-wide benchmark.
# This stays the same when the region filter changes.
company_rate = purchase_rate(df)


# ============================================================
# 2. CALCULATE BUSINESS FINDINGS FROM THE FULL DATASET
# ============================================================

# Regional comparison
north_america = df[df["Region"] == "North America"]
europe = df[df["Region"] == "Europe"]

na_customers = len(north_america)
na_rate = purchase_rate(north_america)
europe_rate = purchase_rate(europe)

na_non_purchasers = (
    na_customers - int(north_america["Purchase Flag"].sum())
)

na_estimate = estimated_extra_purchases(
    na_customers, na_rate, europe_rate
)


# Age groups
age_rates = df.groupby("Age Group")["Purchase Flag"].mean() * 100

if age_rates.empty:
    best_age = "Not available"
    best_age_rate = 0.0
else:
    best_age = age_rates.idxmax()
    best_age_rate = float(age_rates.max())

over60_rate = float(age_rates.get("60+", 0))


# Installer distance:
# Compare customers 5 km or further with customers under 5 km.
far_distances = ["5-10 km", "10+ km"]
near_distances = ["0-1 km", "1-2 km", "2-5 km"]

far_customers_data = df[
    df["Distance from Nearest Installer"].isin(far_distances)
]

near_customers_data = df[
    df["Distance from Nearest Installer"].isin(near_distances)
]

far_customers = len(far_customers_data)

far_rate = purchase_rate(far_customers_data)
near_rate = purchase_rate(near_customers_data)

far_estimate = estimated_extra_purchases(
    far_customers, far_rate, near_rate
)


# Married households with 4 or more dependents.
# Only records with a known dependent value are used here.
married_data = df[
    (df["Marital Status"] == "Married")
    & (df["Household Dependents"].notna())
]

big_households = married_data[
    married_data["Household Dependents"] >= 4
]

other_married_households = married_data[
    married_data["Household Dependents"] < 4
]

big_household_count = len(big_households)

big_household_rate = purchase_rate(big_households)
other_married_rate = purchase_rate(other_married_households)

big_household_estimate = estimated_extra_purchases(
    big_household_count,
    big_household_rate,
    other_married_rate
)


# Lower-income customers
low_income_data = df[df["Income Band"] == "<30k"]
low_income_rate = purchase_rate(low_income_data)


# ============================================================
# 3. DESIGN SETTINGS
# ============================================================

# Colours: each colour keeps one meaning on every chart.
NAVY = "#172B4D"     # titles and main text
BLUE = "#3975B9"     # highest rate / selected region
GOLD = "#D6A743"     # lowest rate = biggest opportunity
GREY = "#8D99A8"     # all other bars
WHITE = "#FFFFFF"
BACKGROUND = "#F3F5F8"
TEXT = "#526174"
FONT = "Arial, sans-serif"

# Font sizes (px). One scale for the whole page, nothing below 13.
SIZE_TITLE = 30       # dashboard title
SIZE_MESSAGE = 17     # message under the title
SIZE_SECTION = 22     # section headings
SIZE_CARD_TITLE = 17  # card headings and chart titles
SIZE_BODY = 15        # card text and section subtitles
SIZE_SMALL = 13       # notes, chart subtitles, key and footer

# Spacing: the same gaps everywhere
GAP = "16px"           # space between cards
SECTION_GAP = "32px"   # space between sections
CARD_PADDING = "22px"

CARD_STYLE = {
    "backgroundColor": WHITE,
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.08)",
}


# Category order
region_order = [
    "Pacific",
    "Europe",
    "North America"
]

age_order = [
    "Under 30",
    "30-39",
    "40-49",
    "50-59",
    "60+"
]

distance_order = [
    "0-1 km",
    "1-2 km",
    "2-5 km",
    "5-10 km",
    "10+ km"
]

income_order = [
    "<30k",
    "30k-49k",
    "50k-69k",
    "70k-99k",
    "100k+"
]

marital_order = [
    "Single",
    "Married"
]

dependents_order = [0, 1, 2, 3, 4, 5]


# Add categories from the data that are not already listed.
def full_category_order(data, column, preferred_order):

    actual_values = data[column].dropna().unique().tolist()

    ordered_values = [
        value for value in preferred_order
        if value in actual_values
    ]

    extra_values = [
        value for value in actual_values
        if value not in ordered_values
    ]

    return ordered_values + sorted(extra_values, key=str)


region_order = full_category_order(
    df, "Region", region_order
)

age_order = full_category_order(
    df, "Age Group", age_order
)

distance_order = full_category_order(
    df, "Distance from Nearest Installer", distance_order
)

income_order = full_category_order(
    df, "Income Band", income_order
)

marital_order = full_category_order(
    df, "Marital Status", marital_order
)

dependents_order = full_category_order(
    df, "Household Dependents", dependents_order
)


# ============================================================
# 4. REUSABLE DASHBOARD COMPONENTS
# ============================================================

def section_heading(title, subtitle):

    return html.Div(
        [
            html.H3(
                title,
                style={
                    "color": NAVY,
                    "fontSize": f"{SIZE_SECTION}px",
                    "margin": "0 0 4px 0"
                }
            ),

            html.P(
                subtitle,
                style={
                    "color": TEXT,
                    "fontSize": f"{SIZE_BODY}px",
                    "margin": "0 0 14px 0"
                }
            ),
        ]
    )


def make_card(label, value, note, highlight=False):

    style = dict(CARD_STYLE)
    style["padding"] = "20px " + CARD_PADDING

    style["backgroundColor"] = GOLD if highlight else WHITE

    text_colour = NAVY if highlight else TEXT

    return html.Div(
        [
            html.P(
                label,
                style={
                    "color": text_colour,
                    "fontSize": f"{SIZE_BODY}px",
                    "fontWeight": "bold",
                    "margin": "0 0 6px 0"
                }
            ),

            html.H2(
                value,
                style={
                    "color": NAVY,
                    "fontSize": "36px",
                    "margin": "0 0 4px 0"
                }
            ),

            html.P(
                note,
                style={
                    "color": text_colour,
                    "fontSize": f"{SIZE_SMALL}px",
                    "margin": "0"
                }
            ),
        ],
        style=style,
    )


def key_item(label, colour, dashed=False):
    """One legend item: a small swatch and a label."""

    if dashed:
        swatch = {
            "width": "26px",
            "height": "0px",
            "borderTop": "2px dashed " + colour
        }
    else:
        swatch = {
            "width": "14px",
            "height": "14px",
            "backgroundColor": colour,
            "borderRadius": "3px"
        }

    return html.Div(
        [
            html.Div(style=swatch),

            html.Span(
                label,
                style={
                    "color": TEXT,
                    "fontSize": f"{SIZE_SMALL}px"
                }
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px"
        },
    )


def chart_box(chart_id):

    style = dict(CARD_STYLE)

    style.update({
        "padding": "12px",
        "minWidth": "0"
    })

    return html.Div(
        dcc.Graph(
            id=chart_id,
            config={
                "displayModeBar": False,
                "responsive": True
            },
        ),
        style=style,
    )


def action_card(
    number,
    heading,
    main_value,
    value_caption,
    action,
    estimate
):
    """Every card has the same five parts in the same order,
    so the numbers and estimate lines line up across the row."""

    style = dict(CARD_STYLE)
    style["padding"] = CARD_PADDING

    return html.Div(
        [
            html.Div(
                number,
                style={
                    "backgroundColor": GOLD,
                    "color": NAVY,
                    "fontWeight": "bold",
                    "width": "32px",
                    "height": "32px",
                    "borderRadius": "50%",
                    "textAlign": "center",
                    "lineHeight": "32px",
                    "marginBottom": "12px"
                }
            ),

            html.H4(
                heading,
                style={
                    "color": NAVY,
                    "fontSize": f"{SIZE_CARD_TITLE}px",
                    "margin": "0 0 10px 0",
                    "lineHeight": "1.3",
                    "minHeight": "46px"
                }
            ),

            html.H2(
                main_value,
                style={
                    "color": NAVY,
                    "fontSize": "34px",
                    "margin": "0"
                }
            ),

            html.P(
                value_caption,
                style={
                    "color": TEXT,
                    "fontSize": f"{SIZE_SMALL}px",
                    "fontWeight": "bold",
                    "margin": "3px 0 12px 0",
                    "lineHeight": "1.4",
                    "minHeight": "38px"
                }
            ),

            html.P(
                action,
                style={
                    "color": NAVY,
                    "fontSize": f"{SIZE_BODY}px",
                    "lineHeight": "1.5",
                    "margin": "0 0 12px 0",
                    "minHeight": "70px"
                }
            ),

            html.P(
                estimate,
                style={
                    "color": TEXT,
                    "fontSize": f"{SIZE_SMALL}px",
                    "fontStyle": "italic",
                    "lineHeight": "1.4",
                    "margin": "0",
                    "borderTop": "1px solid #E3E7ED",
                    "paddingTop": "10px"
                }
            ),
        ],
        style=style,
    )


# ============================================================
# 5. CHART FUNCTION
# ============================================================

def make_chart(data, column, title, order, selected=None):

    grouped = data.groupby(
        column, dropna=True
    )["Purchase Flag"].agg(
        ["mean", "count", "sum"]
    )

    chart_order = [
        value for value in order
        if value in grouped.index
    ]

    chart_order += [
        value for value in grouped.index
        if value not in chart_order
    ]

    grouped = grouped.reindex(chart_order).dropna()

    if grouped.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="No customer records for this selection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=SIZE_BODY, color=TEXT)
        )

        fig.update_layout(
            height=330,
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE
        )

        return fig

    rates = grouped["mean"] * 100

    best_group = rates.idxmax()
    lowest_group = rates.idxmin()

    # Set chart colours.
    colours = []

    for group in rates.index:

        if selected is not None and group == selected:
            colours.append(BLUE)

        elif group == lowest_group:
            colours.append(GOLD)

        elif selected is None and group == best_group:
            colours.append(BLUE)

        else:
            colours.append(GREY)

    # Add the number of customers to each category label.
    # * = fewer than 30 customers.
    labels = []

    for group in rates.index:

        count = int(grouped.loc[group, "count"])

        label = f"{group} (n={count})"

        if count < 30:
            label += " *"

        labels.append(label)

    # Short title plus a one-line subtitle.
    if selected is not None and selected in rates.index:

        subtitle = (
            f"{selected} selected: {rates[selected]:.1f}%. "
            "Other regions shown for comparison"
        )

    else:

        subtitle = (
            f"Highest {best_group} ({rates[best_group]:.1f}%), "
            f"lowest {lowest_group} ({rates[lowest_group]:.1f}%)"
        )

    heading = (
        f"<b>{title}</b><br>"
        f"<span style='font-size:{SIZE_SMALL}px;color:{TEXT}'>{subtitle}</span>"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=rates.values,
            orientation="h",
            marker_color=colours,
            text=[
                f"{value:.1f}%"
                for value in rates.values
            ],
            textposition="outside",
            textfont=dict(size=14, color=NAVY),
            customdata=grouped[["count", "sum"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Purchase rate: %{x:.1f}%<br>"
                "Customers: %{customdata[0]}<br>"
                "Purchases: %{customdata[1]}"
                "<extra></extra>"
            ),
            cliponaxis=False,
        )
    )

    # Show the full company's purchase rate as a dashed line.
    fig.add_vline(
        x=company_rate,
        line_dash="dash",
        line_color=NAVY,
        line_width=1.5
    )

    fig.update_layout(
        title=dict(
            text=heading,
            font=dict(
                size=SIZE_CARD_TITLE,
                color=NAVY
            ),
            x=0.02,
            xanchor="left"
        ),

        # The percentage scale stays visible but quiet:
        # no gridlines and no axis title.
        xaxis=dict(
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            ticksuffix="%",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=SIZE_SMALL, color=TEXT),
        ),

        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=labels,
            autorange="reversed",
            showgrid=False,
            automargin=True,
            tickfont=dict(size=14, color=NAVY),
        ),

        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        showlegend=False,
        height=330,
        margin=dict(l=20, r=50, t=80, b=40),
        bargap=0.4,
        font=dict(family=FONT, color=NAVY),
    )

    return fig


# ============================================================
# 6. DASHBOARD LAYOUT
# ============================================================

app = Dash(__name__)

app.title = "SunVolt Energy Dashboard"

# Responsive grids: 4 columns on large screens, 2 on medium, 1 on small.
app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>SunVolt Energy Dashboard</title>
{%favicon%}
{%css%}
<style>
body {
    margin: 0;
    background: #F3F5F8;
}

.four-col {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
}

.two-col {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}

@media (max-width: 1100px) {
    .four-col {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 800px) {
    .four-col, .two-col {
        grid-template-columns: 1fr;
    }
}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""


app.layout = html.Div(
    [

        # Header
        html.Div(
            [
                html.H1(
                    "SunVolt Energy",
                    style={
                        "color": WHITE,
                        "fontSize": f"{SIZE_TITLE}px",
                        "margin": "0"
                    }
                ),

                html.P(
                    "North America and customers 5 km or more from an "
                    "installer are our biggest opportunities",
                    style={
                        "color": "#DCE4EF",
                        "fontSize": f"{SIZE_MESSAGE}px",
                        "margin": "8px 0 0 0",
                        "lineHeight": "1.5"
                    },
                ),
            ],
            style={
                "backgroundColor": NAVY,
                "padding": "26px 28px",
                "borderRadius": "12px",
                "marginBottom": "20px"
            },
        ),


        # Filter and chart key on one row
        html.Div(
            [

                html.Div(
                    [
                        html.Label(
                            "Filter by region",
                            style={
                                "color": NAVY,
                                "fontWeight": "bold",
                                "fontSize": f"{SIZE_BODY}px"
                            }
                        ),

                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {
                                    "label": "All Regions",
                                    "value": "All Regions"
                                }
                            ] + [
                                {
                                    "label": region,
                                    "value": region
                                }
                                for region in region_order
                            ],
                            value="All Regions",
                            clearable=False,
                            style={
                                "width": "260px",
                                "marginTop": "6px"
                            },
                        ),
                    ]
                ),

                html.Div(
                    [
                        key_item("Highest / selected", BLUE),
                        key_item("Lowest", GOLD),
                        key_item(
                            f"Company average ({company_rate:.1f}%)",
                            NAVY,
                            dashed=True
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "24px",
                        "flexWrap": "wrap",
                        "paddingBottom": "8px"
                    },
                ),
            ],

            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-end",
                "flexWrap": "wrap",
                "gap": GAP,
                "marginBottom": "20px"
            },
        ),


        # KPI cards
        html.Div(
            id="kpi-cards",
            className="four-col",
            style={"marginBottom": SECTION_GAP}
        ),


        # Customer performance charts
        section_heading(
            "Where and who buys",
            "Purchase rate by region, age, installer distance and income"
        ),

        html.Div(
            [
                chart_box("region-chart"),
                chart_box("age-chart"),
                chart_box("distance-chart"),
                chart_box("income-chart")
            ],
            className="two-col",
            style={"marginBottom": SECTION_GAP},
        ),


        # Household charts
        section_heading(
            "Household segments",
            "Purchase rate by marital status and number of dependents"
        ),

        html.Div(
            [
                chart_box("marital-chart"),
                chart_box("dependents-chart")
            ],
            className="two-col",
            style={"marginBottom": SECTION_GAP},
        ),


        # Business recommendations
        section_heading(
            "Recommended actions",
            "Based on all customers, so these do not change with the region filter"
        ),

        html.Div(
            [

                # Recommendation 1
                action_card(
                    "1",
                    "Prioritise North America",
                    f"{na_non_purchasers:,}",
                    f"potential customers (purchase rate {na_rate:.1f}%)",
                    f"Europe buys at {europe_rate:.1f}%. Test local "
                    "campaigns and review installer coverage.",
                    f"Estimate: matching Europe adds about "
                    f"{na_estimate:.0f} purchases",
                ),


                # Recommendation 2
                action_card(
                    "2",
                    "Target the best buyers",
                    f"{best_age_rate:.1f}%",
                    f"purchase rate at ages {best_age}",
                    f"Build campaigns around this group. Ages 60+ buy at "
                    f"{over60_rate:.1f}%, so test simpler offers.",
                    f"Based on all {len(df):,} customers",
                ),


                # Recommendation 3
                action_card(
                    "3",
                    "Close the distance gap",
                    f"{far_rate:.1f}%",
                    "purchase rate 5 km or more from an installer",
                    f"Customers under 5 km buy at {near_rate:.1f}%. Test "
                    "mobile installation or partner installers.",
                    f"Estimate: matching them adds about "
                    f"{far_estimate:.0f} purchases",
                ),


                # Recommendation 4
                action_card(
                    "4",
                    "Offer affordable family options",
                    f"{big_household_rate:.1f}%",
                    f"married households with 4+ dependents "
                    f"(n={big_household_count})",
                    f"Other married households buy at "
                    f"{other_married_rate:.1f}%, under-30k incomes at "
                    f"{low_income_rate:.1f}%. Test payment plans and "
                    "family bundles.",
                    f"Estimate: matching other married households adds "
                    f"about {big_household_estimate:.0f} purchases",
                ),

            ],

            className="four-col",
            style={"marginBottom": SECTION_GAP},
        ),


        # Footer and interpretation note
        html.Div(
            [

                html.P(
                    f"Source: SunVolt Energy customer dataset, "
                    f"{len(df):,} unique customers",
                    style={
                        "fontSize": f"{SIZE_SMALL}px",
                        "color": TEXT,
                        "margin": "0 0 4px 0"
                    },
                ),

                html.P(
                    "n = customers in a group. * = fewer than 30 customers, "
                    "read with caution. Estimates are illustrative, not "
                    "forecasts, and groups overlap, so do not add them "
                    "together. Rates show association, not cause.",
                    style={
                        "fontSize": f"{SIZE_SMALL}px",
                        "color": TEXT,
                        "margin": "0",
                        "lineHeight": "1.6"
                    },
                ),

            ],

            style={
                "borderTop": "1px solid #DCE2E9",
                "paddingTop": "14px"
            },
        ),

    ],

    style={
        "fontFamily": FONT,
        "backgroundColor": BACKGROUND,
        "padding": "25px",
        "maxWidth": "1400px",
        "margin": "auto",
        "minHeight": "100vh",
    },
)


# ============================================================
# 7. UPDATE DASHBOARD WHEN THE REGION FILTER CHANGES
# ============================================================

@app.callback(
    Output("kpi-cards", "children"),
    Output("region-chart", "figure"),
    Output("age-chart", "figure"),
    Output("distance-chart", "figure"),
    Output("income-chart", "figure"),
    Output("marital-chart", "figure"),
    Output("dependents-chart", "figure"),

    Input("region-filter", "value"),
)

def update_dashboard(selected_region):

    # Filter the records used by the KPIs and customer charts.
    if selected_region == "All Regions":

        selected_data = df.copy()

    else:

        selected_data = df[
            df["Region"] == selected_region
        ].copy()


    # Calculate the KPI values.
    total_customers = len(selected_data)

    total_purchases = int(
        selected_data["Purchase Flag"].sum()
    )

    potential_customers = (
        total_customers - total_purchases
    )

    selected_rate = purchase_rate(selected_data)


    # Create the KPI cards.
    cards = [

        make_card(
            "Total Customers",
            f"{total_customers:,}",
            "In this selection"
        ),

        make_card(
            "Solar Purchases",
            f"{total_purchases:,}",
            "Customers who purchased"
        ),

        make_card(
            "Purchase Rate",
            f"{selected_rate:.1f}%",
            f"Company average {company_rate:.1f}%"
        ),

        make_card(
            "Potential Customers",
            f"{potential_customers:,}",
            "Have not yet purchased",
            highlight=True
        ),

    ]


    # Keep all regions visible for comparison.
    region_figure = make_chart(
        df,
        "Region",
        "Purchase rate by region",
        region_order,
        selected=None if selected_region == "All Regions"
        else selected_region,
    )


    # Update the other charts using the selected region.
    age_figure = make_chart(
        selected_data,
        "Age Group",
        "Purchase rate by age",
        age_order
    )

    distance_figure = make_chart(
        selected_data,
        "Distance from Nearest Installer",
        "Purchase rate by installer distance",
        distance_order
    )

    income_figure = make_chart(
        selected_data,
        "Income Band",
        "Purchase rate by income",
        income_order
    )

    marital_figure = make_chart(
        selected_data,
        "Marital Status",
        "Purchase rate by marital status",
        marital_order
    )

    dependents_figure = make_chart(
        selected_data,
        "Household Dependents",
        "Purchase rate by dependents",
        dependents_order
    )


    return (
        cards,
        region_figure,
        age_figure,
        distance_figure,
        income_figure,
        marital_figure,
        dependents_figure,
    )


# ============================================================
# 8. RUN THE DASHBOARD
# ============================================================

server = app.server

if __name__ == "__main__":

    print("SunVolt dashboard is starting...")

    print(f"Customers analysed: {len(df):,}")

    print(
        f"Company purchase rate: {company_rate:.1f}%"
    )

    app.run(debug=False, host="0.0.0.0", port=8050)