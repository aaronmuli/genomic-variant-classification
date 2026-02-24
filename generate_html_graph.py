import plotly.graph_objects as go
import plotly.io as pio
import io
import matplotlib.pyplot as plt
from io import BytesIO


def generate_bar_graph_html(data, graph_type="auto", title=None):
    """
    Generates an embeddable Plotly horizontal bar graph.

    Parameters:
        data: 
            - List of ranked class dictionaries
              OR
            - Dictionary of feature importances
        graph_type: "class" | "features" | "auto"
        title: Optional custom title

    Returns:
        HTML div string (for Flask template)
    """

    fig = go.Figure()

    # -------------------------------
    # Case 1: Class Probability Graph
    # -------------------------------
    if isinstance(data, list) and (graph_type == "class" or graph_type == "auto"):
        data_sorted = sorted(data, key=lambda x: x['rank'])

        labels = [f"Class {r['class']}" for r in data_sorted]
        values = [r['confidence'] for r in data_sorted]

        hover_text = [
            f"Probability: {r['probability']:.4f}<br>Confidence: {r['confidence']:.2f}%"
            for r in data_sorted
        ]

        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            text=[f"{v:.2f}%" for v in values],
            textposition='outside',
            hovertext=hover_text,
            hoverinfo='text'
        ))

        fig.update_layout(
            title=title or "Cancer Variant Classification — Class Probability Ranking",
            xaxis_title="Confidence (%)",
            yaxis_title="Predicted Class",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=40)
        )

    # -----------------------------------
    # Case 2: Medical Feature Importance
    # -----------------------------------
    elif isinstance(data, dict) and (graph_type == "features" or graph_type == "auto"):
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)

        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            text=[f"{v:.2f}" for v in values],
            textposition='outside'
        ))

        fig.update_layout(
            title=title or "Top Medical Keywords Driving Classification",
            xaxis_title="Gain Score",
            yaxis_title="Medical Term",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=40)
        )

    else:
        raise ValueError("Unsupported data format.")

    return pio.to_html(
        fig, 
        full_html=False,
        include_plotlyjs='cdn',
        config={"responsive": True}
    )



def generate_bar_graph_image(data, graph_type="class"):
    labels = []
    values = []

    if graph_type == "class":
        for item in data:
            labels.append(f"Class {item['class']}")
            values.append(item['confidence'])
    else:
        for k, v in data.items():
            labels.append(k)
            values.append(v)

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return buffer
