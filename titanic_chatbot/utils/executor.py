import io
import contextlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import ast

def execute_code(code, df):
    stdout = io.StringIO()
    local_vars = {
        "df": df,
        "plt": plt,
        "pd": pd,
        "sns": sns,
        "px": px,
        "go": go
    }

    fig_to_return = None
    result_to_return = None
    plot_metadata = ""

    try:
        plt.clf()  # Clear any previous matplotlib plots

        # Parse the code and isolate the last expression (e.g., x)
        parsed = ast.parse(code)
        body = parsed.body
        last_expr_code = None
        is_last_expr_print = False

        if body and isinstance(body[-1], ast.Expr):
            last_expr = body[-1].value
            if isinstance(last_expr, ast.Call) and getattr(last_expr.func, 'id', '') == 'print':
                is_last_expr_print = True
            else:
                try:
                    last_expr_code = compile(ast.Expression(last_expr), filename="<ast>", mode="eval")
                except Exception:
                    last_expr_code = None

        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)
            if last_expr_code and not is_last_expr_print:
                result_to_return = eval(last_expr_code, {}, local_vars)

        # Detect matplotlib figure
        if plt.gcf().get_axes():
            fig_to_return = plt.gcf()
        elif "fig" in local_vars and isinstance(local_vars["fig"], (go.Figure, px.Figure)):
            fig_to_return = local_vars["fig"]

        # Extract plot metadata if a figure exists
        if fig_to_return is not None:
            plot_metadata = summarize_plot(fig_to_return)

        output = stdout.getvalue()

        if result_to_return is not None:
            if str(result_to_return).strip() not in output.strip():
                output += f"\n{result_to_return}"

        return output.strip() or "✅ Code executed successfully.", fig_to_return, plot_metadata

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def summarize_plot(fig):
    try:
        if isinstance(fig, plt.Figure):
            ax = fig.axes[0]
            plot_data = []

            # Handle common matplotlib plot types
            for container in ax.containers:  # for bar charts
                labels = [tick.get_text() for tick in ax.get_xticklabels()]
                values = [patch.get_height() for patch in container]
                plot_data.append({
                    "labels": labels,
                    "values": values
                })

            # For histograms, line plots, etc.
            if not plot_data and ax.lines:
                line = ax.lines[0]
                plot_data.append({
                    "x": line.get_xdata().tolist(),
                    "y": line.get_ydata().tolist()
                })

            return {
                "type": "matplotlib",
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "data": plot_data
            }

        elif isinstance(fig, (go.Figure, px.Figure)):
            plot_data = []
            for trace in fig.data:
                trace_data = {}
                if hasattr(trace, "x") and hasattr(trace, "y"):
                    trace_data["x"] = list(trace.x) if trace.x is not None else []
                    trace_data["y"] = list(trace.y) if trace.y is not None else []
                if hasattr(trace, "labels") and hasattr(trace, "values"):
                    trace_data["labels"] = list(trace.labels)
                    trace_data["values"] = list(trace.values)
                plot_data.append(trace_data)

            layout = fig.layout
            return {
                "type": "plotly",
                "title": layout.title.text if layout.title else "",
                "xlabel": layout.xaxis.title.text if layout.xaxis and layout.xaxis.title else "",
                "ylabel": layout.yaxis.title.text if layout.yaxis and layout.yaxis.title else "",
                "data": plot_data
            }

    except Exception as e:
        return {"type": "error", "error": str(e)}
    
    return {"type": "none"}