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

        output = stdout.getvalue()

        if result_to_return is not None:
            if str(result_to_return).strip() not in output.strip():
                output += f"\n{result_to_return}"

        return output.strip() or "✅ Code executed successfully.", fig_to_return

    except Exception as e:
        return f"❌ Error: {str(e)}", None
