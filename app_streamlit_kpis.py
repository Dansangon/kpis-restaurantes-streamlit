import pandas as pd
import streamlit as st
from io import BytesIO

APP_TITLE = "KPIs por Restaurante (Streamlit) - estilo azul + ranking + comparativa"

# ---------- Login (usuario + contraseña) ----------
def check_login():
    def _verify():
        u_ok = st.session_state.get("user", "") == st.secrets["auth"]["username"]
        p_ok = st.session_state.get("pass", "") == st.secrets["auth"]["password"]
        st.session_state["auth_ok"] = bool(u_ok and p_ok)
        if st.session_state["auth_ok"]:
            # borra password del estado para no mantenerlo en memoria
            st.session_state.pop("pass", None)

    if st.session_state.get("auth_ok", False):
        return True

    st.markdown("## Acceso")
    st.text_input("Usuario", key="user")
    st.text_input("Contraseña", type="password", key="pass", on_change=_verify)
    if "auth_ok" in st.session_state and not st.session_state["auth_ok"]:
        st.error("Usuario o contraseña incorrectos")
    return False


# ---------------- Utilidades ----------------
def norm(s):
    return str(s).strip().lower().replace("\n", " ").replace("  ", " ")

def choose_total_column(columns):
    for c in columns:
        cnorm = norm(c)
        if "total" in cnorm and "general" in cnorm:
            return c
    return None

def weekly_columns(df, total_col):
    base = ["RMO_restaurant_id", "Unnamed: 1"]
    w = [c for c in df.columns if c not in base]
    if total_col and total_col in w:
        w.remove(total_col)
    return w

def find_metric_row(block, metric_aliases):
    labels = block["Unnamed: 1"].map(norm)
    want = [norm(a) for a in (metric_aliases if isinstance(metric_aliases, (list, tuple)) else [metric_aliases])]
    idx = labels[labels.isin(want)].index
    if len(idx) == 0:
        return None
    return block.loc[idx[0]]

def fmt_value(concepto, val):
    if pd.isna(val):
        return ""
    if concepto in ("Bad Order rate", "New customers (%)", "Overall Rating Good"):
        return f"{float(val) * 100:,.2f}%"
    else:
        return f"{float(val):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

BLUE = "#B7D5F0"
ALT = "#F4F8FC"
BORDER = "#7DA7D9"
TEXT_DARK = "#0A2745"

def style_blue_table(df: pd.DataFrame):
    # df con index=Concepto y columna(s) de valores
    styler = df.style

    styler = styler.set_table_styles([
        {"selector": "th", "props": f"background-color: {BLUE}; color: {TEXT_DARK}; border: 1px solid {BORDER};"},
        {"selector": "td", "props": f"border: 1px solid {BORDER};"},
        {"selector": "table", "props": f"border-collapse: collapse; width: 100%;"},
    ])

    # Alternar filas
    def _alt_rows(s):
        return [
            f"background-color: {'white' if i % 2 == 0 else ALT};"
            for i in range(len(s))
        ]

    styler = styler.apply(_alt_rows, axis=0, subset=pd.IndexSlice[:, :])
    return styler

# ---------------- Cálculos KPI ----------------
def compute_kpis_for_restaurant(mdf_ffill: pd.DataFrame, restaurant_id: str):
    block = mdf_ffill.loc[mdf_ffill["RMO_restaurant_id"].astype(str) == str(restaurant_id)].copy()
    idx = [
        "Good Orders", "Bad Orders", "Bad Order rate", "Active customers", "New customers (%)",
        "Overall Rating Good", "AOV good", "billing good", "Orders with voucher"
    ]
    if block.empty:
        out = pd.DataFrame({"Grand Total": [pd.NA] * len(idx)}, index=idx)
        raw = {}
        return out, raw

    total_col = choose_total_column(block.columns)
    if total_col is None:
        candidates = [c for c in block.columns if c not in ["RMO_restaurant_id", "Unnamed: 1"]]
        total_col = candidates[0] if candidates else None

    wcols = weekly_columns(block, total_col)

    map_aliases = {
        "good_orders": ["good orders", "good order", "goodorders"],
        "bad_orders": ["bad orders", "bad order", "badorders"],
        "bor_total": ["% bor total", "bad order rate", "% bad order rate"],
        "active_customers": ["active customers", "active cosumers", "active customer"],
        "pct_new_customers": ["% new customers", "new customers (%)", "new customers %"],
        "new_customers_count": ["new customers", "new customers in tr", "new customers total"],
        "rated_orders": ["rated orders", "rated order"],
        "aov": ["aov", "aov good"],
        "gmv": ["gmv", "billing good", "billing"],
        "voucher_orders": ["voucher orders", "orders with voucher", "voucher"],
    }

    def val_total(metric_key):
        row = find_metric_row(block, map_aliases[metric_key])
        if row is None or total_col is None or total_col not in row.index:
            return float("nan")
        return pd.to_numeric(pd.Series(row[total_col]), errors="coerce").iloc[0]

    def mean_weekly(metric_key):
        row = find_metric_row(block, map_aliases[metric_key])
        if row is None or not wcols:
            return float("nan")
        return pd.to_numeric(row[wcols], errors="coerce").mean(skipna=True)

    good_orders_total = val_total("good_orders")
    bad_orders_total = val_total("bad_orders")
    gmv_total = val_total("gmv")
    voucher_total = val_total("voucher_orders")

    row_bor = find_metric_row(block, map_aliases["bor_total"])
    if row_bor is not None and wcols:
        bad_order_rate_avg = pd.to_numeric(row_bor[wcols], errors="coerce").mean(skipna=True)
    else:
        row_bad = find_metric_row(block, map_aliases["bad_orders"])
        row_good = find_metric_row(block, map_aliases["good_orders"])
        if row_bad is not None and row_good is not None and wcols:
            bad = pd.to_numeric(row_bad[wcols], errors="coerce")
            good = pd.to_numeric(row_good[wcols], errors="coerce")
            denom = good.add(bad, fill_value=0)
            rate = (bad / denom)
            bad_order_rate_avg = rate.mean(skipna=True)
        else:
            bad_order_rate_avg = float("nan")

    active_customers_total = val_total("active_customers")
    new_customers_pct_total = val_total("pct_new_customers")
    new_customers_count_total = val_total("new_customers_count")

    row_rated = find_metric_row(block, map_aliases["rated_orders"])
    row_good = find_metric_row(block, map_aliases["good_orders"])
    if row_rated is not None and row_good is not None and wcols:
        rated = pd.to_numeric(row_rated[wcols], errors="coerce")
        good = pd.to_numeric(row_good[wcols], errors="coerce")
        ogr_series = (rated / good)
        overall_rating_good_avg = ogr_series.mean(skipna=True)
    else:
        overall_rating_good_avg = float("nan")

    aov_good_avg = mean_weekly("aov")

    data = {
        "Good Orders": good_orders_total,
        "Bad Orders": bad_orders_total,
        "Bad Order rate": bad_order_rate_avg,
        "Active customers": active_customers_total,
        "New customers (%)": new_customers_pct_total,
        "Overall Rating Good": overall_rating_good_avg,
        "AOV good": aov_good_avg,
        "billing good": gmv_total,
        "Orders with voucher": voucher_total,
    }
    out = pd.DataFrame.from_dict(data, orient="index", columns=["Grand Total"])

    raw = {
        "good_orders_total": good_orders_total,
        "gmv_total": gmv_total,
        "new_customers_count_total": new_customers_count_total,
    }
    return out, raw

@st.cache_data(show_spinner=False)
def compute_all(master_bytes: bytes, dropdown_bytes: bytes, city_filter: str):
    mdf = pd.read_excel(BytesIO(master_bytes), sheet_name=0)
    ddf = pd.read_excel(BytesIO(dropdown_bytes), sheet_name=0)

    # Mapear columnas del dropdown
    cols_norm = {str(c).strip().lower(): c for c in ddf.columns}
    def pick(keys):
        for k in keys:
            if k in cols_norm:
                return cols_norm[k]
        return None

    id_col = pick(["restaurant id", "id restaurante", "id"])
    city_col = pick(["city", "ciudad"])
    name_col = pick(["restaurant name", "nombre restaurante", "name"])

    if id_col is None or city_col is None:
        raise ValueError("El dropdown debe tener columnas equivalentes a 'Restaurant ID' y 'City/Ciudad'.")

    # Normalizar maestra: RMO_restaurant_id y columna de conceptos
    if "RMO_restaurant_id" not in mdf.columns:
        raise ValueError("Falta la columna 'RMO_restaurant_id' en la Tabla Maestra.")

    label_col = "Unnamed: 1"
    if label_col not in mdf.columns:
        label_col = mdf.columns[1]  # fallback: segunda columna
        mdf = mdf.rename(columns={label_col: "Unnamed: 1"})

    mdf["RMO_restaurant_id"] = mdf["RMO_restaurant_id"].ffill()

    # filtro flexible ciudad
    ddf_tmp = ddf.copy()
    ddf_tmp[id_col] = ddf_tmp[id_col].astype(str)

    cf = (city_filter or "").strip().lower()
    if cf:
        ddf_city = ddf_tmp[ddf_tmp[city_col].astype(str).str.lower() == cf]
        if ddf_city.empty:
            ddf_city = ddf_tmp[ddf_tmp[city_col].astype(str).str.lower().str.contains(cf, na=False)]
        if ddf_city.empty:
            # fallback: todo
            ddf_city = ddf_tmp
            warn = f"No hay coincidencias para ciudad '{city_filter}'. Se usarán TODOS los restaurantes."
        else:
            warn = None
    else:
        ddf_city = ddf_tmp
        warn = None

    if name_col:
        ddf_city["_display_name"] = ddf_city[id_col].astype(str) + " - " + ddf_city[name_col].astype(str)
    else:
        ddf_city["_display_name"] = ddf_city[id_col].astype(str)

    results = {}
    raw_stats = {}
    for _, row in ddf_city.iterrows():
        rid = str(row[id_col])
        df, raw = compute_kpis_for_restaurant(mdf, rid)
        results[rid] = df
        raw_stats[rid] = raw

    return results, ddf_city, raw_stats, warn

# ---------------- Exportación Excel ----------------
def to_excel_all(per_restaurant, dropdown_df):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
        wb = xw.book
        header_fmt = wb.add_format({"bold": True, "bg_color": BLUE, "border": 1, "font_color": TEXT_DARK, "align": "center"})
        left_fmt = wb.add_format({"border": 1})
        num_fmt = wb.add_format({"border": 1, "num_format": "#,##0.00"})
        pct_fmt = wb.add_format({"border": 1, "num_format": "0.00%"})

        for rid, df in per_restaurant.items():
            display = rid
            row = dropdown_df.loc[dropdown_df["_display_name"].str.startswith(str(rid))]
            if not row.empty:
                display = str(row["_display_name"].iloc[0])

            sheet = str(rid)[:31]
            df_reset = df.reset_index().rename(columns={"index": "Concepto"})
            df_reset.to_excel(xw, sheet_name=sheet, index=False, startrow=1)
            ws = xw.sheets[sheet]
            ws.merge_range(0, 0, 0, 1, display, header_fmt)
            ws.write(1, 0, "Concepto", header_fmt)
            ws.write(1, 1, "Grand Total", header_fmt)

            for r in range(2, 2 + len(df_reset)):
                ws.set_row(r, 18)
                concepto = str(df_reset.iloc[r - 2, 0])
                val = df_reset.iloc[r - 2, 1]
                ws.write(r, 0, concepto, left_fmt)
                if concepto in ("Bad Order rate", "New customers (%)", "Overall Rating Good"):
                    ws.write_number(r, 1, float(val) if pd.notna(val) else 0.0, pct_fmt)
                else:
                    ws.write_number(r, 1, float(val) if pd.notna(val) else 0.0, num_fmt)

            ws.set_column(0, 0, 35)
            ws.set_column(1, 1, 18)

    bio.seek(0)
    return bio.getvalue()

def to_excel_compare(df_compare: pd.DataFrame):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
        wb = xw.book
        ws_name = "Comparativa"
        df_compare.to_excel(xw, sheet_name=ws_name, index=True, startrow=1)
        ws = xw.sheets[ws_name]

        header_fmt = wb.add_format({"bold": True, "bg_color": BLUE, "border": 1, "font_color": TEXT_DARK, "align": "center"})
        num_fmt = wb.add_format({"border": 1, "num_format": "#,##0.00"})
        pct_fmt = wb.add_format({"border": 1, "num_format": "0.00%"})
        left_fmt = wb.add_format({"border": 1})

        ws.write(1, 0, "Concepto", header_fmt)
        for j, col in enumerate(df_compare.columns, start=1):
            ws.write(1, j, str(col), header_fmt)

        for i, concepto in enumerate(df_compare.index, start=2):
            ws.write(i, 0, concepto, left_fmt)
            for j, col in enumerate(df_compare.columns, start=1):
                val = df_compare.loc[concepto, col]
                if concepto in ("Bad Order rate", "New customers (%)", "Overall Rating Good"):
                    ws.write_number(i, j, float(val) if pd.notna(val) else 0.0, pct_fmt)
                else:
                    ws.write_number(i, j, float(val) if pd.notna(val) else 0.0, num_fmt)

        ws.set_column(0, 0, 32)
        for j in range(1, 1 + len(df_compare.columns)):
            ws.set_column(j, j, 18)

    bio.seek(0)
    return bio.getvalue()

if not check_login():
    st.stop()


# ---------------- UI Streamlit ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Carga de archivos")
    master_file = st.file_uploader("Tabla Maestra (.xlsx)", type=["xlsx"], key="master")
    dropdown_file = st.file_uploader("Dropdown (.xlsx)", type=["xlsx"], key="dropdown")
    st.subheader("Filtros")
    city = st.text_input("Ciudad (vacío = todas)", value="Valencia")

    run = st.button("Calcular", type="primary", use_container_width=True)

if "computed" not in st.session_state:
    st.session_state.computed = False

if run:
    if not master_file or not dropdown_file:
        st.error("Selecciona ambos archivos (Tabla Maestra y Dropdown).")
    else:
        try:
            with st.spinner("Calculando KPIs..."):
                per_restaurant, dropdown_df, raw_stats, warn = compute_all(
                    master_file.getvalue(),
                    dropdown_file.getvalue(),
                    city
                )
            st.session_state.per_restaurant = per_restaurant
            st.session_state.dropdown_df = dropdown_df
            st.session_state.raw_stats = raw_stats
            st.session_state.warn = warn
            st.session_state.computed = True
            st.success("Cálculo completado.")
        except Exception as e:
            st.session_state.computed = False
            st.error(str(e))

if st.session_state.computed:
    if st.session_state.get("warn"):
        st.warning(st.session_state.warn)

    per_restaurant = st.session_state.per_restaurant
    dropdown_df = st.session_state.dropdown_df
    raw_stats = st.session_state.raw_stats

    tabs = st.tabs(["KPIs por restaurante", "Ranking", "Comparativa (2–3)", "Exportar"])

    # ---- Tab 1: KPIs ----
    with tabs[0]:
        st.subheader("KPIs")
        show_all = st.checkbox("Ver TODOS los restaurantes", value=False)

        displays = list(dropdown_df["_display_name"])
        if not displays:
            st.info("No hay restaurantes en el dropdown con el filtro actual.")
        else:
            if show_all:
                for rid, df in per_restaurant.items():
                    row = dropdown_df.loc[dropdown_df["_display_name"].str.startswith(str(rid))]
                    title = str(row["_display_name"].iloc[0]) if not row.empty else rid

                    st.markdown(f"### {title}")
                    df_show = df.copy()
                    df_show["Grand Total"] = df_show.index.map(lambda c: fmt_value(c, df_show.loc[c, "Grand Total"]))
                    df_show = df_show.rename_axis("Concepto")
                    st.dataframe(style_blue_table(df_show), use_container_width=True, height=360)
                    st.divider()
            else:
                pick = st.selectbox("Selecciona restaurante", displays)
                rid = str(pick).split(" - ")[0]
                df = per_restaurant.get(rid)

                st.markdown(f"### {pick}")
                df_show = df.copy()
                df_show["Grand Total"] = df_show.index.map(lambda c: fmt_value(c, df_show.loc[c, "Grand Total"]))
                df_show = df_show.rename_axis("Concepto")
                st.dataframe(style_blue_table(df_show), use_container_width=True, height=360)

        # ---- Tab 2: Ranking ----
    with tabs[1]:
        st.subheader("Ranking (Top 1)")

        rows = []
        for rid, raw in raw_stats.items():
            row = dropdown_df.loc[dropdown_df["_display_name"].str.startswith(str(rid))]
            display = str(row["_display_name"].iloc[0]) if not row.empty else rid
            rows.append({
                "Restaurante": display,
                "GMV (billing good)": raw.get("gmv_total", float("nan")),
                "Good Orders": raw.get("good_orders_total", float("nan")),
                "New Customers (conteo)": raw.get("new_customers_count_total", float("nan")),
            })
        df_rank = pd.DataFrame(rows)

        def top_of(col):
            s = pd.to_numeric(df_rank[col], errors="coerce")
            if s.isna().all():
                return "(sin datos)", None
            idx = s.idxmax()
            return df_rank.loc[idx, "Restaurante"], float(s.loc[idx])

        def fmt_num_es(x):
            if x is None or pd.isna(x):
                return "—"
            return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        def fmt_int_es(x):
            if x is None or pd.isna(x):
                return "—"
            return f"{int(round(x)):,}".replace(",", ".")

        c1, c2, c3 = st.columns(3)
        n1, v1 = top_of("GMV (billing good)")
        n2, v2 = top_of("Good Orders")
        n3, v3 = top_of("New Customers (conteo)")

        with c1:
            st.metric(
                label="Mayor facturación (billing good)",
                value=n1,
                delta=f"GMV: {fmt_num_es(v1)}"
            )
        with c2:
            st.metric(
                label="Más Good Orders",
                value=n2,
                delta=f"Good Orders: {fmt_int_es(v2)}"
            )
        with c3:
            st.metric(
                label="Más New Customers (conteo)",
                value=n3,
                delta=f"New Customers: {fmt_int_es(v3)}"
            )

        with st.expander("Ver tabla completa del ranking"):
            st.dataframe(df_rank.sort_values("GMV (billing good)", ascending=False), use_container_width=True)


    # ---- Tab 3: Comparativa ----
    with tabs[2]:
        st.subheader("Comparativa (2–3 restaurantes)")

        picks = st.multiselect(
            "Selecciona 2 o 3 restaurantes",
            list(dropdown_df["_display_name"]),
            max_selections=3
        )

        if len(picks) >= 2:
            conceptos = [
                "Good Orders","Bad Orders","Bad Order rate","Active customers",
                "New customers (%)","Overall Rating Good","AOV good","billing good","Orders with voucher"
            ]
            combined = pd.DataFrame(index=conceptos)
            for disp in picks:
                rid = str(disp).split(" - ")[0]
                df = per_restaurant.get(rid)
                combined[disp] = df.loc[conceptos, "Grand Total"]

            # tabla bonita (formateada como texto)
            combined_show = combined.copy()
            for concepto in combined_show.index:
                combined_show.loc[concepto] = [
                    fmt_value(concepto, combined_show.loc[concepto, col]) for col in combined_show.columns
                ]
            combined_show = combined_show.rename_axis("Concepto")
            st.dataframe(style_blue_table(combined_show), use_container_width=True, height=380)

            excel_bytes = to_excel_compare(combined)
            st.download_button(
                "Descargar comparativa (Excel)",
                data=excel_bytes,
                file_name="comparativa_2_3.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Selecciona 2 o 3 restaurantes para generar la tabla comparativa.")

    # ---- Tab 4: Exportar ----
    with tabs[3]:
        st.subheader("Exportaciones")
        excel_all = to_excel_all(per_restaurant, dropdown_df)
        st.download_button(
            "Descargar TODOS los restaurantes (Excel)",
            data=excel_all,
            file_name="kpis_por_restaurante.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("Carga los dos Excel y pulsa **Calcular** en la barra lateral.")
