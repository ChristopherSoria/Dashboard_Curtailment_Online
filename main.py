import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date
from matplotlib.patches import Patch

# ==========================================
# 1. Configuração Inicial
# ==========================================
st.set_page_config(page_title="Dashboard Curtailment", layout="wide")
st.title("Análise de Curtailment: Eólico e Solar")

# ==========================================
# 2. Carregamento
# ==========================================
@st.cache_data
def carregar_dados(fonte):
    base_path = "Base_Dados"
    arquivo = f"curtailment_horario_{fonte.lower()}.parquet"
    caminho = os.path.join(base_path, arquivo)

    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_parquet(caminho)
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # Compatibilidade com versões antigas dos Parquets.
    renomear = {}
    for razao in ["REL", "CNF", "ENE"]:
        antiga = f"restrita_{razao}"
        nova = f"geracao_restrita_{razao}"
        if nova not in df.columns and antiga in df.columns:
            renomear[antiga] = nova
    if renomear:
        df = df.rename(columns=renomear)

    return df


@st.cache_data
def carregar_balanco():
    base_path = "Base_Dados"
    caminho = os.path.join(base_path, "balanco_submercado_horario.parquet")

    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_parquet(caminho)
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # Os indicadores podem vir prontos do Parquet. Se não vierem,
    # são calculados a partir das variáveis originais.
    required = ["val_carga", "val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]
    if all(c in df.columns for c in required):
        carga = pd.to_numeric(df["val_carga"], errors="coerce")
        hidro = pd.to_numeric(df["val_gerhidraulica"], errors="coerce")
        term = pd.to_numeric(df["val_gertermica"], errors="coerce")
        eol = pd.to_numeric(df["val_gereolica"], errors="coerce")
        sol = pd.to_numeric(df["val_gersolar"], errors="coerce")
        den = carga.where(carga > 0)

        calculos = {
            "hydro_norm": 100.0 * hidro / den,
            "term_norm": 100.0 * term / den,
            "eol_norm": 100.0 * eol / den,
            "sol_norm": 100.0 * sol / den,
            "sync_norm": 100.0 * (hidro + term) / den,
            "ibr_norm": 100.0 * (eol + sol) / den,
        }
        for col, valores in calculos.items():
            if col not in df.columns:
                df[col] = valores
        if "penetracao_ibr" not in df.columns:
            df["penetracao_ibr"] = df["ibr_norm"]

    return df


@st.cache_data
def carregar_usina():
    base_path = "Base_Dados"
    caminho = os.path.join(base_path, "geracao_usina_horario.parquet")

    if not os.path.exists(caminho):
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_parquet(caminho)
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    if "din_instante" in df.columns:
        df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
    return df


df_eolico = carregar_dados("Eolico")
df_solar = carregar_dados("Solar")
df_balanco = carregar_balanco()
df_usina = carregar_usina()

# ==========================================
# 3. Barra Lateral
# ==========================================
st.sidebar.header("Configurações")

analise_escolhida = st.sidebar.selectbox(
    "Escolha a Análise:",
    [
        "0. Sumário e Guia do Dashboard",
        "1. Curtailment por Dia da Semana",
        "2. Curtailment por Hora do Dia",
        "3. Geração vs Referência (Dia da Semana)",
        "4. Geração vs Referência (Hora do Dia)",
        "5. Carga e Carga Líquida por Hora",
        "6. Carga por Dia da Semana",
        "7. Fotovoltaica vs MMGD por Hora",
        "8. Participação da Geração por Fonte",
        "9. IBR, Geração Síncrona e Médias Anuais",
        "10. Resumo Anual de Curtailment",
        "11. Comparação Anual e por Estado/Subsistema",
        "12. Solar: MMGD x Centralizada x Restrita",
        "13. Solar: Variabilidade (CV)",
        "14. Eólica: Geração x Restrita",
        "15. Eólica: Variabilidade (CV)",
    ]
)

fonte_escolhida = st.sidebar.selectbox(
    "Fonte para análises de curtailment:",
    ["Eólico", "Solar", "Eólico + Solar"]
)

# Base usada apenas para montar os filtros espaciais na lateral.
if fonte_escolhida == "Eólico":
    _df_curt_sidebar = df_eolico
elif fonte_escolhida == "Solar":
    _df_curt_sidebar = df_solar
else:
    _df_curt_sidebar = pd.concat([df_eolico, df_solar], ignore_index=True)

st.sidebar.subheader("Escopo do Curtailment")
escopos_disponiveis = ["Total"]
if "id_subsistema" in _df_curt_sidebar.columns:
    escopos_disponiveis.append("Subsistema")
if "id_estado" in _df_curt_sidebar.columns:
    escopos_disponiveis.append("Estado")

escopo_curtailment = st.sidebar.selectbox(
    "Nível espacial:",
    escopos_disponiveis
)

valor_escopo_curtailment = None
if escopo_curtailment == "Subsistema":
    opcoes = sorted(
        _df_curt_sidebar["id_subsistema"].dropna().astype(str).str.strip().unique().tolist()
    )
    if opcoes:
        idx = opcoes.index("SIN") if "SIN" in opcoes else 0
        valor_escopo_curtailment = st.sidebar.selectbox("Subsistema:", opcoes, index=idx)
elif escopo_curtailment == "Estado":
    opcoes = sorted(
        _df_curt_sidebar["id_estado"].dropna().astype(str).str.strip().unique().tolist()
    )
    if opcoes:
        valor_escopo_curtailment = st.sidebar.selectbox("Estado:", opcoes)

st.sidebar.divider()

# Balanço: evita somar SIN e subsistemas simultaneamente.
st.sidebar.subheader("Balanço do Sistema")
if not df_balanco.empty and "id_subsistema" in df_balanco.columns:
    subsistemas_balanco = sorted(
        df_balanco["id_subsistema"].dropna().astype(str).str.strip().unique().tolist()
    )
else:
    subsistemas_balanco = []

if subsistemas_balanco:
    idx_bal = subsistemas_balanco.index("SIN") if "SIN" in subsistemas_balanco else 0
    subsistema_balanco = st.sidebar.selectbox(
        "Subsistema para carga/geração:",
        subsistemas_balanco,
        index=idx_bal
    )
else:
    subsistema_balanco = None
    st.sidebar.caption("Parquet de balanço sem `id_subsistema`; será usada a base disponível.")

st.sidebar.divider()
st.sidebar.subheader("Filtro de Período")

# O período disponível é definido automaticamente pelas bases carregadas.
datas_disponiveis = []
for df_base in [df_eolico, df_solar, df_balanco, df_usina]:
    if not df_base.empty and "Data" in df_base.columns:
        datas_validas = pd.to_datetime(df_base["Data"], errors="coerce").dropna()
        if not datas_validas.empty:
            datas_disponiveis.extend([datas_validas.min(), datas_validas.max()])

if datas_disponiveis:
    min_date = pd.to_datetime(min(datas_disponiveis)).date()
    max_date = pd.to_datetime(max(datas_disponiveis)).date()
else:
    min_date = date(2024, 4, 1)
    max_date = date(2026, 4, 30)

intervalo_datas = st.sidebar.date_input(
    "Selecione as datas:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ==========================================
# 4. Funções Auxiliares
# ==========================================
def escolher_base_curtailment(df_eolico, df_solar, fonte):
    if fonte == "Eólico":
        return df_eolico.copy(), "#6c757d", "Eólico"

    elif fonte == "Solar":
        return df_solar.copy(), "#d62828", "Solar"

    else:
        df_total = pd.concat([df_eolico, df_solar], ignore_index=True)
        return df_total, "#1f77b4", "Eólico + Solar"


def filtrar_escopo_curtailment(df, escopo, valor):
    if df.empty or escopo == "Total" or valor is None:
        return df.copy()

    coluna = "id_subsistema" if escopo == "Subsistema" else "id_estado"
    if coluna not in df.columns:
        return df.iloc[0:0].copy()

    valores = df[coluna].astype(str).str.strip().str.upper()
    return df.loc[valores == str(valor).strip().upper()].copy()


def descricao_escopo(escopo, valor):
    if escopo == "Total" or valor is None:
        return "Total"
    return f"{escopo}: {valor}"


def filtrar_balanco_subsistema(df, subsistema):
    if df.empty or subsistema is None or "id_subsistema" not in df.columns:
        return df.copy()
    valores = df["id_subsistema"].astype(str).str.strip().str.upper()
    return df.loc[valores == str(subsistema).strip().upper()].copy()


# ==========================================
# 5. Funções dos Gráficos
# ==========================================
def gerar_grafico_curtailment_mensal_percentual(df_filtrado, cor_hex, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    if df_filtrado.empty:
        return None

    df = df_filtrado.copy()
    df["AnoMes"] = df["Data"].dt.to_period("M").astype(str)

    df_grouped = df.groupby("AnoMes")[
        ["geracao_restrita", "val_geracaoreferencia"]
    ].sum().reset_index()

    df_grouped["Curtailment_%"] = np.where(
        df_grouped["val_geracaoreferencia"] > 0,
        100 * df_grouped["geracao_restrita"] / df_grouped["val_geracaoreferencia"],
        0
    )

    df_grouped["TWh_restrita"] = df_grouped["geracao_restrita"] / 1e6

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df_grouped))

    bars = ax.bar(
        x,
        df_grouped["Curtailment_%"],
        color=cor_hex,
        width=0.7
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped["AnoMes"], rotation=90)
    ax.set_ylabel("Curtailment (%)")
    ax.set_xlabel("Ano-Mês")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    offset = (
        df_grouped["Curtailment_%"].max() * 0.02
        if df_grouped["Curtailment_%"].max() > 0
        else 0.2
    )

    for bar, twh in zip(bars, df_grouped["TWh_restrita"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{twh:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
        )

    plt.tight_layout()
    return fig


def gerar_grafico_curtailment_mensal_por_tipo(df_filtrado, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    if df_filtrado.empty:
        return None

    colunas = [
        "geracao_restrita_REL",
        "geracao_restrita_CNF",
        "geracao_restrita_ENE"
    ]

    faltando = [c for c in colunas if c not in df_filtrado.columns]

    if faltando:
        st.error(f"Faltam colunas no parquet: {faltando}")
        return None

    df = df_filtrado.copy()
    df["AnoMes"] = df["Data"].dt.to_period("M").astype(str)

    df_grouped = df.groupby("AnoMes")[colunas].sum().reset_index()

    df_grouped["REL_GWh"] = df_grouped["geracao_restrita_REL"] / 1e3
    df_grouped["CNF_GWh"] = df_grouped["geracao_restrita_CNF"] / 1e3
    df_grouped["ENE_GWh"] = df_grouped["geracao_restrita_ENE"] / 1e3

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(df_grouped))
    width = 0.25

    ax.bar(x - width, df_grouped["REL_GWh"], width=width, label="REL")
    ax.bar(x, df_grouped["CNF_GWh"], width=width, label="CNF")
    ax.bar(x + width, df_grouped["ENE_GWh"], width=width, label="ENE")

    ax.set_xticks(x)
    ax.set_xticklabels(df_grouped["AnoMes"], rotation=90)
    ax.set_ylabel("Curtailment (GWh)")
    ax.set_xlabel("Ano-Mês")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Tipo de restrição")
    ax.set_title(titulo)

    plt.tight_layout()
    return fig


def gerar_grafico_dia_semana(df_filtrado, cor_hex, titulo):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14})

    if df_filtrado.empty:
        return None

    df_filtrado = df_filtrado.copy()
    df_filtrado["Dow"] = df_filtrado["Data"].dt.dayofweek
    dow_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    df_grouped = df_filtrado.groupby("Dow")[
        ["geracao_restrita", "val_geracaoreferencia", "geracao_restrita_ENE"]
    ].sum().reindex(range(7), fill_value=0).reset_index()

    df_grouped["%_restricted"] = np.where(
        df_grouped["val_geracaoreferencia"] > 0,
        100 * df_grouped["geracao_restrita"] / df_grouped["val_geracaoreferencia"],
        0
    )

    df_grouped["%_ENE"] = np.where(
        df_grouped["val_geracaoreferencia"] > 0,
        100 * df_grouped["geracao_restrita_ENE"] / df_grouped["val_geracaoreferencia"],
        0
    )

    df_grouped["TWh_restrita"] = df_grouped["geracao_restrita"] / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    ene_orange = "#ff7f0e"

    bars_total = ax.bar(
        x,
        df_grouped["%_restricted"],
        width=0.7,
        color=cor_hex,
        label="Total"
    )

    ax.bar(
        x,
        df_grouped["%_ENE"],
        width=0.7,
        facecolor="none",
        edgecolor=ene_orange,
        hatch="///",
        linewidth=2.0,
        label="ENE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Curtailment (%)")
    ax.set_xlabel("Dia da semana")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        handles=[
            Patch(facecolor=cor_hex, edgecolor=cor_hex, label="Total"),
            Patch(facecolor="none", edgecolor=ene_orange, hatch="///", linewidth=2.0, label="ENE")
        ],
        loc="upper right",
        fontsize=10
    )

    offset = (
        df_grouped["%_restricted"].max() * 0.05
        if df_grouped["%_restricted"].max() > 0
        else 0.2
    )

    for bar, twh in zip(bars_total, df_grouped["TWh_restrita"]):
        if twh > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{twh:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
            )

    plt.tight_layout()
    return fig


def gerar_grafico_hora_dia(df_filtrado, cor_hex, titulo):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14})

    if df_filtrado.empty:
        return None

    df_grouped = df_filtrado.groupby("Hora")[
        ["geracao_restrita", "val_geracaoreferencia", "geracao_restrita_ENE"]
    ].sum().reindex(range(24), fill_value=0).reset_index()

    df_grouped["%_restricted"] = np.where(
        df_grouped["val_geracaoreferencia"] > 0,
        100 * df_grouped["geracao_restrita"] / df_grouped["val_geracaoreferencia"],
        0
    )

    df_grouped["%_ENE"] = np.where(
        df_grouped["val_geracaoreferencia"] > 0,
        100 * df_grouped["geracao_restrita_ENE"] / df_grouped["val_geracaoreferencia"],
        0
    )

    df_grouped["TWh_restrita"] = df_grouped["geracao_restrita"] / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    ene_orange = "#ff7f0e"

    bars = ax.bar(
        x,
        df_grouped["%_restricted"],
        width=0.75,
        color=cor_hex,
        label="Total"
    )

    ax.bar(
        x,
        df_grouped["%_ENE"],
        width=0.75,
        facecolor="none",
        edgecolor=ene_orange,
        hatch="///",
        linewidth=2.0,
        label="ENE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Curtailment (%)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        handles=[
            Patch(facecolor=cor_hex, edgecolor=cor_hex, label="Total"),
            Patch(facecolor="none", edgecolor=ene_orange, hatch="///", linewidth=2.0, label="ENE")
        ],
        loc="upper right",
        fontsize=10
    )

    offset = (
        df_grouped["%_restricted"].max() * 0.05
        if df_grouped["%_restricted"].max() > 0
        else 0.2
    )

    for bar, twh in zip(bars, df_grouped["TWh_restrita"]):
        if twh > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{twh:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
            )

    plt.tight_layout()
    return fig

def gerar_grafico_geracao_vs_ref_dia(df_filtrado, cor_hex, titulo):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14})

    if df_filtrado.empty:
        return None

    df_filtrado = df_filtrado.copy()
    df_filtrado["Dow"] = df_filtrado["Data"].dt.dayofweek
    dow_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    df_grouped = df_filtrado.groupby("Dow")[
        ["val_geracao", "val_geracaoreferencia"]
    ].sum().reindex(range(7), fill_value=0).reset_index()

    df_grouped["TWh_val_geracao"] = df_grouped["val_geracao"] / 1e6
    df_grouped["TWh_val_referencia"] = df_grouped["val_geracaoreferencia"] / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    width = 0.7

    ax.bar(
        x,
        df_grouped["TWh_val_referencia"],
        width=width,
        color=cor_hex,
        alpha=0.25,
        edgecolor=cor_hex,
        linewidth=1.5,
        hatch="////",
        label="Geração de referência (TWh)"
    )

    bars_gen = ax.bar(
        x,
        df_grouped["TWh_val_geracao"],
        width=width,
        color=cor_hex,
        alpha=0.85,
        edgecolor=cor_hex,
        linewidth=1.5,
        label="Geração verificada (TWh)",
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Energia (TWh)")
    ax.set_xlabel("Dia da semana")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
        fontsize=10
    )

    y_max = df_grouped["TWh_val_referencia"].max()
    offset = y_max * 0.05 if y_max > 0 else 0.05

    for bar, val in zip(bars_gen, df_grouped["TWh_val_geracao"]):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
            )

    plt.tight_layout()
    return fig


def gerar_grafico_geracao_vs_ref_hora(df_filtrado, cor_hex, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 10
    })

    if df_filtrado.empty:
        return None

    colunas_necessarias = [
        "val_geracao",
        "geracao_restrita_REL",
        "geracao_restrita_CNF",
        "geracao_restrita_ENE"
    ]

    faltando = [col for col in colunas_necessarias if col not in df_filtrado.columns]

    if faltando:
        st.error(f"Faltam colunas no arquivo parquet: {faltando}")
        return None

    df_grouped = df_filtrado.groupby("Hora")[
        colunas_necessarias
    ].sum().reindex(range(24), fill_value=0).reset_index()

    df_grouped["TWh_val_geracao"] = df_grouped["val_geracao"] / 1e6
    df_grouped["TWh_REL"] = df_grouped["geracao_restrita_REL"] / 1e6
    df_grouped["TWh_CNF"] = df_grouped["geracao_restrita_CNF"] / 1e6
    df_grouped["TWh_ENE"] = df_grouped["geracao_restrita_ENE"] / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    width = 0.75

    ax.bar(
        x,
        df_grouped["TWh_val_geracao"],
        width=width,
        color=cor_hex,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
        label="Geração verificada (TWh)"
    )

    bottom = df_grouped["TWh_val_geracao"].to_numpy()

    cores_razao = {
        "REL": "#1f77b4",
        "CNF": "#2ca02c",
        "ENE": "#ff7f0e",
    }

    for razao in ["REL", "CNF", "ENE"]:
        valores = df_grouped[f"TWh_{razao}"].to_numpy()

        ax.bar(
            x,
            valores,
            width=width,
            bottom=bottom,
            facecolor="none",
            edgecolor=cores_razao[razao],
            hatch="///",
            linewidth=2.0,
            zorder=4,
            label=f"Corte {razao} (TWh)"
        )

        bottom = bottom + valores

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Energia (TWh)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        handles=[
            Patch(facecolor=cor_hex, edgecolor="black", label="Geração verificada (TWh)"),
            Patch(facecolor="none", edgecolor=cores_razao["REL"], hatch="///", linewidth=2.0, label="Corte REL (TWh)"),
            Patch(facecolor="none", edgecolor=cores_razao["CNF"], hatch="///", linewidth=2.0, label="Corte CNF (TWh)"),
            Patch(facecolor="none", edgecolor=cores_razao["ENE"], hatch="///", linewidth=2.0, label="Corte ENE (TWh)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True
    )

    plt.tight_layout()
    return fig


def gerar_grafico_carga_liquida_trimestres(df_balanco, ano, titulo, trimestres_selecionados=None):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 9
    })

    if df_balanco.empty:
        return None

    df_in = df_balanco.copy()
    if "Ano" not in df_in.columns:
        df_in["Ano"] = df_in["Data"].dt.year
    if "Mes" not in df_in.columns:
        df_in["Mes"] = df_in["Data"].dt.month

    df_in = df_in[df_in["Ano"] == int(ano)].copy()
    if df_in.empty:
        return None

    quarters = {
        1: ("T1 (Jan-Mar)", 1, 3, "-"),
        2: ("T2 (Abr-Jun)", 4, 6, "--"),
        3: ("T3 (Jul-Set)", 7, 9, ":"),
        4: ("T4 (Out-Dez)", 10, 12, "-."),
    }
    selecionados = [1, 2, 3, 4] if trimestres_selecionados is None else trimestres_selecionados

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    plotou = False

    for q in selecionados:
        if q not in quarters:
            continue
        q_name, m1, m2, estilo = quarters[q]
        df_q = df_in[(df_in["Mes"] >= m1) & (df_in["Mes"] <= m2)].copy()
        if df_q.empty:
            continue

        carga = df_q.groupby("Hora")["val_carga"].mean().reindex(range(24))
        solar = df_q.groupby("Hora")["val_gersolar"].mean().reindex(range(24))
        liquida = carga - solar

        ax.plot(x, carga / 1e3, linestyle=estilo, linewidth=2.0, color="#6f42c1", label=f"{q_name} — carga")
        ax.plot(x, liquida / 1e3, linestyle=estilo, linewidth=2.0, color="#17a2b8", label=f"{q_name} — carga - solar")
        plotou = True

    if not plotou:
        plt.close(fig)
        return None

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Média (GW)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, frameon=True)
    plt.tight_layout()
    return fig


def gerar_grafico_carga_dia_semana(df_filtrado, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14
    })

    if df_filtrado.empty:
        return None

    df_in = df_filtrado.copy()
    df_in["Dow"] = df_in["Data"].dt.dayofweek

    dow_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    x = np.arange(7)

    df_grouped = (
        df_in.groupby("Dow")[["val_carga"]]
        .sum()
        .reindex(range(7), fill_value=0)
        .reset_index()
    )

    df_grouped["TWh_carga"] = df_grouped["val_carga"] / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        x,
        df_grouped["TWh_carga"],
        width=0.7,
        color="#6f42c1",
        edgecolor="#6f42c1",
        alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Carga (TWh)")
    ax.set_xlabel("Dia da semana")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    y_max = df_grouped["TWh_carga"].max()
    offset = y_max * 0.02 if y_max > 0 else 0.2

    for bar, twh in zip(bars, df_grouped["TWh_carga"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{twh:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
        )

    plt.tight_layout()
    return fig


def gerar_grafico_fotovoltaica_vs_mmgd_hora(df_filtrado, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 10
    })

    if df_filtrado.empty or "val_geracao_fotovoltaica" not in df_filtrado.columns:
        return None

    mmgd_col = (
        "val_geracao_solar_mmgd"
        if "val_geracao_solar_mmgd" in df_filtrado.columns
        else "val_geracao_mmgd"
    )
    if mmgd_col not in df_filtrado.columns:
        return None

    df_grouped = (
        df_filtrado.groupby("Hora")[["val_geracao_fotovoltaica", mmgd_col]]
        .sum()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )

    # A base de usinas já é horária. Não há divisão por 2.
    df_grouped["TWh_fotovoltaica"] = df_grouped["val_geracao_fotovoltaica"] / 1e6
    df_grouped["TWh_mmgd"] = df_grouped[mmgd_col] / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    label_mmgd = "MMGD Solar (TWh)" if mmgd_col == "val_geracao_solar_mmgd" else "MMGD (TWh)"

    ax.bar(x, df_grouped["TWh_fotovoltaica"], width=0.75, color="#6c757d", edgecolor="#6c757d", alpha=0.85, label="Fotovoltaica (TWh)")
    ax.bar(x, df_grouped["TWh_mmgd"], width=0.75, facecolor="none", edgecolor="black", hatch="///", linewidth=2.0, label=label_mmgd)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Energia acumulada no período (TWh)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)
    plt.tight_layout()
    return fig


def gerar_grafico_participacao_geracao(df_balanco, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    colunas = [
        "val_gerhidraulica",
        "val_gertermica",
        "val_gereolica",
        "val_gersolar"
    ]

    faltando = [c for c in colunas if c not in df_balanco.columns]

    if faltando:
        st.error(f"Faltam colunas na base de balanço: {faltando}")
        return None

    df_grouped = df_balanco.groupby("Ano")[colunas].sum()
    df_percent = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

    colors = {
        "val_gerhidraulica": "#1f77b4",
        "val_gertermica": "#ff7f0e",
        "val_gereolica": "#2ca02c",
        "val_gersolar": "#d62728"
    }

    legend_names = {
        "val_gereolica": "Eólico",
        "val_gersolar": "Fotovoltaico",
        "val_gerhidraulica": "Hidráulica",
        "val_gertermica": "Térmica"
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for ano in df_percent.index:
        valores = df_percent.loc[ano].sort_values(ascending=False)
        bottom = 0

        for fonte, valor in valores.items():
            ax.bar(
                ano,
                valor,
                bottom=bottom,
                color=colors[fonte],
                edgecolor="black",
                hatch="//" if fonte in ["val_gereolica", "val_gersolar"] else ""
            )
            bottom += valor

    ax.set_xticks(df_percent.index)
    ax.set_xticklabels(df_percent.index, rotation=90)
    ax.set_ylabel("Geração (%)")
    ax.set_xlabel("Ano")
    ax.set_ylim(0, 100)
    ax.set_title(titulo)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    handles = [
        Patch(
            facecolor=colors[key],
            edgecolor="black",
            hatch="//" if key in ["val_gereolica", "val_gersolar"] else "",
            label=legend_names[key]
        )
        for key in legend_names.keys()
    ]

    ax.legend(
        handles=handles,
        title="Fonte de geração",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    return fig



# ==========================================
# 5.1 Novas análises: IBR, resumos e estabilidade
# ==========================================
def calcular_indicadores_balanco(df):
    """Calcula participações normalizadas a partir das médias/valores disponíveis."""
    if df.empty:
        return pd.DataFrame()

    required = ["val_carga", "val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    den = out["val_carga"].where(out["val_carga"] > 0)
    out["hydro_norm"] = 100.0 * out["val_gerhidraulica"] / den
    out["term_norm"] = 100.0 * out["val_gertermica"] / den
    out["eol_norm"] = 100.0 * out["val_gereolica"] / den
    out["sol_norm"] = 100.0 * out["val_gersolar"] / den
    out["sync_norm"] = 100.0 * (out["val_gerhidraulica"] + out["val_gertermica"]) / den
    out["ibr_norm"] = 100.0 * (out["val_gereolica"] + out["val_gersolar"]) / den
    out["penetracao_ibr"] = out["ibr_norm"]
    return out


def gerar_grafico_indicador_hora_ano(df, indicador, label_y, titulo):
    """Reproduz a lógica original: média horária por ano e depois cálculo da razão."""
    if df.empty:
        return None

    required = ["Ano", "Hora", "val_carga", "val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]
    if any(c not in df.columns for c in required):
        return None

    cols_val = ["val_carga", "val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]
    grouped = df.groupby(["Ano", "Hora"])[cols_val].mean().reset_index()
    grouped = calcular_indicadores_balanco(grouped)
    if grouped.empty or indicador not in grouped.columns:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    anos = sorted(grouped["Ano"].dropna().unique())
    for ano in anos:
        serie = grouped[grouped["Ano"] == ano].set_index("Hora")[indicador].reindex(range(24))
        ax.plot(range(24), serie.values, marker="o", linewidth=1.8, label=str(int(ano)))

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel(label_y)
    ax.set_title(titulo)
    ax.grid(True, linestyle="--", alpha=0.6)
    if anos:
        ax.legend(title="Ano", ncol=min(7, len(anos)))
    plt.tight_layout()
    return fig


def gerar_tabela_medias_anuais_balanco(df):
    """Médias anuais por subsistema e participação de cada fonte na geração total."""
    if df.empty:
        return pd.DataFrame()

    cols = ["val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar", "val_carga"]
    if any(c not in df.columns for c in cols):
        return pd.DataFrame()

    base = df.copy()
    if "Ano" not in base.columns:
        base["Ano"] = base["Data"].dt.year

    grupo = ["Ano"]
    if "id_subsistema" in base.columns:
        grupo = ["id_subsistema", "Ano"]

    medias = base.groupby(grupo)[cols].mean().reset_index()
    medias = calcular_indicadores_balanco(medias)

    total_ger = medias[["val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]].sum(axis=1).replace(0, np.nan)
    for col in ["val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]:
        medias[f"{col}_percentual"] = 100.0 * medias[col] / total_ger

    return medias


def resumo_anual_curtailment(df, ano):
    """Resumo anual usando o parquet horário, que já está em MWh."""
    if df.empty:
        return None

    required = ["Data", "geracao_restrita", "val_geracao", "val_geracaoreferencia"]
    if any(c not in df.columns for c in required):
        return None

    base = df[df["Data"].dt.year == int(ano)].copy()
    if base.empty:
        return None

    total_curt = pd.to_numeric(base["geracao_restrita"], errors="coerce").sum()
    total_ger = pd.to_numeric(base["val_geracao"], errors="coerce").sum()
    total_ref = pd.to_numeric(base["val_geracaoreferencia"], errors="coerce").sum()
    perc = 100.0 * total_curt / total_ref if total_ref > 0 else 0.0

    return {
        "curtailment_mwh": total_curt,
        "geracao_mwh": total_ger,
        "referencia_mwh": total_ref,
        "percentual": perc,
        "data_min": base["Data"].min(),
        "data_max": base["Data"].max(),
        "meses": int(base["Data"].dt.month.nunique()),
    }


def gerar_tabela_comparacao_anual_curtailment(df):
    """Tabela anual total do SIN com G, C, R, %Curt e deltas quando anos são completos."""
    if df.empty:
        return pd.DataFrame()

    required = ["Data", "geracao_restrita", "val_geracao", "val_geracaoreferencia"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    base = df.copy()
    base["Ano"] = base["Data"].dt.year

    resumo = (
        base.groupby("Ano")[["val_geracao", "geracao_restrita", "val_geracaoreferencia"]]
        .sum()
        .reset_index()
    )
    cobertura = (
        base.groupby("Ano")["Data"]
        .agg(Data_Inicial="min", Data_Final="max", Meses=lambda s: s.dt.month.nunique())
        .reset_index()
    )
    resumo = resumo.merge(cobertura, on="Ano", how="left")

    # Os parquets de curtailment do dashboard já foram convertidos para MWh.
    resumo["G_GWh"] = resumo["val_geracao"] / 1e3
    resumo["C_GWh"] = resumo["geracao_restrita"] / 1e3
    resumo["R_GWh"] = resumo["val_geracaoreferencia"] / 1e3
    resumo["Curtailment_%"] = np.where(
        resumo["R_GWh"] > 0,
        100.0 * resumo["C_GWh"] / resumo["R_GWh"],
        0.0,
    )

    resumo["Ano_Completo"] = resumo["Meses"] == 12
    for col, nome in [
        ("G_GWh", "Delta_G_%"),
        ("C_GWh", "Delta_C_%"),
        ("R_GWh", "Delta_R_%"),
        ("Curtailment_%", "Delta_Curt_%"),
    ]:
        delta = resumo[col].pct_change() * 100.0
        anterior_completo = resumo["Ano_Completo"].shift(1).fillna(False).astype(bool)
        atual_completo = resumo["Ano_Completo"].astype(bool)
        resumo[nome] = delta.where(anterior_completo & atual_completo, np.nan)

    return resumo[[
        "Ano", "Meses", "Data_Inicial", "Data_Final",
        "G_GWh", "Delta_G_%", "C_GWh", "Delta_C_%",
        "R_GWh", "Delta_R_%", "Curtailment_%", "Delta_Curt_%",
        "Ano_Completo",
    ]]


def gerar_tabela_estado_subsistema_anual(df, anos_selecionados=None):
    """Resumo anual por id_subsistema e id_estado usando Parquet já convertido para MWh."""
    required = ["Data", "id_subsistema", "id_estado", "val_geracao", "geracao_restrita", "val_geracaoreferencia"]
    if df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame()

    base = df[required].copy()
    base["Ano"] = base["Data"].dt.year
    if anos_selecionados:
        base = base[base["Ano"].isin([int(a) for a in anos_selecionados])]
    if base.empty:
        return pd.DataFrame()

    grupo = ["id_subsistema", "id_estado", "Ano"]
    resumo = (
        base.groupby(grupo)[["val_geracao", "geracao_restrita", "val_geracaoreferencia"]]
        .sum()
        .reset_index()
    )
    cobertura = (
        base.groupby(grupo)["Data"]
        .agg(Data_Inicial="min", Data_Final="max", Meses=lambda s: s.dt.month.nunique())
        .reset_index()
    )
    resumo = resumo.merge(cobertura, on=grupo, how="left")
    resumo["G_GWh"] = resumo["val_geracao"] / 1e3
    resumo["C_GWh"] = resumo["geracao_restrita"] / 1e3
    resumo["R_GWh"] = resumo["val_geracaoreferencia"] / 1e3
    resumo["Curtailment_%"] = np.where(resumo["R_GWh"] > 0, 100.0 * resumo["C_GWh"] / resumo["R_GWh"], 0.0)
    resumo["Ano_Completo"] = resumo["Meses"] == 12

    resumo = resumo.sort_values(["id_subsistema", "id_estado", "Ano"]).reset_index(drop=True)
    por_local = resumo.groupby(["id_subsistema", "id_estado"], sort=False)
    resumo["Ano_Anterior"] = por_local["Ano"].shift(1)
    resumo["Completo_Anterior"] = por_local["Ano_Completo"].shift(1).fillna(False)

    for col, nome in [("G_GWh", "Delta_G_%"), ("C_GWh", "Delta_C_%"), ("R_GWh", "Delta_R_%"), ("Curtailment_%", "Delta_Curt_%")]:
        anterior = por_local[col].shift(1)
        delta = np.where(anterior.abs() > 1e-12, 100.0 * (resumo[col] - anterior) / anterior.abs(), np.nan)
        anos_consecutivos = resumo["Ano"] == (resumo["Ano_Anterior"] + 1)
        valido = anos_consecutivos & resumo["Ano_Completo"] & resumo["Completo_Anterior"].astype(bool)
        resumo[nome] = np.where(valido, delta, np.nan)

    return resumo[[
        "id_subsistema", "id_estado", "Ano", "Meses", "Data_Inicial", "Data_Final",
        "G_GWh", "Delta_G_%", "C_GWh", "Delta_C_%", "R_GWh", "Delta_R_%",
        "Curtailment_%", "Delta_Curt_%", "Ano_Completo"
    ]]


def gerar_latex_tabela_total(tabela):
    """Gera uma versão LaTeX simples da tabela anual total exibida no Streamlit."""
    if tabela.empty:
        return ""

    linhas = []
    for _, row in tabela.iterrows():
        vals = []
        for col in ["G_GWh", "Delta_G_%", "C_GWh", "Delta_C_%", "R_GWh", "Delta_R_%", "Curtailment_%", "Delta_Curt_%"]:
            val = row[col]
            vals.append("-" if pd.isna(val) else f"{val:.2f}")
        linhas.append(f"{int(row['Ano'])} & " + " & ".join(vals) + r" \\")

    corpo = "\n".join(linhas)
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Comparação anual total (SIN)}\n"
        "\\label{tab:total_anual}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "Ano & G (GWh) & $\\Delta\\%$ & C (GWh) & $\\Delta\\%$ & R (GWh) & $\\Delta\\%$ & \\%Curt & $\\Delta\\%$ \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _criar_instante(df):
    out = df.copy()
    out["Data"] = pd.to_datetime(out["Data"], errors="coerce")
    out["Hora"] = pd.to_numeric(out["Hora"], errors="coerce")
    out = out.dropna(subset=["Data", "Hora"])
    out["Hora"] = out["Hora"].astype(int)
    out["Instante"] = out["Data"].dt.normalize() + pd.to_timedelta(out["Hora"], unit="h")
    return out


def preparar_series_solar_estabilidade(df_usina, df_solar):
    """Séries horárias: MMGD solar, solar centralizada e centralizada + curtailment."""
    required_usina = ["Data", "Hora", "val_geracao_solar_mmgd", "val_geracao_solar_centralizada"]
    if df_usina.empty or df_solar.empty or any(c not in df_usina.columns for c in required_usina):
        return pd.DataFrame()
    if "geracao_restrita" not in df_solar.columns:
        return pd.DataFrame()

    usina = _criar_instante(df_usina[required_usina])
    usina = (
        usina.groupby("Instante")[["val_geracao_solar_mmgd", "val_geracao_solar_centralizada"]]
        .sum()
        .reset_index()
        .rename(columns={
            "val_geracao_solar_mmgd": "mmgd_mwh",
            "val_geracao_solar_centralizada": "solar_central_mwh",
        })
    )

    # Curtailment possui linhas por estado/subsistema; somamos para obter o total nacional por hora.
    restrita = _criar_instante(df_solar[["Data", "Hora", "geracao_restrita"]])
    restrita = (
        restrita.groupby("Instante")["geracao_restrita"]
        .sum()
        .reset_index()
        .rename(columns={"geracao_restrita": "solar_restrita_mwh"})
    )

    base = usina.merge(restrita, on="Instante", how="outer")
    for col in ["mmgd_mwh", "solar_central_mwh", "solar_restrita_mwh"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)
    base["solar_central_mais_restrita_mwh"] = base["solar_central_mwh"] + base["solar_restrita_mwh"]
    return base.sort_values("Instante")


def preparar_series_eolica_estabilidade(df_usina, df_eolico):
    """Séries horárias: geração eólica e geração eólica + curtailment."""
    required_usina = ["Data", "Hora", "val_geracao_eolica"]
    if df_usina.empty or df_eolico.empty or any(c not in df_usina.columns for c in required_usina):
        return pd.DataFrame()
    if "geracao_restrita" not in df_eolico.columns:
        return pd.DataFrame()

    ger = _criar_instante(df_usina[required_usina])
    ger = (
        ger.groupby("Instante")["val_geracao_eolica"]
        .sum()
        .reset_index()
        .rename(columns={"val_geracao_eolica": "eolica_mwh"})
    )

    rest = _criar_instante(df_eolico[["Data", "Hora", "geracao_restrita"]])
    rest = (
        rest.groupby("Instante")["geracao_restrita"]
        .sum()
        .reset_index()
        .rename(columns={"geracao_restrita": "eolica_restrita_mwh"})
    )

    base = ger.merge(rest, on="Instante", how="outer")
    base[["eolica_mwh", "eolica_restrita_mwh"]] = base[["eolica_mwh", "eolica_restrita_mwh"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    base["eolica_mais_restrita_mwh"] = base["eolica_mwh"] + base["eolica_restrita_mwh"]
    return base.sort_values("Instante")


def _filtrar_e_agregar_estabilidade(df, start_date, end_date, colunas, freq):
    if df.empty:
        return pd.DataFrame()

    inicio = pd.to_datetime(start_date)
    fim_exclusivo = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    base = df[(df["Instante"] >= inicio) & (df["Instante"] < fim_exclusivo)].copy()
    if base.empty:
        return pd.DataFrame()

    base = base.set_index("Instante")[colunas]
    if freq:
        base = base.resample(freq).sum(min_count=1)
    return base


def gerar_grafico_solar_estabilidade(df_usina, df_solar, start_date, end_date, freq="D"):
    base = preparar_series_solar_estabilidade(df_usina, df_solar)
    cols = ["mmgd_mwh", "solar_central_mwh", "solar_central_mais_restrita_mwh"]
    serie = _filtrar_e_agregar_estabilidade(base, start_date, end_date, cols, freq)
    if serie.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))
    if serie["mmgd_mwh"].notna().any():
        ax.plot(serie.index, serie["mmgd_mwh"], label="MMGD Solar", linewidth=1.8)
    ax.plot(serie.index, serie["solar_central_mwh"], label="Solar centralizada", linewidth=1.8)
    ax.plot(
        serie.index,
        serie["solar_central_mais_restrita_mwh"],
        label="Centralizada + restrita",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Energia (MWh)")
    ax.set_title("Solar: MMGD, geração centralizada e geração centralizada + restrita")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(ncol=3)
    plt.tight_layout()
    return fig


def gerar_grafico_eolica_estabilidade(df_usina, df_eolico, start_date, end_date, freq="D"):
    base = preparar_series_eolica_estabilidade(df_usina, df_eolico)
    cols = ["eolica_mwh", "eolica_mais_restrita_mwh"]
    serie = _filtrar_e_agregar_estabilidade(base, start_date, end_date, cols, freq)
    if serie.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie["eolica_mwh"], label="Geração eólica", linewidth=1.8)
    ax.plot(
        serie.index,
        serie["eolica_mais_restrita_mwh"],
        label="Geração + restrita",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Energia (MWh)")
    ax.set_title("Eólica: geração e geração + restrita")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


def calcular_cv_por_hora(base, coluna, start_date, end_date, hora_inicio, hora_fim):
    if base.empty or coluna not in base.columns:
        return pd.Series(dtype=float)

    inicio = pd.to_datetime(start_date)
    fim_exclusivo = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df = base[(base["Instante"] >= inicio) & (base["Instante"] < fim_exclusivo)].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["Hora"] = df["Instante"].dt.hour
    df = df[(df["Hora"] >= hora_inicio) & (df["Hora"] <= hora_fim)]
    estat = df.groupby("Hora")[coluna].agg(["mean", "std"])
    estat["cv"] = np.where(estat["mean"].abs() > 1e-12, estat["std"] / estat["mean"].abs(), np.nan)
    return estat["cv"].reindex(range(hora_inicio, hora_fim + 1))


def gerar_grafico_cv_periodo(base, series_config, start_date, end_date, hora_inicio, hora_fim, titulo):
    if base.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    plotou = False
    for coluna, label, linestyle in series_config:
        cv = calcular_cv_por_hora(base, coluna, start_date, end_date, hora_inicio, hora_fim)
        if cv.notna().any():
            ax.plot(cv.index, cv.values, label=label, linewidth=2, linestyle=linestyle)
            plotou = True

    if not plotou:
        plt.close(fig)
        return None

    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Coeficiente de Variação (CV)")
    ax.set_xticks(range(hora_inicio, hora_fim + 1))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_title(titulo)
    plt.tight_layout()
    return fig


def _intervalo_trimestre(ano, trimestre):
    mes_inicio = 1 + (int(trimestre) - 1) * 3
    inicio = pd.Timestamp(int(ano), mes_inicio, 1)
    if trimestre == 4:
        fim = pd.Timestamp(int(ano) + 1, 1, 1)
    else:
        fim = pd.Timestamp(int(ano), mes_inicio + 3, 1)
    return inicio, fim


def gerar_grafico_cv_trimestres(base, series_config, ano, trimestres, hora_inicio, hora_fim, titulo):
    trimestres = sorted(set(int(q) for q in trimestres))
    if base.empty or not trimestres:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes_flat = axes.flatten()
    plotou_algum = False

    for ax in axes_flat:
        ax.set_visible(False)

    for ax, trimestre in zip(axes_flat, trimestres):
        ax.set_visible(True)
        inicio, fim_exclusivo = _intervalo_trimestre(ano, trimestre)
        # calcular_cv_por_hora trabalha com fim inclusivo; usamos o dia anterior ao início do trimestre seguinte.
        fim_inclusivo = fim_exclusivo - pd.Timedelta(days=1)
        plotou = False

        for coluna, label, linestyle in series_config:
            cv = calcular_cv_por_hora(base, coluna, inicio, fim_inclusivo, hora_inicio, hora_fim)
            if cv.notna().any():
                ax.plot(cv.index, cv.values, label=label, linewidth=1.8, linestyle=linestyle)
                plotou = True

        ax.set_title(f"T{trimestre} — {int(ano)}")
        ax.set_xticks(range(hora_inicio, hora_fim + 1, 2 if hora_fim - hora_inicio > 12 else 1))
        ax.set_xlabel("Hora")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=0)
        if plotou:
            plotou_algum = True

    if not plotou_algum:
        plt.close(fig)
        return None

    axes_flat[0].set_ylabel("CV")
    axes_flat[2].set_ylabel("CV")

    handles, labels = [], []
    for ax in axes_flat:
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(titulo, y=1.01, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ==========================================
# 6. Lógica de Exibição
# ==========================================
if analise_escolhida == "0. Sumário e Guia do Dashboard":
    st.markdown("## Sumário do Dashboard")
    st.markdown(
        """
        O dashboard integra **curtailment eólico e solar**, **carga**, **geração por fonte**,
        **penetração IBR**, **participação da geração síncrona** e análises de
        **variabilidade de eólica e solar**. Os limites de datas são lidos diretamente dos
        arquivos Parquet, portanto novas atualizações das bases passam a aparecer sem alterar
        datas fixas no `main.py`.
        """
    )

    if datas_disponiveis:
        st.info(f"Período detectado nas bases: {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}.")

    st.markdown("### Como os filtros funcionam")
    st.markdown(
        """
        - **Fonte de curtailment:** Eólico, Solar ou Eólico + Solar.
        - **Escopo do curtailment:** Total, Subsistema ou Estado, quando `id_subsistema` e `id_estado` estiverem nos Parquets.
        - **Subsistema do balanço:** usado nas análises de carga, participação e IBR; isso evita somar o SIN com seus subsistemas.
        - **Período:** livre, limitado somente pela cobertura real das bases.
        - **CV e trimestres:** o período, ano e trimestres são escolhidos no próprio dashboard.
        """
    )

    st.markdown("### Análises disponíveis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Curtailment**  
            **1.** Curtailment por Dia da Semana  
            **2.** Curtailment por Hora do Dia  
            **3.** Geração vs Referência por Dia da Semana  
            **4.** Geração vs Referência por Hora, com REL/CNF/ENE  
            **10.** Resumo Anual de Curtailment  
            **11.** Comparação Anual e tabela por Estado/Subsistema

            **Sistema elétrico**  
            **5.** Carga e Carga Líquida por Hora, com escolha de ano e trimestres  
            **6.** Carga por Dia da Semana  
            **8.** Participação da Geração por Fonte  
            **9.** IBR, Geração Síncrona, participações e Médias Anuais
            """
        )
    with c2:
        st.markdown(
            """
            **Solar**  
            **7.** Fotovoltaica vs MMGD por Hora  
            **12.** MMGD Solar × Centralizada × Centralizada + Restrita  
            **13.** Variabilidade Solar (CV), por período ou trimestres

            **Eólica**  
            **14.** Geração Eólica × Geração + Restrita  
            **15.** Variabilidade Eólica (CV), por período ou trimestres
            """
        )

    st.markdown("### Estrutura esperada das bases")
    status = []
    status.append({
        "Base": "Curtailment",
        "Período": f"{df_eolico['Data'].min().date()} a {df_eolico['Data'].max().date()}" if not df_eolico.empty else "Não carregada",
        "Estado/Subsistema": "OK" if all(c in df_eolico.columns for c in ["id_estado", "id_subsistema"]) else "Atualizar Parquet",
    })
    status.append({
        "Base": "Balanço",
        "Período": f"{df_balanco['Data'].min().date()} a {df_balanco['Data'].max().date()}" if not df_balanco.empty else "Não carregada",
        "Estado/Subsistema": "OK" if "id_subsistema" in df_balanco.columns else "Atualizar Parquet",
    })
    status.append({
        "Base": "Usinas",
        "Período": f"{df_usina['Data'].min().date()} a {df_usina['Data'].max().date()}" if not df_usina.empty else "Não carregada",
        "Estado/Subsistema": "Séries solar/eólica OK" if all(c in df_usina.columns for c in ["val_geracao_solar_mmgd", "val_geracao_solar_centralizada", "val_geracao_eolica"]) else "Atualizar Parquet",
    })
    st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)

    st.caption(
        "Unidades: os Parquets de curtailment já estão convertidos para MWh; a base de usinas é horária e não recebe divisão por 2. "
        "As participações do balanço são calculadas a partir de carga e geração por fonte."
    )

elif len(intervalo_datas) == 2:
    start_date = pd.to_datetime(intervalo_datas[0])
    end_date = pd.to_datetime(intervalo_datas[1])

    df_curtailment_total, cor_fonte, nome_fonte = escolher_base_curtailment(df_eolico, df_solar, fonte_escolhida)
    df_curtailment = filtrar_escopo_curtailment(
        df_curtailment_total, escopo_curtailment, valor_escopo_curtailment
    )
    nome_escopo = descricao_escopo(escopo_curtailment, valor_escopo_curtailment)

    df_balanco_sub = filtrar_balanco_subsistema(df_balanco, subsistema_balanco)

    mask_c = (df_curtailment["Data"] >= start_date) & (df_curtailment["Data"] <= end_date) if not df_curtailment.empty else pd.Series(False, index=df_curtailment.index)
    mask_b = (df_balanco_sub["Data"] >= start_date) & (df_balanco_sub["Data"] <= end_date) if not df_balanco_sub.empty else pd.Series(False, index=df_balanco_sub.index)
    mask_u = (df_usina["Data"] >= start_date) & (df_usina["Data"] <= end_date) if not df_usina.empty else pd.Series(False, index=df_usina.index)

    df_filtro_curtailment = df_curtailment.loc[mask_c].copy() if not df_curtailment.empty else pd.DataFrame()
    df_filtro_balanco = df_balanco_sub.loc[mask_b].copy() if not df_balanco_sub.empty else pd.DataFrame()
    df_filtro_usina = df_usina.loc[mask_u].copy() if not df_usina.empty else pd.DataFrame()

    analises_sem_periodo_cabecalho = {
        "5. Carga e Carga Líquida por Hora",
        "10. Resumo Anual de Curtailment",
        "11. Comparação Anual e por Estado/Subsistema",
    }
    if analise_escolhida not in analises_sem_periodo_cabecalho:
        st.subheader(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")

    if analise_escolhida in {
        "1. Curtailment por Dia da Semana",
        "2. Curtailment por Hora do Dia",
        "3. Geração vs Referência (Dia da Semana)",
        "4. Geração vs Referência (Hora do Dia)",
        "10. Resumo Anual de Curtailment",
        "11. Comparação Anual e por Estado/Subsistema",
    }:
        st.caption(f"Fonte: {nome_fonte} | Escopo: {nome_escopo}")

    if analise_escolhida == "1. Curtailment por Dia da Semana":
        st.markdown(f"### Curtailment por dia da semana — {nome_fonte}")
        fig = gerar_grafico_dia_semana(df_filtro_curtailment, cor_fonte, f"Perfil Semanal - {nome_fonte} - {nome_escopo}")
        st.pyplot(fig) if fig else st.warning("Sem dados para o período/escopo selecionado.")

    elif analise_escolhida == "2. Curtailment por Hora do Dia":
        st.markdown(f"### Curtailment por hora do dia — {nome_fonte}")
        fig = gerar_grafico_hora_dia(df_filtro_curtailment, cor_fonte, f"Perfil Horário - {nome_fonte} - {nome_escopo}")
        st.pyplot(fig) if fig else st.warning("Sem dados para o período/escopo selecionado.")

    elif analise_escolhida == "3. Geração vs Referência (Dia da Semana)":
        st.markdown(f"### Geração vs referência por dia da semana — {nome_fonte}")
        fig = gerar_grafico_geracao_vs_ref_dia(df_filtro_curtailment, cor_fonte, f"Geração vs Referência - {nome_fonte} - {nome_escopo}")
        st.pyplot(fig) if fig else st.warning("Sem dados para o período/escopo selecionado.")

    elif analise_escolhida == "4. Geração vs Referência (Hora do Dia)":
        st.markdown(f"### Geração verificada + cortes por hora — {nome_fonte}")
        fig = gerar_grafico_geracao_vs_ref_hora(df_filtro_curtailment, cor_fonte, f"Geração Verificada + Cortes por Hora - {nome_fonte} - {nome_escopo}")
        st.pyplot(fig) if fig else st.warning("Sem dados para o período/escopo selecionado.")

    elif analise_escolhida == "5. Carga e Carga Líquida por Hora":
        st.markdown("### Carga média e carga líquida por hora")
        st.caption(f"Subsistema: {subsistema_balanco or 'base disponível'}")
        anos = sorted(df_balanco_sub["Data"].dropna().dt.year.unique().astype(int).tolist()) if not df_balanco_sub.empty else []
        if not anos:
            st.warning("Sem anos disponíveis na base de balanço.")
        else:
            ano = st.selectbox("Ano:", anos, index=len(anos) - 1, key="ano_carga")
            trimestres = st.multiselect("Trimestres:", [1, 2, 3, 4], default=[1, 2, 3, 4], key="trimestres_carga")
            fig = gerar_grafico_carga_liquida_trimestres(
                df_balanco_sub, ano, f"Carga média e carga líquida por hora - {ano} - {subsistema_balanco or ''}", trimestres
            )
            st.pyplot(fig) if fig else st.warning("Sem dados para o ano/trimestres selecionados.")

    elif analise_escolhida == "6. Carga por Dia da Semana":
        st.markdown("### Carga total por dia da semana")
        st.caption(f"Subsistema: {subsistema_balanco or 'base disponível'}")
        fig = gerar_grafico_carga_dia_semana(df_filtro_balanco, "Carga por Dia da Semana")
        st.pyplot(fig) if fig else st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "7. Fotovoltaica vs MMGD por Hora":
        st.markdown("### Geração fotovoltaica e MMGD por hora")
        st.caption("A base de usinas já é horária; portanto esta análise não aplica divisão por 2.")
        fig = gerar_grafico_fotovoltaica_vs_mmgd_hora(df_filtro_usina, "Geração Fotovoltaica vs MMGD por Hora")
        st.pyplot(fig) if fig else st.warning("Sem dados ou colunas necessárias para o período selecionado.")

    elif analise_escolhida == "8. Participação da Geração por Fonte":
        st.markdown("### Participação percentual da geração por fonte")
        st.caption(f"Subsistema: {subsistema_balanco or 'base disponível'}")
        fig = gerar_grafico_participacao_geracao(df_filtro_balanco, "Participação da geração por fonte")
        st.pyplot(fig) if fig else st.warning("Sem dados disponíveis.")

    elif analise_escolhida == "9. IBR, Geração Síncrona e Médias Anuais":
        st.markdown("### IBR, geração síncrona e participações por fonte")
        st.caption(f"Subsistema do gráfico: {subsistema_balanco or 'base disponível'}")

        indicadores = {
            "Penetração IBR": ("ibr_norm", "Participação IBR [%]"),
            "Geração síncrona": ("sync_norm", "Participação da geração síncrona [%]"),
            "Hidráulica": ("hydro_norm", "Participação hidráulica [%]"),
            "Térmica": ("term_norm", "Participação térmica [%]"),
            "Eólica": ("eol_norm", "Participação eólica [%]"),
            "Solar": ("sol_norm", "Participação solar [%]"),
        }
        escolha_ind = st.selectbox("Indicador do perfil horário:", list(indicadores.keys()))
        indicador, label_y = indicadores[escolha_ind]
        fig = gerar_grafico_indicador_hora_ano(
            df_filtro_balanco, indicador, label_y, f"{escolha_ind}: média horária por ano - {subsistema_balanco or ''}"
        )
        st.pyplot(fig) if fig else st.warning("Não foi possível calcular o indicador com a base disponível.")

        st.markdown("#### Médias anuais por subsistema")
        # Para a tabela usamos todos os subsistemas, respeitando o período global.
        if not df_balanco.empty:
            mask_all = (df_balanco["Data"] >= start_date) & (df_balanco["Data"] <= end_date)
            tabela = gerar_tabela_medias_anuais_balanco(df_balanco.loc[mask_all].copy())
        else:
            tabela = pd.DataFrame()

        if not tabela.empty:
            tabela_view = tabela.rename(columns={
                "id_subsistema": "Subsistema",
                "val_gerhidraulica": "Hidráulica média (MW)",
                "val_gertermica": "Térmica média (MW)",
                "val_gereolica": "Eólica média (MW)",
                "val_gersolar": "Solar média (MW)",
                "val_carga": "Carga média (MW)",
                "ibr_norm": "IBR (%)",
                "sync_norm": "Síncrona (%)",
                "val_gerhidraulica_percentual": "Hidráulica / geração (%)",
                "val_gertermica_percentual": "Térmica / geração (%)",
                "val_gereolica_percentual": "Eólica / geração (%)",
                "val_gersolar_percentual": "Solar / geração (%)",
            })
            st.dataframe(tabela_view.round(2), use_container_width=True, hide_index=True)

    elif analise_escolhida == "10. Resumo Anual de Curtailment":
        st.markdown(f"### Resumo anual de curtailment — {nome_fonte}")
        anos = sorted(df_curtailment["Data"].dropna().dt.year.unique().astype(int).tolist()) if not df_curtailment.empty else []
        if not anos:
            st.warning("Sem anos disponíveis para a fonte/escopo selecionado.")
        else:
            ano_resumo = st.selectbox("Ano do resumo:", anos, index=len(anos) - 1, key="ano_resumo_curt")
            resumo = resumo_anual_curtailment(df_curtailment, ano_resumo)
            if resumo:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Curtailment", f"{resumo['curtailment_mwh'] / 1e3:,.1f} GWh")
                c2.metric("Geração realizada", f"{resumo['geracao_mwh'] / 1e3:,.1f} GWh")
                c3.metric("Geração de referência", f"{resumo['referencia_mwh'] / 1e3:,.1f} GWh")
                c4.metric("Curtailment", f"{resumo['percentual']:.2f}%")
                st.caption(
                    f"Cobertura em {ano_resumo}: {resumo['data_min'].strftime('%d/%m/%Y')} a "
                    f"{resumo['data_max'].strftime('%d/%m/%Y')} ({resumo['meses']} mês(es) com dados)."
                )
                if resumo["meses"] < 12:
                    st.warning("Ano incompleto: os totais representam somente o período disponível.")

    elif analise_escolhida == "11. Comparação Anual e por Estado/Subsistema":
        st.markdown(f"### Comparação anual — {nome_fonte}")
        st.markdown("#### Escopo selecionado")
        tabela = gerar_tabela_comparacao_anual_curtailment(df_curtailment)
        if tabela.empty:
            st.warning("Sem dados suficientes para a comparação anual do escopo selecionado.")
        else:
            view = tabela.copy()
            view["Cobertura"] = np.where(view["Ano_Completo"], "Completo", "Parcial")
            view = view.drop(columns=["Ano_Completo"])
            st.dataframe(view.round(2), use_container_width=True, hide_index=True)
            st.caption("Deltas somente entre anos consecutivos completos; anos parciais não são comparados com anos completos.")
            st.download_button(
                "Baixar comparação do escopo em CSV",
                view.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"comparacao_anual_{nome_fonte.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            latex = gerar_latex_tabela_total(tabela)
            if latex:
                st.download_button("Baixar tabela total em LaTeX", latex, file_name="tabela_total.tex", mime="text/plain")

        st.markdown("#### Detalhamento por Estado e Subsistema")
        if all(c in df_curtailment_total.columns for c in ["id_estado", "id_subsistema"]):
            anos_det = sorted(df_curtailment_total["Data"].dropna().dt.year.unique().astype(int).tolist())
            anos_sel = st.multiselect("Anos da tabela detalhada:", anos_det, default=anos_det, key="anos_detalhe_curt")
            detalhe = gerar_tabela_estado_subsistema_anual(df_curtailment_total, anos_sel)
            if not detalhe.empty:
                detalhe_view = detalhe.copy()
                detalhe_view["Cobertura"] = np.where(detalhe_view["Ano_Completo"], "Completo", "Parcial")
                detalhe_view = detalhe_view.drop(columns=["Ano_Completo"])
                detalhe_view = detalhe_view.sort_values(["Ano", "Curtailment_%"], ascending=[True, False])
                st.dataframe(detalhe_view.round(2), use_container_width=True, hide_index=True)
                st.download_button(
                    "Baixar Estado/Subsistema em CSV",
                    detalhe_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"curtailment_estado_subsistema_{nome_fonte.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Sem dados para os anos selecionados.")
        else:
            st.warning("O Parquet de curtailment ainda não contém `id_estado` e `id_subsistema`. Regenere a base com essas dimensões.")

    elif analise_escolhida == "12. Solar: MMGD x Centralizada x Restrita":
        st.markdown("### Solar: MMGD, centralizada e centralizada + restrita")
        agregacao = st.radio("Agregação do gráfico:", ["Horária", "Diária"], horizontal=True, key="agregacao_solar_estabilidade")
        freq = "h" if agregacao == "Horária" else "D"
        if not all(c in df_usina.columns for c in ["val_geracao_solar_mmgd", "val_geracao_solar_centralizada"]):
            st.warning("Atualize `geracao_usina_horario.parquet` com `val_geracao_solar_mmgd` e `val_geracao_solar_centralizada`.")
        fig = gerar_grafico_solar_estabilidade(df_usina, df_solar, start_date, end_date, freq=freq)
        st.pyplot(fig) if fig else st.warning("Sem dados suficientes para o período selecionado.")

    elif analise_escolhida == "13. Solar: Variabilidade (CV)":
        st.markdown("### Solar: coeficiente de variação por hora")
        base_cv = preparar_series_solar_estabilidade(df_usina, df_solar)
        modo = st.radio("Visualização:", ["Período selecionado", "Comparar trimestres"], horizontal=True, key="modo_cv_solar")
        config = [
            ("mmgd_mwh", "MMGD Solar", "-"),
            ("solar_central_mwh", "Solar centralizada", "-"),
            ("solar_central_mais_restrita_mwh", "Centralizada + restrita", "--"),
        ]
        st.caption("CV solar calculado entre 06h e 18h. O período não é fixo: usa as datas selecionadas na lateral.")
        if modo == "Período selecionado":
            fig = gerar_grafico_cv_periodo(base_cv, config, start_date, end_date, 6, 18, f"CV Solar — {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        else:
            anos_cv = sorted(base_cv["Instante"].dropna().dt.year.unique().astype(int).tolist()) if not base_cv.empty else []
            if not anos_cv:
                fig = None
            else:
                ano_cv = st.selectbox("Ano:", anos_cv, index=len(anos_cv) - 1, key="ano_cv_solar")
                qs = st.multiselect("Trimestres:", [1, 2, 3, 4], default=[1, 2, 3, 4], key="trimestres_cv_solar")
                fig = gerar_grafico_cv_trimestres(base_cv, config, ano_cv, qs, 6, 18, f"Solar: CV por trimestre — {ano_cv}")
        st.pyplot(fig) if fig else st.warning("Sem dados suficientes para calcular o CV nessa seleção.")

    elif analise_escolhida == "14. Eólica: Geração x Restrita":
        st.markdown("### Eólica: geração e geração + restrita")
        agregacao = st.radio("Agregação do gráfico:", ["Horária", "Diária"], horizontal=True, key="agregacao_eolica_estabilidade")
        freq = "h" if agregacao == "Horária" else "D"
        if "val_geracao_eolica" not in df_usina.columns:
            st.warning("Atualize `geracao_usina_horario.parquet` com `val_geracao_eolica`.")
        fig = gerar_grafico_eolica_estabilidade(df_usina, df_eolico, start_date, end_date, freq=freq)
        st.pyplot(fig) if fig else st.warning("Sem dados suficientes para o período selecionado.")

    elif analise_escolhida == "15. Eólica: Variabilidade (CV)":
        st.markdown("### Eólica: coeficiente de variação por hora")
        base_cv = preparar_series_eolica_estabilidade(df_usina, df_eolico)
        modo = st.radio("Visualização:", ["Período selecionado", "Comparar trimestres"], horizontal=True, key="modo_cv_eolica")
        config = [
            ("eolica_mwh", "Geração eólica", "-"),
            ("eolica_mais_restrita_mwh", "Geração + restrita", "--"),
        ]
        st.caption("CV eólico calculado para 0–23h. O período não é fixo: usa as datas selecionadas na lateral.")
        if modo == "Período selecionado":
            fig = gerar_grafico_cv_periodo(base_cv, config, start_date, end_date, 0, 23, f"CV Eólico — {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        else:
            anos_cv = sorted(base_cv["Instante"].dropna().dt.year.unique().astype(int).tolist()) if not base_cv.empty else []
            if not anos_cv:
                fig = None
            else:
                ano_cv = st.selectbox("Ano:", anos_cv, index=len(anos_cv) - 1, key="ano_cv_eolica")
                qs = st.multiselect("Trimestres:", [1, 2, 3, 4], default=[1, 2, 3, 4], key="trimestres_cv_eolica")
                fig = gerar_grafico_cv_trimestres(base_cv, config, ano_cv, qs, 0, 23, f"Eólica: CV por trimestre — {ano_cv}")
        st.pyplot(fig) if fig else st.warning("Sem dados suficientes para calcular o CV nessa seleção.")

else:
    st.info("Por favor, selecione uma data de início e fim completa no menu lateral.")
