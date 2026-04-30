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
    base_path = r"C:\Users\bruno\OneDrive\Área de Trabalho\Harvard\TCC\Dashboard\Base_Dados"
    arquivo = f"curtailment_horario_{fonte.lower()}.parquet"
    caminho = os.path.join(base_path, arquivo)

    df = pd.read_parquet(caminho)
    df["Data"] = pd.to_datetime(df["Data"])
    return df


@st.cache_data
def carregar_balanco():
    base_path = r"C:\Users\bruno\OneDrive\Área de Trabalho\Harvard\TCC\Dashboard\Base_Dados"
    caminho = os.path.join(base_path, "balanco_submercado_horario.parquet")

    df = pd.read_parquet(caminho)
    df["Data"] = pd.to_datetime(df["Data"])
    return df


@st.cache_data
def carregar_usina():
    base_path = r"C:\Users\bruno\OneDrive\Área de Trabalho\Harvard\TCC\Dashboard\Base_Dados"
    caminho = os.path.join(base_path, "geracao_usina_horario.parquet")

    df = pd.read_parquet(caminho)
    df["Data"] = pd.to_datetime(df["Data"])
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
        "0. Painel Geral de Curtailment",
        "1. Curtailment por Dia da Semana",
        "2. Curtailment por Hora do Dia",
        "3. Geração vs Referência (Dia da Semana)",
        "4. Geração vs Referência (Hora do Dia)",
        "5. Carga e Carga Líquida por Hora",
        "6. Carga por Dia da Semana",
        "7. Fotovoltaica vs MMGD por Hora",
        "8. Participação da Geração por Fonte",
    ]
)

fonte_escolhida = st.sidebar.selectbox(
    "Fonte para análises de curtailment:",
    ["Eólico", "Solar", "Eólico + Solar"]
)

st.sidebar.divider()

st.sidebar.subheader("Filtro de Período")
min_date = date(2024, 4, 1)
max_date = date(2025, 12, 31)

intervalo_datas = st.sidebar.date_input(
    "Selecione as datas:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

ano_carga = st.sidebar.selectbox(
    "Ano para análise de carga líquida:",
    [2024, 2025],
    index=1
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


def gerar_grafico_carga_liquida_trimestres(df_balanco, ano, titulo):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 9
    })

    if df_balanco.empty:
        return None

    df_in = df_balanco.copy()
    df_in = df_in[df_in["Ano"] == int(ano)].copy()

    if df_in.empty:
        return None

    quarters = {
        "T1 (Jan-Mar)": (1, 3),
        "T2 (Abr-Jun)": (4, 6),
        "T3 (Jul-Set)": (7, 9),
        "T4 (Out-Dez)": (10, 12),
    }

    line_styles = {
        "T1 (Jan-Mar)": "-",
        "T2 (Abr-Jun)": "--",
        "T3 (Jul-Set)": ":",
        "T4 (Out-Dez)": "-.",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)

    for q_name, (m1, m2) in quarters.items():
        df_q = df_in[(df_in["Mes"] >= m1) & (df_in["Mes"] <= m2)].copy()

        carga = df_q.groupby("Hora")["val_carga"].mean().reindex(range(24), fill_value=0)
        solar = df_q.groupby("Hora")["val_gersolar"].mean().reindex(range(24), fill_value=0)
        liquida = carga - solar

        ax.plot(
            x,
            carga / 1e3,
            linestyle=line_styles[q_name],
            linewidth=2.0,
            color="#6f42c1",
            label=f"{q_name} — carga"
        )

        ax.plot(
            x,
            liquida / 1e3,
            linestyle=line_styles[q_name],
            linewidth=2.0,
            color="#17a2b8",
            label=f"{q_name} — carga - solar"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Média (GW)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        frameon=True
    )

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

    if df_filtrado.empty:
        return None

    df_grouped = df_filtrado.groupby("Hora")[
        ["val_geracao_fotovoltaica", "val_geracao_mmgd"]
    ].sum().reindex(range(24), fill_value=0).reset_index()

    df_grouped["TWh_fotovoltaica"] = df_grouped["val_geracao_fotovoltaica"] / 1e6
    df_grouped["TWh_mmgd"] = df_grouped["val_geracao_mmgd"] / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)

    ax.bar(
        x,
        df_grouped["TWh_fotovoltaica"],
        width=0.75,
        color="#6c757d",
        edgecolor="#6c757d",
        alpha=0.85,
        label="Fotovoltaica (TWh)"
    )

    ax.bar(
        x,
        df_grouped["TWh_mmgd"],
        width=0.75,
        facecolor="none",
        edgecolor="black",
        hatch="///",
        linewidth=2.0,
        label="Pequenas Usinas (MMGD) (TWh)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_ylabel("Energia (TWh)")
    ax.set_xlabel("Hora do dia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_title(titulo)

    ax.legend(
        handles=[
            Patch(facecolor="#6c757d", edgecolor="#6c757d", label="Fotovoltaica (TWh)"),
            Patch(facecolor="none", edgecolor="black", hatch="///", linewidth=2.0, label="Pequenas Usinas (MMGD) (TWh)")
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True
    )

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
# 6. Lógica de Exibição
# ==========================================
if len(intervalo_datas) == 2:
    start_date, end_date = pd.to_datetime(intervalo_datas[0]), pd.to_datetime(intervalo_datas[1])

    df_curtailment, cor_fonte, nome_fonte = escolher_base_curtailment(
        df_eolico,
        df_solar,
        fonte_escolhida
    )

    mask_c = (df_curtailment["Data"] >= start_date) & (df_curtailment["Data"] <= end_date)
    mask_b = (df_balanco["Data"] >= start_date) & (df_balanco["Data"] <= end_date)
    mask_u = (df_usina["Data"] >= start_date) & (df_usina["Data"] <= end_date)

    df_filtro_curtailment = df_curtailment.loc[mask_c].copy()
    df_filtro_balanco = df_balanco.loc[mask_b].copy()
    df_filtro_usina = df_usina.loc[mask_u].copy()

    if analise_escolhida != "5. Carga e Carga Líquida por Hora":
        st.subheader(f"Período: {start_date.strftime('%m/%Y')} a {end_date.strftime('%m/%Y')}")

    if analise_escolhida == "0. Painel Geral de Curtailment":
        st.markdown(f"### Painel geral de curtailment — {nome_fonte}")

        fig1 = gerar_grafico_curtailment_mensal_percentual(
            df_filtro_curtailment,
            cor_fonte,
            f"Curtailment mensal percentual - {nome_fonte}"
        )

        if fig1:
            st.pyplot(fig1)
        else:
            st.warning("Sem dados para o gráfico mensal percentual.")

        fig2 = gerar_grafico_curtailment_mensal_por_tipo(
            df_filtro_curtailment,
            f"Curtailment mensal por tipo de restrição - {nome_fonte}"
        )

        if fig2:
            st.pyplot(fig2)
        else:
            st.warning("Sem dados para o gráfico mensal por tipo.")

        fig3 = gerar_grafico_dia_semana(
            df_filtro_curtailment,
            cor_fonte,
            f"Curtailment por dia da semana - {nome_fonte}"
        )

        if fig3:
            st.pyplot(fig3)
        else:
            st.warning("Sem dados para o gráfico por dia da semana.")

        fig4 = gerar_grafico_hora_dia(
            df_filtro_curtailment,
            cor_fonte,
            f"Curtailment por hora do dia - {nome_fonte}"
        )

        if fig4:
            st.pyplot(fig4)
        else:
            st.warning("Sem dados para o gráfico por hora.")

    elif analise_escolhida == "1. Curtailment por Dia da Semana":
        st.markdown(f"### Curtailment por dia da semana — {nome_fonte}")

        fig = gerar_grafico_dia_semana(
            df_filtro_curtailment,
            cor_fonte,
            f"Perfil Semanal - {nome_fonte}"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "2. Curtailment por Hora do Dia":
        st.markdown(f"### Curtailment por hora do dia — {nome_fonte}")

        fig = gerar_grafico_hora_dia(
            df_filtro_curtailment,
            cor_fonte,
            f"Perfil Horário - {nome_fonte}"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "3. Geração vs Referência (Dia da Semana)":
        st.markdown(f"### Geração vs referência por dia da semana — {nome_fonte}")

        fig = gerar_grafico_geracao_vs_ref_dia(
            df_filtro_curtailment,
            cor_fonte,
            f"Geração vs Referência - {nome_fonte}"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "4. Geração vs Referência (Hora do Dia)":
        st.markdown(f"### Geração verificada + cortes por hora — {nome_fonte}")

        fig = gerar_grafico_geracao_vs_ref_hora(
            df_filtro_curtailment,
            cor_fonte,
            f"Geração Verificada + Cortes por Hora - {nome_fonte}"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "5. Carga e Carga Líquida por Hora":
        st.subheader(f"Ano analisado: {ano_carga}")

        fig = gerar_grafico_carga_liquida_trimestres(
            df_balanco,
            ano_carga,
            f"Carga média e carga líquida por hora - {ano_carga}"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o ano selecionado.")

    elif analise_escolhida == "6. Carga por Dia da Semana":
        st.markdown("### Carga total por dia da semana")

        fig = gerar_grafico_carga_dia_semana(
            df_filtro_balanco,
            "Carga por Dia da Semana"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "7. Fotovoltaica vs MMGD por Hora":
        st.markdown("### Geração fotovoltaica e MMGD por hora")

        fig = gerar_grafico_fotovoltaica_vs_mmgd_hora(
            df_filtro_usina,
            "Geração Fotovoltaica vs MMGD por Hora"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados para o período selecionado.")

    elif analise_escolhida == "8. Participação da Geração por Fonte":
        st.markdown("### Participação percentual da geração por fonte")

        fig = gerar_grafico_participacao_geracao(
            df_balanco,
            "Participação da geração por fonte"
        )

        if fig:
            st.pyplot(fig)
        else:
            st.warning("Sem dados disponíveis.")

else:
    st.info("Por favor, selecione uma data de início e fim completa no menu lateral.")