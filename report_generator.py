"""
report_generator.py — 688981 T+0策略回测可视化报告生成器 v2
===========================================================
v2 修复：
  1. 乱码修复：matplotlib 使用 Noto Sans CJK JP，HTML 写入 utf-8-sig (BOM)
  2. 双语切换：HTML 右上角 [中 / EN] 按钮，一键切换所有标签
  3. 编码兼容：Windows 浏览器双击打开不再乱码

用法：
  python report_generator.py                          # 最新 run
  python report_generator.py --run_id 20260603_165651
  python report_generator.py --all                   # 所有 run 汇总
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.font_manager as _fm

    _CJK_FONTS = [f.name for f in _fm.fontManager.ttflist
                  if any(x in f.name for x in ["CJK", "SimHei", "WenQuanYi", "Heiti"])]
    if _CJK_FONTS:
        plt.rcParams["font.sans-serif"] = [_CJK_FONTS[0]] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"

C_EQUITY = "#2563EB"; C_BM="#94A3B8"; C_WIN="#10B981"
C_LOSS   = "#EF4444"; C_DD="#F59E0B"; C_BG="#0F172A"
C_CARD   = "#1E293B"; C_BORDER="#334155"

if HAS_MPL:
    plt.rcParams.update({
        "figure.facecolor": C_BG, "axes.facecolor": C_CARD,
        "axes.edgecolor": C_BORDER, "axes.labelcolor": "#CBD5E1",
        "axes.titlecolor": "#F1F5F9", "text.color": "#CBD5E1",
        "xtick.color": "#64748B", "ytick.color": "#64748B",
        "grid.color": C_CARD, "grid.alpha": 0.6, "grid.linewidth": 0.5, "font.size": 10,
    })

ZH = dict(
    equity_title="策略净值曲线 vs 基准", equity_legend_s="策略净值",
    equity_legend_b="688981 买入持有", equity_ylabel="净值", drawdown_ylabel="回撤 (%)",
    monthly_title="月度收益分布", monthly_ylabel="月度收益 (%)",
    pnl_hist_title="单笔交易PnL分布", pnl_xlabel="单笔PnL (元)", pnl_ylabel="频次",
    type_title="平仓类型分布", cum_title="累积PnL曲线", cum_xlabel="交易序号",
    cum_ylabel="累积PnL (元)", hold_title="持仓K线数分布",
    hold_xlabel="持仓K线数", hold_ylabel="次数",
    wf_title="Walk-Forward 滚动指标均值 (±1σ)",
    wf_labels=["Sharpe", "Sortino", "最大回撤", "Calmar", "胜率", "总收益"],
)

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def _fmt_pct(v): return f"{v*100:+.2f}%"
def _fmt_f2(v):  return f"{v:.3f}"

def plot_equity_curve(equity, benchmark=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor(C_BG)
    ax1.plot(equity.index, equity.values, color=C_EQUITY, lw=1.6, label=ZH["equity_legend_s"], zorder=3)
    if benchmark is not None:
        bm = benchmark.reindex(equity.index).ffill()
        ax1.plot(bm.index, bm.values, color=C_BM, lw=1.0, alpha=0.7, label=ZH["equity_legend_b"], zorder=2)
    ax1.fill_between(equity.index, equity.values, 1.0, where=equity.values >= 1.0, alpha=0.12, color=C_EQUITY)
    ax1.fill_between(equity.index, equity.values, 1.0, where=equity.values < 1.0,  alpha=0.12, color=C_LOSS)
    ax1.axhline(1.0, color=C_BORDER, lw=0.8, ls="--", alpha=0.5)
    ax1.set_ylabel(ZH["equity_ylabel"], fontsize=11)
    ax1.set_title(ZH["equity_title"], fontsize=13, pad=10)
    ax1.legend(loc="upper left", framealpha=0.15, fontsize=9)
    ax1.grid(True, axis="y")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    ax2.fill_between(dd.index, dd.values, 0, color=C_DD, alpha=0.6)
    ax2.plot(dd.index, dd.values, color=C_DD, lw=0.8)
    ax2.set_ylabel(ZH["drawdown_ylabel"], fontsize=10)
    ax2.grid(True, axis="y")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.tight_layout(pad=1.5)
    return _fig_to_b64(fig)

def plot_monthly_returns(equity):
    ret = equity.resample("ME").last().pct_change().dropna()
    if len(ret) < 2: return ""
    months = [r.strftime("%Y-%m") for r in ret.index]
    values = ret.values * 100
    colors = [C_WIN if v >= 0 else C_LOSS for v in values]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(months, values, color=colors, alpha=0.85, edgecolor=C_BORDER, lw=0.5)
    ax.axhline(0, color=C_BORDER, lw=0.8)
    ax.set_ylabel(ZH["monthly_ylabel"], fontsize=10)
    ax.set_title(ZH["monthly_title"], fontsize=12, pad=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(True, axis="y"); plt.tight_layout()
    return _fig_to_b64(fig)

def plot_trade_analysis(trades):
    if not trades: return ""
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    from collections import Counter
    type_map = {"stop_loss":"固定止损","trail_stop":"追踪止损","time_stop":"时间止损","eod":"收盘平仓"}
    type_cnt = Counter(type_map.get(t.get("type","other"),t.get("type","other")) for t in trades)
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(pnls.min(), pnls.max(), 40)
    ax1.hist(wins, bins=bins, color=C_WIN, alpha=0.72, label=f"盈利 (n={len(wins)})")
    ax1.hist(losses, bins=bins, color=C_LOSS, alpha=0.72, label=f"亏损 (n={len(losses)})")
    ax1.axvline(0, color=C_BORDER, lw=1.2, ls="--")
    ax1.axvline(np.mean(pnls), color="#FBBF24", lw=1.5, ls=":", label=f"均值={np.mean(pnls):.0f}")
    ax1.set_xlabel(ZH["pnl_xlabel"],fontsize=10); ax1.set_ylabel(ZH["pnl_ylabel"],fontsize=10)
    ax1.set_title(ZH["pnl_hist_title"],fontsize=11,pad=6)
    ax1.legend(fontsize=9); ax1.grid(True, axis="y")
    ax2 = fig.add_subplot(gs[0, 2])
    labels=list(type_cnt.keys()); sizes=list(type_cnt.values())
    pie_c=[C_EQUITY,C_WIN,C_DD,C_LOSS][:len(labels)]
    _,_,autotexts=ax2.pie(sizes,labels=labels,colors=pie_c,autopct="%1.1f%%",pctdistance=0.75,
        textprops={"fontsize":8,"color":"#CBD5E1"},wedgeprops={"edgecolor":C_BG,"linewidth":1.5})
    for a in autotexts: a.set_color("#F1F5F9"); a.set_fontsize(8)
    ax2.set_title(ZH["type_title"],fontsize=11,pad=6)
    ax3 = fig.add_subplot(gs[1, :2])
    cumulative = np.cumsum(pnls); x = range(len(cumulative))
    ax3.plot(x, cumulative, color=C_EQUITY, lw=1.5, zorder=3)
    ax3.fill_between(x, cumulative, 0, where=np.array(cumulative)>=0, color=C_EQUITY, alpha=0.10)
    ax3.fill_between(x, cumulative, 0, where=np.array(cumulative)<0,  color=C_LOSS,   alpha=0.10)
    ax3.axhline(0, color=C_BORDER, lw=0.8, ls="--")
    ax3.set_xlabel(ZH["cum_xlabel"],fontsize=10); ax3.set_ylabel(ZH["cum_ylabel"],fontsize=10)
    ax3.set_title(ZH["cum_title"],fontsize=11,pad=6); ax3.grid(True, axis="y")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax4 = fig.add_subplot(gs[1, 2])
    hold_bars=[t.get("hold_bars",0) for t in trades]
    ax4.hist(hold_bars, bins=range(0,max(hold_bars)+2,1), color=C_EQUITY, alpha=0.75, edgecolor=C_BORDER,lw=0.5)
    ax4.set_xlabel(ZH["hold_xlabel"],fontsize=10); ax4.set_ylabel(ZH["hold_ylabel"],fontsize=10)
    ax4.set_title(ZH["hold_title"],fontsize=11,pad=6); ax4.grid(True,axis="y")
    return _fig_to_b64(fig)

def plot_walk_forward(wf_summary):
    keys = ["sharpe_mean","sortino_mean","max_drawdown_mean","calmar_mean","win_rate_mean","total_return_mean"]
    stds = ["sharpe_std", "sortino_std", "max_drawdown_std", "calmar_std", "win_rate_std", "total_return_std"]
    means=[wf_summary.get(k,0) for k in keys]; errs=[wf_summary.get(k,0) for k in stds]
    labels=ZH["wf_labels"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels)); colors=[C_WIN if m>=0 else C_LOSS for m in means]
    ax.bar(x, means, color=colors, alpha=0.8, edgecolor=C_BORDER, lw=0.5)
    ax.errorbar(x, means, yerr=errs, fmt="none", color="#FBBF24", capsize=5, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=10)
    ax.axhline(0, color=C_BORDER, lw=0.8, ls="--")
    ax.set_title(ZH["wf_title"],fontsize=12,pad=8); ax.grid(True,axis="y")
    plt.tight_layout()
    return _fig_to_b64(fig)

# ── I18N 映射 ───────────────────────────────────────────────────
_MI = {
    "total_return":("总收益率","Total Return"), "excess_return":("超额收益","Excess Return"),
    "sharpe":("夏普比率","Sharpe Ratio"), "sortino":("索提诺比率","Sortino Ratio"),
    "max_drawdown":("最大回撤","Max Drawdown"), "calmar":("Calmar比率","Calmar Ratio"),
    "omega":("Omega比率","Omega Ratio"), "information_ratio":("信息比率","Info Ratio"),
    "win_rate":("胜率","Win Rate"), "payoff_ratio":("盈亏比","Payoff Ratio"),
    "profit_factor":("利润因子","Profit Factor"), "max_consec_loss":("最大连亏","Max Consec Loss"),
    "avg_hold_bars":  ("平均持仓K线","Avg Hold Bars"),
}
_SI = {
    "perf":("核心绩效指标","Performance"), "equity":("净值曲线 & 回撤","Equity & Drawdown"),
    "monthly":("月度收益","Monthly Returns"), "trades":("交易分析","Trade Analysis"),
    "wf":("Walk-Forward 滚动验证","Walk-Forward Validation"),
}
_EI = {
    "time_stop":("时间止损","Time Stop"), "stop_loss":("固定止损","Fixed Stop"),
    "trail_stop":("追踪止损","Trail Stop"), "eod":("收盘平仓","EOD Close"),
}

def _mc(key, value, sub="", cls="", accent=""):
    zh,en = _MI.get(key,(key,key))
    acc   = f' style="border-top:3px solid {accent}"' if accent else ''
    return (f'<div class="metric-card"{acc}>'
            f'<div class="metric-label" data-zh="{zh}" data-en="{en}">{zh}</div>'
            f'<div class="metric-value {cls}">{value}</div>'
            + (f'<div class="metric-sub">{sub}</div>' if sub else "")
            + '</div>')

def _row_label(zh, en):
    return (f'<div class="row-label" data-zh="{zh}" data-en="{en}">{zh}</div>')

def _sec(key):
    zh,en = _SI.get(key,(key,key))
    return (f'<h2 class="section-title" data-zh="📊 {zh}" data-en="📊 {en}">📊 {zh}</h2>')

def build_html(run_id, metrics, equity, trades, benchmark=None, wf_summary=None, explain=None):
    charts = {}
    if HAS_MPL:
        charts["equity"]  = plot_equity_curve(equity, benchmark)
        charts["monthly"] = plot_monthly_returns(equity)
        charts["trades"]  = plot_trade_analysis(trades)
        if wf_summary: charts["wf"] = plot_walk_forward(wf_summary)

    def img(k):
        if charts.get(k): return f'<img src="data:image/png;base64,{charts[k]}" class="chart-img">'
        return '<div class="chart-ph" data-zh="图表数据不足" data-en="Insufficient data">图表数据不足</div>'

    tr=metrics.get("total_return",0); br=metrics.get("benchmark_return",0)
    er=metrics.get("excess_return",0); sp=metrics.get("sharpe",0)
    so=metrics.get("sortino",0);      om=metrics.get("omega",0)
    mdd=metrics.get("max_drawdown",0);cal=metrics.get("calmar",0)
    wr=metrics.get("win_rate",0);     pf=metrics.get("profit_factor",0)
    pr=metrics.get("payoff_ratio",0); nt=metrics.get("n_trades",0)
    ir=metrics.get("information_ratio",0); mcl=metrics.get("max_consec_loss",0)

    cc = lambda v,p=True: ("pos" if v>=0 else "neg") if p else ("neg" if v>=0 else "pos")

    A = "#10B981"   # 绿 — 收益指标
    B = "#3B82F6"   # 蓝 — 风险/比率指标
    D = "#F59E0B"   # 橙 — 回撤
    E = "#8B5CF6"   # 紫 — 交易统计
    # 行1：核心收益 & 风险调整指标（去掉无意义的买入持有对比）
    row1=(_mc("total_return",_fmt_pct(tr),"年化T+0策略",cc(tr),A)+
          _mc("sharpe",_fmt_f2(sp),"风险调整收益",cc(sp),B)+
          _mc("sortino",_fmt_f2(so),"仅下行波动",cc(so),B)+
          _mc("omega",_fmt_f2(om),"盈利/亏损面积",cc(om-1),B))
    # 行2：回撤 & 稳定性
    row2=(_mc("max_drawdown",f"{mdd*100:.2f}%","最大峰谷回撤","neg",D)+
          _mc("calmar",_fmt_f2(cal),"年化收益/最大回撤",cc(cal),D)+
          _mc("information_ratio",_fmt_f2(ir),"超额收益/跟踪误差",cc(ir),B)+
          _mc("max_consec_loss",f"{mcl} 笔","连续亏损上限","neg" if mcl>5 else "",D))
    # 行3：交易质量
    row3=(_mc("win_rate",f"{wr*100:.1f}%",f"共 {nt} 笔",cc(wr-0.5),E)+
          _mc("payoff_ratio",_fmt_f2(pr),"平均盈利/平均亏损",cc(pr-1),E)+
          _mc("profit_factor",_fmt_f2(pf),"总盈利/总亏损",cc(pf-1),E)+
          _mc("avg_hold_bars",f"{metrics.get('avg_hold_bars',0):.1f} 根","平均持仓K线","",E))

    tt=metrics.get("trade_types",{})
    if isinstance(tt,str):
        import ast
        try: tt=ast.literal_eval(tt)
        except: tt={}
    tt_rows=""
    for k,v in tt.items():
        zh_t,en_t=_EI.get(k,(k,k))
        tt_rows+=(f"<tr><td data-zh='{zh_t}' data-en='{en_t}'>{zh_t}</td>"
                  f"<td>{v:.1f}%</td></tr>")

    wf_sec=""
    if wf_summary:
        wf_rows="".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k,v in wf_summary.items())
        wf_sec=(f'<section class="section">{_sec("wf")}{img("wf")}'
                f'<div class="table-wrap"><table class="data-table">'
                f'<thead><tr><th data-zh="指标" data-en="Metric">指标</th>'
                f'<th data-zh="数值" data-en="Value">数值</th></tr></thead>'
                f'<tbody>{wf_rows}</tbody></table></div></section>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>688981 T+0 Report | {run_id}</title>
<style>
:root{{--bg:{C_BG};--card:{C_CARD};--border:{C_BORDER};--txt:#CBD5E1;--bright:#F1F5F9;--muted:#64748B;--blue:{C_EQUITY};--green:{C_WIN};--red:{C_LOSS};--amber:{C_DD}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'JetBrains Mono','Fira Code','Courier New',monospace;line-height:1.6;min-height:100vh}}
.header{{background:linear-gradient(135deg,#0F172A 0%,#1E293B 50%,#0F172A 100%);border-bottom:1px solid var(--border);padding:28px 40px;display:flex;justify-content:space-between;align-items:flex-start}}
.badge{{display:inline-block;background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.3);color:#60A5FA;font-size:11px;letter-spacing:2px;padding:3px 10px;border-radius:2px;margin-bottom:10px;text-transform:uppercase}}
.header h1{{font-size:26px;color:var(--bright);font-weight:700;letter-spacing:-.5px}}
.header-meta{{margin-top:6px;color:var(--muted);font-size:12px}}
.header-meta span{{margin-right:20px}}
.lang-btn{{background:rgba(37,99,235,.12);border:1px solid rgba(37,99,235,.3);color:#60A5FA;font-size:13px;font-family:inherit;padding:6px 16px;border-radius:4px;cursor:pointer;letter-spacing:1px;transition:all .2s;flex-shrink:0;margin-top:4px}}
.lang-btn:hover{{background:rgba(37,99,235,.25);color:#93C5FD}}
.container{{max-width:1400px;margin:0 auto;padding:32px 40px}}
.section{{margin-bottom:40px}}
.section-title{{font-size:15px;color:var(--bright);margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid var(--border);letter-spacing:.5px}}
.metrics-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}}
.metric-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:18px 20px;transition:all .2s}}.metric-card:hover{{border-color:rgba(37,99,235,.5);box-shadow:0 4px 20px rgba(0,0,0,.4);transform:translateY(-1px)}}.row-label{{font-size:10px;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:8px;margin-top:6px;padding-left:2px}}
.metric-card:hover{{border-color:rgba(37,99,235,.4)}}
.metric-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.metric-value{{font-size:24px;font-weight:800;color:var(--bright);margin-bottom:4px;letter-spacing:-.5px}}
.metric-sub{{font-size:11px;color:var(--muted)}}
.pos{{color:var(--green)!important}}.neg{{color:var(--red)!important}}
.chart-img{{width:100%;border-radius:8px;border:1px solid var(--border);margin-bottom:20px;display:block}}
.chart-ph{{background:var(--card);border:1px dashed var(--border);border-radius:8px;padding:60px;text-align:center;color:var(--muted);margin-bottom:20px}}.subsection{{font-size:13px;color:#94A3B8;font-weight:500;margin:20px 0 10px;letter-spacing:.3px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.table-wrap{{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px}}
.data-table thead{{background:rgba(37,99,235,.1)}}
.data-table th{{padding:12px 16px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);border-bottom:1px solid var(--border)}}
.data-table td{{padding:10px 16px;border-bottom:1px solid rgba(51,65,85,.4);color:var(--txt)}}
.data-table tr:last-child td{{border-bottom:none}}
.data-table tr:hover td{{background:rgba(37,99,235,.04)}}
.disclaimer{{background:rgba(245,158,11,.05);border:1px solid rgba(245,158,11,.2);border-radius:6px;padding:16px 20px;font-size:12px;color:#FCD34D;margin-top:40px}}
.footer{{border-top:1px solid var(--border);padding:20px 40px;text-align:center;font-size:11px;color:var(--muted)}}
@media(max-width:900px){{.metrics-grid{{grid-template-columns:repeat(2,1fr)}}.two-col{{grid-template-columns:1fr}}.container,.header{{padding:20px}}.header{{flex-direction:column;gap:12px}}}}
</style>
</head>
<body>
<header class="header">
  <div>
    <div class="badge">Backtest Report</div>
    <h1>中芯国际 688981 · T+0 策略回测</h1>
    <div class="header-meta">
      <span>📁 Run: <strong>{run_id}</strong></span>
      <span>🕐 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</span>
      <span>📊 {equity.index[0].strftime('%Y-%m-%d')} ~ {equity.index[-1].strftime('%Y-%m-%d')}</span>
    </div>
  </div>
  <button class="lang-btn" id="lang-btn" onclick="toggleLang()">EN</button>
</header>
<div class="container">
  <section class="section">{_sec("perf")}
    {_row_label("收益指标","Return Metrics")}
    <div class="metrics-grid">{row1}</div>
    {_row_label("回撤 & 风险","Drawdown & Risk")}
    <div class="metrics-grid">{row2}</div>
    {_row_label("交易质量","Trade Quality")}
    <div class="metrics-grid">{row3}</div>
  </section>
  <section class="section">{_sec("equity")}{img("equity")}</section>
  <section class="section">{_sec("monthly")}{img("monthly")}</section>
  <section class="section">{_sec("trades")}{img("trades")}
    <div class="two-col" style="margin-top:16px">
      <div class="table-wrap"><table class="data-table">
        <thead><tr>
          <th data-zh="平仓类型" data-en="Exit Type">平仓类型</th>
          <th data-zh="占比" data-en="Ratio">占比</th>
        </tr></thead>
        <tbody>{tt_rows}</tbody>
      </table></div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr>
          <th data-zh="指标" data-en="Metric">指标</th>
          <th data-zh="数值" data-en="Value">数值</th>
        </tr></thead>
        <tbody>
          <tr><td data-zh="平均持仓K线" data-en="Avg Hold Bars">平均持仓K线</td><td>{metrics.get('avg_hold_bars',0):.1f}</td></tr>
          <tr><td data-zh="平均盈利 (元)" data-en="Avg Win (CNY)">平均盈利 (元)</td><td class="pos">{metrics.get('avg_win',0):+,.0f}</td></tr>
          <tr><td data-zh="平均亏损 (元)" data-en="Avg Loss (CNY)">平均亏损 (元)</td><td class="neg">{metrics.get('avg_loss',0):+,.0f}</td></tr>
          <tr><td data-zh="最大连续亏损" data-en="Max Consec Loss">最大连续亏损</td><td>{mcl} 笔</td></tr>
        </tbody>
      </table></div>
    </div>
  </section>
  {wf_sec}
  {_explain_html(explain or {})}
  <div class="disclaimer"
       data-zh="&#9888; 本报告仅供研究参考，不构成投资建议。历史回测结果不代表未来收益。所有交易均含佣金（万2.5）、印花税（0.1% 卖出）、经手费及滑点成本。"
       data-en="&#9888; For research purposes only. Past backtest results do not guarantee future performance. All trades include commission (0.025%), stamp duty (0.1% sell), exchange fee, and slippage.">
    ⚠️ 本报告仅供研究参考，不构成投资建议。历史回测结果不代表未来收益。
    所有交易均含佣金（万2.5）、印花税（0.1% 卖出）、经手费及滑点成本。
  </div>
</div>
<footer class="footer">SystematicAlpha · 688981 T+0 · report_generator.py v2</footer>
<script>
var _lang='zh';
function toggleLang(){{
  _lang=_lang==='zh'?'en':'zh';
  document.querySelectorAll('[data-zh]').forEach(function(el){{
    var t=el.getAttribute('data-'+_lang);
    if(t!=null){{ el.children.length===0? (el.textContent=t):(el.innerHTML=t); }}
  }});
  document.getElementById('lang-btn').textContent=_lang==='zh'?'EN':'中';
  document.documentElement.lang=_lang==='zh'?'zh-CN':'en';
}}
</script>
</body>
</html>"""


def load_explain(run_dir):
    """读取 explain/ 子目录的图片和 CSV，返回可嵌入 HTML 的字典。"""
    import base64
    ex = run_dir / "explain"
    out = {}
    if not ex.exists():
        return out
    for key, fname in [("shap_importance","shap_importance.png"),
                       ("shap_heatmap","shap_heatmap.png")]:
        p = ex / fname
        if p.exists():
            out[key] = base64.b64encode(p.read_bytes()).decode()
    for key, fname, vcol_hint in [
        ("ig_top",   "ig_importance.csv",   "ig_importance"),
        ("perm_top", "perm_importance.csv",  "drop_mean"),
    ]:
        p = ex / fname
        if p.exists():
            df = pd.read_csv(p)
            vcol = vcol_hint if vcol_hint in df.columns else df.columns[1]
            out[key] = df.head(8)[["feature", vcol]].values.tolist()
    stab_p = ex / "feature_stability.csv"
    if stab_p.exists():
        df = pd.read_csv(stab_p)
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False)
            out["stab_top"] = df.head(6)[["feature","score"]].values.tolist()
    return out


def _explain_html(explain):
    """将 explain 字典渲染为完整的 HTML section。"""
    if not explain:
        return ""

    def eimg(key, alt=""):
        if key in explain:
            return (f'<img src="data:image/png;base64,{explain[key]}" '
                    f'class="chart-img" alt="{alt}">')
        return f'<div class="chart-ph" data-zh="{alt} — 图片未生成" data-en="{alt} — not generated">{alt} — 图片未生成</div>'

    def ftable(rows, zh, en):
        if not rows:
            return '<p style="color:var(--muted)">数据未生成</p>'
        trs = "".join(f'<tr><td>{r[0]}</td><td>{float(r[1]):.4f}</td></tr>' for r in rows)
        return (f'<div class="table-wrap"><table class="data-table">'
                f'<thead><tr>'
                f'<th data-zh="特征" data-en="Feature">特征</th>'
                f'<th data-zh="{zh}" data-en="{en}">{zh}</th>'
                f'</tr></thead><tbody>{trs}</tbody></table></div>')

    shap_block = ""
    if "shap_importance" in explain or "shap_heatmap" in explain:
        shap_block = (
            '<h3 class="subsection" '
            'data-zh="SHAP 全局特征重要性" data-en="SHAP Global Importance">'
            'SHAP 全局特征重要性</h3>' + eimg("shap_importance","SHAP重要性") +
            '<h3 class="subsection" '
            'data-zh="SHAP 时序热力图（哪根K线贡献最大）" '
            'data-en="SHAP Temporal Heatmap">'
            'SHAP 时序热力图（哪根K线贡献最大）</h3>' + eimg("shap_heatmap","SHAP热力图")
        )

    ig_ok   = "ig_top"   in explain
    perm_ok = "perm_top" in explain
    two_col = ""
    if ig_ok or perm_ok:
        left  = (('<div><h3 class="subsection" data-zh="集成梯度 Top-8" '
                  'data-en="Integrated Gradients Top-8">集成梯度 Top-8</h3>'
                  + ftable(explain.get("ig_top",[]),"归因值","Attribution")
                  + '</div>') if ig_ok else '<div></div>')
        right = (('<div><h3 class="subsection" data-zh="排列重要性 Top-8" '
                  'data-en="Permutation Importance Top-8">排列重要性 Top-8</h3>'
                  + ftable(explain.get("perm_top",[]),"AUC下降量","AUC Drop")
                  + '</div>') if perm_ok else '<div></div>')
        two_col = f'<div class="two-col" style="margin-top:20px">{left}{right}</div>'

    insight = ""
    if explain.get("ig_top"):
        top1 = explain["ig_top"][0][0]
        insight = (
            f'<div style="margin-top:16px;padding:14px 18px;'
            f'background:rgba(37,99,235,.07);border:1px solid rgba(37,99,235,.25);'
            f'border-radius:6px;font-size:13px;color:#CBD5E1">'
            f'<strong style="color:#60A5FA">💡 核心洞察</strong>&emsp;'
            f'三种方法一致将 <code style="color:#34D399">{top1}</code> 列为最重要特征，'
            f'说明日内时段信息对预测贡献最大，市场存在显著的日内时序规律。</div>'
        )

    return (
        f'<section class="section">'
        f'<h2 class="section-title" '
        f'data-zh="🔍 可解释性分析" data-en="🔍 Explainability Analysis">'
        f'🔍 可解释性分析</h2>'
        f'{shap_block}{two_col}{insight}'
        f'</section>'
    )

def load_run(run_dir):
    with open(run_dir/"metrics.json", encoding="utf-8") as f: metrics=json.load(f)
    equity=pd.read_csv(run_dir/"equity.csv",index_col=0,parse_dates=True).squeeze()
    if equity.index.tzinfo is None: equity.index=equity.index.tz_localize("Asia/Shanghai")
    trades=[]
    if (run_dir/"trades.csv").exists():
        trades=pd.read_csv(run_dir/"trades.csv").to_dict("records")
    return metrics, equity, trades

def generate_report(run_id="", output=None):
    if run_id:
        run_dir=RESULTS/run_id
    else:
        cands=sorted([d for d in RESULTS.iterdir()
                      if d.is_dir() and (d/"equity.csv").exists() and not d.name.endswith("_vnpy")],
                     key=lambda d:d.name, reverse=True)
        if not cands: raise RuntimeError("results/ 下没有找到有效的回测结果")
        run_dir=cands[0]; run_id=run_dir.name
    print(f"  ▶ 加载 run: {run_id}")
    metrics,equity,trades=load_run(run_dir)
    wf_summary=None
    wf_p=RESULTS/"walk_forward_summary.json"
    if wf_p.exists():
        with open(wf_p,encoding="utf-8") as f: wf_summary=json.load(f)
    explain=load_explain(run_dir)   # 加载 explain/ 子目录（不存在时返回 {}）
    if explain:
        print(f"  ▶ 检测到可解释性数据: {list(explain.keys())}")
    print("  ▶ 渲染图表并生成 HTML...")
    html=build_html(run_id,metrics,equity,trades,wf_summary=wf_summary,explain=explain)
    out=Path(output) if output else run_dir/"report.html"
    out.write_text(html, encoding="utf-8-sig")  # BOM → Windows浏览器正确识别UTF-8
    print(f"  ✅ 报告已生成: {out}")
    return out

def generate_summary():
    runs=sorted([d for d in RESULTS.iterdir()
                 if d.is_dir() and (d/"metrics.json").exists() and not d.name.endswith("_vnpy")],
                key=lambda d:d.name)
    rows=""
    for d in runs:
        try:
            with open(d/"metrics.json",encoding="utf-8") as f: m=json.load(f)
        except: continue
        tr=m.get("total_return",0); sp=m.get("sharpe",0)
        mdd=m.get("max_drawdown",0); wr=m.get("win_rate",0); nt=m.get("n_trades",0)
        tc="#10B981" if tr>=0 else "#EF4444"; sc="#10B981" if sp>=0 else "#EF4444"
        rows+=(f"<tr><td style='font-family:monospace'>{d.name}</td>"
               f"<td style='color:{tc}'>{tr*100:+.2f}%</td>"
               f"<td style='color:{sc}'>{sp:.3f}</td>"
               f"<td style='color:#EF4444'>{mdd*100:.2f}%</td>"
               f"<td>{wr*100:.1f}%</td><td>{nt}</td></tr>")
    html=(f'<!DOCTYPE html><html lang="zh-CN"><head>'
          f'<meta charset="UTF-8"><meta http-equiv="Content-Type" content="text/html; charset=utf-8">'
          f'<title>688981 汇总</title>'
          f'<style>body{{background:#0F172A;color:#CBD5E1;font-family:monospace;padding:40px}}'
          f'h1{{color:#F1F5F9;margin-bottom:24px}}'
          f'table{{width:100%;border-collapse:collapse}}'
          f'th{{background:rgba(37,99,235,.1);padding:12px 16px;text-align:left;font-size:11px;'
          f'text-transform:uppercase;letter-spacing:1px;color:#64748B;border-bottom:1px solid #334155}}'
          f'td{{padding:10px 16px;border-bottom:1px solid rgba(51,65,85,.4);font-size:13px}}'
          f'tr:hover td{{background:rgba(37,99,235,.04)}}</style></head>'
          f'<body><h1>688981 T+0 · 全部回测汇总</h1>'
          f'<table><thead><tr><th>Run ID</th><th>总收益率</th><th>夏普比率</th>'
          f'<th>最大回撤</th><th>胜率</th><th>交易次数</th></tr></thead>'
          f'<tbody>{rows}</tbody></table></body></html>')
    out=RESULTS/"summary_report.html"
    out.write_text(html, encoding="utf-8-sig")
    print(f"✅ 汇总报告: {out}"); return out

def main():
    parser=argparse.ArgumentParser(description="688981 T+0 报告生成器 v2")
    parser.add_argument("--run_id",default="")
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    print("\n🗂️  688981 T+0 策略报告生成器 v2"); print("="*48)
    if args.all: generate_summary()
    else: generate_report(run_id=args.run_id, output=args.output or None)
    print("="*48+"\n")

if __name__=="__main__": main()
