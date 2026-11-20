"""实验统计服务：版本化观测、双比例 z 检验与带实际门槛的统计摘要。

业务规则：
- 显著性水平取自实验管理里的实验配置，统计入口不再固定按 1.96 判断；
- 水平缺失、无法解析或不在 (0, 1) 区间时直接报错，绝不默默按 5% 发结论；
- 对照组与实验组的样本量、转化人数必须来自同一次有效观测（同一观测版本快照）；
- 日报生成期间的补报会登记为新的观测版本，旧版本数据保持原样；
- 统计摘要按（实验编号, 观测版本）发布后冻结，之后更改门槛不会重新解释旧版本；
- 数据集市报表详情读取发布时的样本与显著性水平，历史查询可重现当时的显著结论。
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from typing import Any

from app.fixtures import EXPERIMENT_RESULTS
from app.store import store

OBSERVATION_TABLE = "_experiment_observations"
SUMMARY_TABLE = "_experiment_stat_summaries"
GROUPS = ("control", "treatment")


# ---------------------------------------------------------------------------
# 观测版本：一次有效观测必须同时携带完整的对照组与实验组
# ---------------------------------------------------------------------------

def _observations() -> list[dict[str, Any]]:
    """读取观测版本表，首次访问时把 fixtures 里的实验结果登记为第 1 版观测。"""
    table = store.rows(OBSERVATION_TABLE)
    if not table:
        grouped: dict[str, list[dict]] = {}
        for row in EXPERIMENT_RESULTS:
            grouped.setdefault(str(row.get("实验编号")), []).append(row)
        for code, rows in grouped.items():
            table.append({
                "实验编号": code,
                "观测版本": 1,
                "来源": "初始观测",
                "记录时间": datetime.now().isoformat(),
                "groups": [
                    {"分组": r.get("分组"), "样本量": r.get("样本量"), "转化人数": r.get("转化人数")}
                    for r in rows
                ],
            })
    return table


def list_observations(experiment_code: str) -> list[dict[str, Any]]:
    """列出实验的全部观测版本，经营方据此区分报表采用的是哪一版。"""
    return [dict(obs) for obs in _observations() if obs.get("实验编号") == experiment_code]


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _validate_groups(groups: Any) -> tuple[dict[str, dict[str, int]] | None, str]:
    """校验一次观测：必须同时携带完整的对照组与实验组，且数值自洽。"""
    if not isinstance(groups, (list, tuple)):
        return None, "观测数据缺少分组明细"
    parsed: dict[str, dict[str, int]] = {}
    for item in groups:
        if not isinstance(item, dict):
            return None, "观测分组明细格式不正确"
        name = str(item.get("分组") or "").strip()
        if name not in GROUPS:
            return None, f"未知分组「{name}」，只接受 control/treatment"
        if name in parsed:
            return None, f"分组「{name}」在同一次观测中重复出现"
        n = _to_int(item.get("样本量"))
        conv = _to_int(item.get("转化人数"))
        if n is None or n <= 0:
            return None, f"分组「{name}」样本量必须为正整数"
        if conv is None or conv < 0:
            return None, f"分组「{name}」转化人数必须为非负整数"
        if conv > n:
            return None, f"分组「{name}」转化人数不能超过样本量"
        parsed[name] = {"样本量": n, "转化人数": conv}
    missing = [g for g in GROUPS if g not in parsed]
    if missing:
        return None, f"观测缺少分组：{'、'.join(missing)}，两组的样本量和转化数须来自同一次有效观测"
    return parsed, ""


def record_observation(
    experiment_code: str,
    groups: Any,
    source: str = "补报更正",
) -> tuple[dict[str, Any] | None, str]:
    """登记新的观测版本（如日报生成期间的补报），两组数据必须同批提交。"""
    code = str(experiment_code or "").strip()
    if not code:
        return None, "实验编号不能为空"
    parsed, error = _validate_groups(groups)
    if parsed is None:
        return None, error
    table = _observations()
    versions = [int(obs.get("观测版本", 0)) for obs in table if obs.get("实验编号") == code]
    version = max(versions, default=0) + 1
    entry = {
        "实验编号": code,
        "观测版本": version,
        "来源": source,
        "记录时间": datetime.now().isoformat(),
        "groups": [
            {"分组": name, "样本量": parsed[name]["样本量"], "转化人数": parsed[name]["转化人数"]}
            for name in GROUPS
        ],
    }
    table.append(entry)
    return dict(entry), f"实验 {code} 已登记第 {version} 版观测"


def get_observation(experiment_code: str, version: int | None = None) -> tuple[dict[str, Any] | None, str]:
    """取指定版本观测，缺省取最新一版；取出的两组数据必然来自同一版本。"""
    candidates = [obs for obs in _observations() if obs.get("实验编号") == experiment_code]
    if not candidates:
        return None, f"实验 {experiment_code} 没有任何观测数据"
    if version is None:
        chosen = max(candidates, key=lambda obs: int(obs.get("观测版本", 0)))
    else:
        chosen = next((obs for obs in candidates if int(obs.get("观测版本", 0)) == version), None)
        if chosen is None:
            return None, f"实验 {experiment_code} 不存在第 {version} 版观测"
    parsed, error = _validate_groups(chosen.get("groups"))
    if parsed is None:
        return None, f"实验 {experiment_code} 第 {chosen.get('观测版本')} 版观测无效：{error}"
    return {
        "观测版本": int(chosen["观测版本"]),
        "来源": chosen.get("来源"),
        "groups": parsed,
    }, ""


# ---------------------------------------------------------------------------
# 显著性水平：必须来自实验配置且有效，不允许默默回退到 5%
# ---------------------------------------------------------------------------

def parse_significance_level(raw: Any) -> tuple[float | None, str]:
    """解析实验配置的显著性水平；无效时返回错误原因，绝不回退到 0.05。"""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None, "显著性水平为空"
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        value = float(text)
    except ValueError:
        return None, f"显著性水平「{raw}」不是有效数值"
    if is_percent:
        value /= 100.0
    if not 0.0 < value < 1.0:
        return None, f"显著性水平 {value} 超出 (0, 1) 有效区间"
    return value, ""


def experiment_alpha(experiment_code: str) -> tuple[float | None, str]:
    """从实验管理配置读取显著性水平；同一编号多次登记时以最新登记为准。"""
    matches = [row for row in store.rows("experiment") if str(row.get("实验编号")) == experiment_code]
    if not matches:
        return None, f"实验 {experiment_code} 未在实验管理中登记，无法确定显著性水平"
    return parse_significance_level(matches[-1].get("显著性水平"))


# ---------------------------------------------------------------------------
# 双比例 z 检验：门槛由显著性水平决定（双侧）
# ---------------------------------------------------------------------------

def compute_two_proportion(n_c: int, conv_c: int, n_t: int, conv_t: int, alpha: float) -> dict[str, Any]:
    """对同一次观测的两组数据做双比例 z 检验，返回带实际门槛的统计量。"""
    p_c = conv_c / n_c
    p_t = conv_t / n_t
    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se if se else 0.0
    cohen_h = 2 * (math.asin(math.sqrt(p_t)) - math.asin(math.sqrt(p_c)))
    z_critical = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    significant = abs(z) > z_critical
    return {
        "control_rate": round(p_c, 4),
        "treatment_rate": round(p_t, 4),
        "absolute_lift": round(p_t - p_c, 4),
        "relative_lift": round((p_t - p_c) / p_c, 4) if p_c else None,
        "z_score": round(z, 4),
        "cohen_h": round(cohen_h, 4),
        "alpha": alpha,
        "z_critical": round(z_critical, 4),
        "significant": significant,
        "conclusion": "显著" if significant else "不显著",
    }


# ---------------------------------------------------------------------------
# 统计摘要：按（实验编号, 观测版本）发布后冻结
# ---------------------------------------------------------------------------

def _summaries() -> list[dict[str, Any]]:
    return store.rows(SUMMARY_TABLE)


def get_published_summary(experiment_code: str, version: int | None = None) -> dict[str, Any] | None:
    """读取已发布的统计摘要；缺省返回最新一版，指定版本时重现当时结论。"""
    matches = [s for s in _summaries() if s.get("experiment_code") == experiment_code]
    if version is not None:
        matches = [s for s in matches if int(s.get("observation_version", 0)) == version]
    if not matches:
        return None
    chosen = max(matches, key=lambda s: int(s.get("observation_version", 0)))
    return dict(chosen)


def _write_mart_snapshot(summary: dict[str, Any]) -> None:
    """把带实际门槛的统计摘要写入数据集市，供集市报表详情读取。"""
    store.rows("data_mart").append({
        "主题编号": f"DATA-STAT-{summary['experiment_code']}",
        "region": "all",
        "channel": "all",
        "日期": str(date.today()),
        "实验编号": summary["experiment_code"],
        "观测版本": summary["observation_version"],
        "对照组样本量": summary["sample_size_control"],
        "对照组转化人数": summary["conversions_control"],
        "实验组样本量": summary["sample_size_treatment"],
        "实验组转化人数": summary["conversions_treatment"],
        "控制组转化率": summary["control_rate"],
        "实验组转化率": summary["treatment_rate"],
        "相对提升": summary["relative_lift"],
        "显著性水平": summary["alpha"],
        "临界值": summary["z_critical"],
        "z值": summary["z_score"],
        "显著性": summary["significant"],
        "结论": summary["conclusion"],
    })


def run_experiment_stat(experiment_code: str, version: int | None = None) -> dict[str, Any]:
    """统计入口：按实验配置的显著性水平，对指定观测版本做双比例检验并发布摘要。"""
    observation, error = get_observation(experiment_code, version)
    if observation is None:
        return {"error": error}
    obs_version = observation["观测版本"]

    published = get_published_summary(experiment_code, obs_version)
    if published is not None:
        # 已发布的版本保持原样本、原门槛与原结论，不被新配置重新解释
        published["reused_published"] = True
        return published

    alpha, error = experiment_alpha(experiment_code)
    if alpha is None:
        return {"error": f"显著性水平无效，不能发布结论：{error}"}

    control = observation["groups"]["control"]
    treatment = observation["groups"]["treatment"]
    stats = compute_two_proportion(
        control["样本量"], control["转化人数"],
        treatment["样本量"], treatment["转化人数"],
        alpha,
    )
    summary = {
        "experiment_code": experiment_code,
        "observation_version": obs_version,
        "observation_source": observation.get("来源"),
        "alpha": stats["alpha"],
        "z_critical": stats["z_critical"],
        "significant": stats["significant"],
        "conclusion": stats["conclusion"],
        "z_score": stats["z_score"],
        "control_rate": stats["control_rate"],
        "treatment_rate": stats["treatment_rate"],
        "absolute_lift": stats["absolute_lift"],
        "relative_lift": stats["relative_lift"],
        "cohen_h": stats["cohen_h"],
        "sample_size_control": control["样本量"],
        "conversions_control": control["转化人数"],
        "sample_size_treatment": treatment["样本量"],
        "conversions_treatment": treatment["转化人数"],
        "published_at": datetime.now().isoformat(),
    }
    _summaries().append(summary)
    _write_mart_snapshot(summary)
    return dict(summary)


def mart_experiment_report(experiment_code: str, version: int | None = None) -> tuple[dict[str, Any] | None, str]:
    """集市报表详情：读取产生显著性结果的样本与显著性水平，输出与统计入口一致。"""
    summary = get_published_summary(experiment_code, version)
    if summary is None:
        scope = f"第 {version} 版观测的" if version is not None else "任何"
        return None, f"实验 {experiment_code} 尚未发布{scope}统计摘要"
    return summary, ""
