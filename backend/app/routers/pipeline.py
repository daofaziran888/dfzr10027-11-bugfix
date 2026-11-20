"""数据管道接口：触发跨模块计算并查询产物。"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas import ActionResult
from app.services import experiment_stats
from app.services.pipeline_orchestrator import (
    list_pipeline_results,
    run_all_pipelines,
    run_attribution_compare_pipeline,
    run_cohort_extract_pipeline,
    run_cube_pipeline,
    run_experiment_stat_pipeline,
    run_forecast_pipeline,
    run_metric_agg_pipeline,
    run_quality_scan,
)

router = APIRouter(prefix="/api/pipeline", tags=["数据管道"])


@router.post("/run-all", response_model=ActionResult)
def trigger_all() -> ActionResult:
    """顺序执行全部 7 条跨模块数据管道。"""
    summary = run_all_pipelines()
    return ActionResult(
        ok=True,
        message="全部管道已执行",
        entry={"pipelines": list(summary.keys()), "count": len(summary)},
    )


@router.post("/metric-agg", response_model=ActionResult)
def trigger_metric_agg(
    metric_code: str | None = Query(default=None, description="指标编号，如 METR-GMV"),
) -> ActionResult:
    result = run_metric_agg_pipeline(metric_code)
    return ActionResult(ok=True, message="指标聚合管道已执行", entry=result)


@router.post("/forecast", response_model=ActionResult)
def trigger_forecast(
    horizon: int = Query(default=7, ge=1, le=30, description="预测天数"),
) -> ActionResult:
    result = run_forecast_pipeline(horizon)
    return ActionResult(ok=True, message="预测管道已执行", entry=result)


@router.post("/cube", response_model=ActionResult)
def trigger_cube(
    dim_codes: list[str] | None = Query(default=None, description="维度编号列表"),
) -> ActionResult:
    result = run_cube_pipeline(dim_codes)
    return ActionResult(ok=True, message="立方体构建管道已执行", entry=result)


@router.post("/cohort-extract", response_model=ActionResult)
def trigger_cohort_extract() -> ActionResult:
    result = run_cohort_extract_pipeline()
    return ActionResult(ok=True, message="分群提取管道已执行", entry=result)


@router.post("/experiment-stat", response_model=ActionResult)
def trigger_experiment_stat(
    experiment_code: str | None = Query(default=None, description="实验编号"),
    version: int | None = Query(default=None, description="观测版本，缺省取最新一版"),
) -> ActionResult:
    """统计入口：按实验配置的显著性水平计算双比例检验；水平无效时明确报错而不是默默按 5% 发结论。"""
    result = run_experiment_stat_pipeline(experiment_code, version)
    if "error" in result:
        return ActionResult(ok=False, message=result["error"], entry=result)
    if result.get("reused_published"):
        return ActionResult(
            ok=True,
            message=f"实验统计第 {result['observation_version']} 版摘要已发布，按原门槛原结论返回",
            entry=result,
        )
    return ActionResult(ok=True, message="实验统计管道已执行", entry=result)


@router.post("/experiment-observation", response_model=ActionResult)
def record_experiment_observation(payload: dict[str, object]) -> ActionResult:
    """登记实验观测版本（如日报生成期间的补报）；两组样本量与转化人数必须同批提交。"""
    code = str(payload.get("实验编号") or payload.get("experiment_code") or "").strip()
    groups = payload.get("groups") or payload.get("分组数据")
    source = str(payload.get("来源") or payload.get("source") or "补报更正")
    entry, message = experiment_stats.record_observation(code, groups, source)
    if entry is None:
        return ActionResult(ok=False, message=message)
    return ActionResult(ok=True, message=message, entry=entry)


@router.get("/experiment-stat/{experiment_code}", response_model=ActionResult)
def get_experiment_stat(
    experiment_code: str,
    version: int | None = Query(default=None, description="观测版本，缺省取最新已发布"),
) -> ActionResult:
    """复查已发布的实验统计摘要；指定版本可重现当时的显著性结论。"""
    summary = experiment_stats.get_published_summary(experiment_code, version)
    if summary is None:
        scope = f"第 {version} 版观测的" if version is not None else "任何"
        return ActionResult(ok=False, message=f"实验 {experiment_code} 尚未发布{scope}统计摘要")
    return ActionResult(ok=True, message="实验统计摘要", entry=summary)


@router.get("/experiment-observations/{experiment_code}")
def list_experiment_observations(experiment_code: str) -> dict[str, object]:
    """列出实验的全部观测版本，用于区分报表采用的是哪一版数据。"""
    return {"items": experiment_stats.list_observations(experiment_code)}


@router.post("/quality-scan", response_model=ActionResult)
def trigger_quality_scan() -> ActionResult:
    result = run_quality_scan()
    return ActionResult(ok=True, message="质量扫描管道已执行", entry=result)


@router.post("/attribution-compare", response_model=ActionResult)
def trigger_attribution_compare(
    link_code: str | None = Query(default=None, description="链路编号"),
) -> ActionResult:
    result = run_attribution_compare_pipeline(link_code)
    return ActionResult(ok=True, message="归因对比管道已执行", entry=result)


@router.get("/results")
def get_results() -> dict[str, object]:
    """查询全部管道产物。"""
    return {"items": list_pipeline_results()}
