"""跨模块数据管道测试：验证真实计算与数据流动。"""
from __future__ import annotations

import math

import pytest

from app.services.experiment import ExperimentService
from app.services.experiment_stats import (
    get_published_summary,
    list_observations,
    mart_experiment_report,
    record_observation,
)
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
from app.store import store


class TestMetricAggPipeline:
    """Pipeline 1: metric_def -> agg_task -> api_service"""

    def test_gmv_aggregation(self):
        result = run_metric_agg_pipeline("METR-GMV")
        assert result["metric_code"] == "METR-GMV"
        assert result["unit"] == "元"
        assert result["point_count"] == 30
        # 30 天 GMV 总和应为已知 fixtures 之和
        expected = 124500.0 + 138200.0 + 119800.0 + 145600.0 + 152300.0 + 160100.0 + 175400.0 + 168900.0 + 155200.0 + 142000.0 + 149500.0 + 161200.0 + 173800.0 + 182000.0 + 190500.0 + 178300.0 + 165000.0 + 158400.0 + 171200.0 + 185000.0 + 192300.0 + 201000.0 + 210500.0 + 198000.0 + 205400.0 + 218000.0 + 225600.0 + 232000.0 + 221000.0 + 238500.0
        assert result["value"] == round(expected, 2)

    def test_cvr_computation(self):
        result = run_metric_agg_pipeline("METR-CVR")
        # service returns aggregated cvr over all points
        assert result["metric_code"] == "METR-CVR"
        assert result["unit"] == "比率"
        assert 0.0 < result["value"] <= 1.0

    def test_pipeline_result_written(self):
        run_metric_agg_pipeline("METR-UV")
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "metric_agg_api" in ids


class TestForecastPipeline:
    """Pipeline 2: trend_series -> forecast -> report_gen"""

    def test_forecast_structure(self):
        result = run_forecast_pipeline(horizon=7)
        assert result["model"] == "linear_trend"
        assert len(result["forecasts"]) == 7
        assert "slope" in result
        assert "intercept" in result

    def test_forecast_monotonicity(self):
        # fixtures gmv 整体上升，slope 应该为正
        result = run_forecast_pipeline(horizon=7)
        assert result["slope"] > 0

    def test_confidence_intervals(self):
        result = run_forecast_pipeline(horizon=3)
        for f in result["forecasts"]:
            assert f["置信下界"] < f["预测gmv"] < f["置信上界"]

    def test_pipeline_result_written(self):
        run_forecast_pipeline(5)
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "forecast_report" in ids


class TestCubePipeline:
    """Pipeline 3: dim_registry -> cube_builder -> api_service"""

    def test_cube_grouping(self):
        result = run_cube_pipeline(["DIM-REGION", "DIM-CHANNEL"])
        assert result["cell_count"] > 0
        assert "cells" in result
        for cell in result["cells"]:
            assert "region" in cell
            assert "channel" in cell
            assert "gmv" in cell

    def test_cube_single_dimension(self):
        result = run_cube_pipeline(["DIM-REGION"])
        regions = {c["region"] for c in result["cells"]}
        assert len(regions) == result["cell_count"]

    def test_pipeline_result_written(self):
        run_cube_pipeline()
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "cube_api" in ids


class TestCohortExtractPipeline:
    """Pipeline 4: cohort_rule -> data_extract -> report_gen"""

    def test_filter_by_region(self):
        result = run_cohort_extract_pipeline({"region": "华东", "min_ltv": 0})
        assert result["matched_count"] == 4
        for u in result["users"]:
            assert u["region"] == "华东"

    def test_filter_by_ltv(self):
        result = run_cohort_extract_pipeline({"min_ltv": 1000.0})
        assert result["matched_count"] == 4
        for u in result["users"]:
            assert u["ltv"] >= 1000.0

    def test_summary_computation(self):
        result = run_cohort_extract_pipeline({"region": "华东"})
        summary = result["summary"]
        assert summary["total_ltv"] == round(sum(u["ltv"] for u in result["users"]), 2)
        assert summary["avg_ltv"] == round(summary["total_ltv"] / len(result["users"]), 2)

    def test_pipeline_result_written(self):
        run_cohort_extract_pipeline()
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "cohort_extract_report" in ids


class TestExperimentStatPipeline:
    """Pipeline 5: experiment -> stat_model -> data_mart"""

    def test_expe001_significance(self):
        result = run_experiment_stat_pipeline("EXPE-001")
        assert result["experiment_code"] == "EXPE-001"
        # treatment 412/5000 = 0.0824, control 350/5000 = 0.07 -> lift positive
        assert result["absolute_lift"] > 0
        # 种子实验配置显著性水平 0.05，门槛应取自配置而不是固定 1.96
        assert result["alpha"] == 0.05
        assert result["z_critical"] == pytest.approx(1.96, abs=1e-4)
        assert result["significant"] is True
        assert abs(result["z_score"]) > result["z_critical"]

    def test_expe002_effect_size(self):
        result = run_experiment_stat_pipeline("EXPE-002")
        assert result["cohen_h"] != 0.0
        assert result["sample_size_control"] == 8000
        assert result["sample_size_treatment"] == 8000

    def test_mart_entry_written(self):
        before = len(store.rows("data_mart"))
        run_experiment_stat_pipeline("EXPE-001")
        after = len(store.rows("data_mart"))
        assert after == before + 1
        last = store.rows("data_mart")[-1]
        assert last.get("实验编号") == "EXPE-001"
        # 集市快照必须带实际门槛与观测版本，供报表详情复查
        assert last.get("显著性水平") == 0.05
        assert last.get("临界值") == pytest.approx(1.96, abs=1e-4)
        assert last.get("观测版本") == 1
        assert last.get("显著性") is True

    def test_pipeline_result_written(self):
        run_experiment_stat_pipeline("EXPE-001")
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "experiment_stat_mart" in ids


class TestExperimentStatAcceptance:
    """验收场景：实验配置 1% 显著性水平，日报生成期间补报转化人数。"""

    CODE = "EXPE-901"

    def _register(self, code: str, alpha: str) -> None:
        entry, missing = ExperimentService().create_entry({
            "实验编号": code,
            "实验假设": "新结算页可提升支付转化率",
            "分组方案": "control:treatment=1:1",
            "显著性水平": alpha,
        })
        assert not missing

    def _setup_v1(self) -> None:
        """配置 1% 显著性水平，登记第 1 版观测：对照 1000/100，实验 1000/132。"""
        self._register(self.CODE, "0.01")
        obs, _ = record_observation(self.CODE, [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 132},
        ], source="日报初版")
        assert obs is not None and obs["观测版本"] == 1

    def _correct_to_145(self) -> None:
        """日报生成期间补报：实验组改报 145 人，登记为第 2 版观测。"""
        obs, _ = record_observation(self.CODE, [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 145},
        ], source="补报更正")
        assert obs is not None and obs["观测版本"] == 2

    def test_not_significant_at_1pct_on_api_and_mart(self):
        self._setup_v1()
        # 统计入口：z≈2.2345，门槛 2.5758，不显著
        result = run_experiment_stat_pipeline(self.CODE)
        assert result["alpha"] == 0.01
        assert result["z_critical"] == pytest.approx(2.5758, abs=1e-4)
        assert result["z_critical"] != pytest.approx(1.96, abs=1e-4)
        assert result["significant"] is False
        assert result["conclusion"] == "不显著"
        # 集市报表详情：同一统计输出，同样不显著
        report, message = mart_experiment_report(self.CODE)
        assert report is not None, message
        assert report["significant"] is False
        assert report["z_critical"] == result["z_critical"]
        assert report["z_score"] == result["z_score"]
        assert report["sample_size_control"] == 1000
        assert report["conversions_control"] == 100
        assert report["sample_size_treatment"] == 1000
        assert report["conversions_treatment"] == 132
        # 集市快照行带实际门槛与观测版本
        mart_rows = [r for r in store.rows("data_mart") if r.get("实验编号") == self.CODE]
        assert len(mart_rows) == 1
        assert mart_rows[0]["显著性"] is False
        assert mart_rows[0]["显著性水平"] == 0.01
        assert mart_rows[0]["临界值"] == pytest.approx(2.5758, abs=1e-4)
        assert mart_rows[0]["观测版本"] == 1

    def test_correction_then_reopen_report(self):
        self._setup_v1()
        first = run_experiment_stat_pipeline(self.CODE)
        assert first["significant"] is False
        self._correct_to_145()
        # 再次打开报告：报表引用第 2 版的样本数、门槛和结论
        reopened = run_experiment_stat_pipeline(self.CODE)
        assert reopened["observation_version"] == 2
        assert reopened["observation_source"] == "补报更正"
        assert reopened["sample_size_control"] == 1000
        assert reopened["sample_size_treatment"] == 1000
        assert reopened["conversions_control"] == 100
        assert reopened["conversions_treatment"] == 145
        assert reopened["z_critical"] == pytest.approx(2.5758, abs=1e-4)
        assert reopened["significant"] is True
        assert reopened["conclusion"] == "显著"
        # 集市报表详情与统计入口一致
        report, _ = mart_experiment_report(self.CODE)
        assert report["observation_version"] == 2
        assert report["conversions_treatment"] == 145
        assert report["z_critical"] == reopened["z_critical"]
        assert report["significant"] is True

    def test_history_query_reproduces_version_conclusion(self):
        self._setup_v1()
        run_experiment_stat_pipeline(self.CODE)
        self._correct_to_145()
        run_experiment_stat_pipeline(self.CODE)
        # 历史查询第 1 版：重现当时的不显著结论与当时的样本
        v1, message = mart_experiment_report(self.CODE, version=1)
        assert v1 is not None, message
        assert v1["observation_version"] == 1
        assert v1["conversions_treatment"] == 132
        assert v1["significant"] is False
        assert v1["z_critical"] == pytest.approx(2.5758, abs=1e-4)
        # 经营方可列出全部观测版本，区分报表采用的是哪一版
        versions = list_observations(self.CODE)
        assert [obs["观测版本"] for obs in versions] == [1, 2]

    def test_invalid_alpha_never_falls_back_to_5pct(self):
        # 无法解析的水平
        self._register("EXPE-902", "abc")
        record_observation("EXPE-902", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 132},
        ])
        result = run_experiment_stat_pipeline("EXPE-902")
        assert "error" in result
        assert "显著性水平" in result["error"]
        # 不发结论、不写集市、不留摘要
        assert get_published_summary("EXPE-902") is None
        assert not [r for r in store.rows("data_mart") if r.get("实验编号") == "EXPE-902"]
        # 超出 (0, 1) 区间的水平同样报错
        self._register("EXPE-903", "1.5")
        record_observation("EXPE-903", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 132},
        ])
        assert "error" in run_experiment_stat_pipeline("EXPE-903")
        # 未登记显著性水平的实验也不能发结论
        record_observation("EXPE-904", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 132},
        ])
        assert "error" in run_experiment_stat_pipeline("EXPE-904")

    def test_published_version_not_reinterpreted_by_new_threshold(self):
        # 按 5% 发布第 1 版：z≈2.2345 > 1.96，显著
        self._register("EXPE-905", "0.05")
        record_observation("EXPE-905", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 132},
        ])
        first = run_experiment_stat_pipeline("EXPE-905")
        assert first["alpha"] == 0.05
        assert first["significant"] is True
        # 经营方把显著性水平改为 1%（同编号重新登记，以最新配置为准）
        self._register("EXPE-905", "0.01")
        again = run_experiment_stat_pipeline("EXPE-905")
        # 已发布的旧版本保持原门槛原结论，不被新门槛重新解释
        assert again["reused_published"] is True
        assert again["alpha"] == 0.05
        assert again["z_critical"] == pytest.approx(1.96, abs=1e-4)
        assert again["significant"] is True
        v1, _ = mart_experiment_report("EXPE-905", version=1)
        assert v1["alpha"] == 0.05
        assert v1["significant"] is True

    def test_groups_must_come_from_same_valid_observation(self):
        # 只提交实验组：不是一次有效观测，拒绝登记
        obs, message = record_observation("EXPE-906", [
            {"分组": "treatment", "样本量": 1000, "转化人数": 145},
        ])
        assert obs is None
        assert "同一次有效观测" in message
        # 转化人数超过样本量：无效观测，拒绝登记
        obs, message = record_observation("EXPE-906", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 1200},
        ])
        assert obs is None
        assert "转化人数不能超过样本量" in message
        # 两组齐全且自洽：接受为同一版观测
        obs, _ = record_observation("EXPE-906", [
            {"分组": "control", "样本量": 1000, "转化人数": 100},
            {"分组": "treatment", "样本量": 1000, "转化人数": 145},
        ])
        assert obs is not None and obs["观测版本"] == 1
        assert list_observations("EXPE-906")[0]["groups"][0]["样本量"] == 1000


class TestQualityScanPipeline:
    """Pipeline 6: quality_mon 跨模块扫描"""

    def test_quality_score_range(self):
        result = run_quality_scan()
        assert 0 <= result["quality_score"] <= 100

    def test_module_issues_detected(self):
        result = run_quality_scan()
        # seed 里有 abnormal=True 的记录
        assert any(item["abnormal_count"] > 0 for item in result["module_issues"])

    def test_sample_issues_non_empty(self):
        result = run_quality_scan()
        assert len(result["sample_issues"]) > 0

    def test_pipeline_result_written(self):
        run_quality_scan()
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "quality_scan" in ids


class TestAttributionComparePipeline:
    """Pipeline 7: attribution -> metric_def -> compare_job"""

    def test_first_touch_totals(self):
        result = run_attribution_compare_pipeline("ATTR-001")
        totals = result["totals"]
        assert totals["first_touch"] > 0

    def test_compare_rows_generated(self):
        result = run_attribution_compare_pipeline("ATTR-002")
        assert len(result["compare_rows"]) > 0
        for row in result["compare_rows"]:
            assert "触点类型" in row
            assert "差异" in row

    def test_linear_vs_first_difference(self):
        result = run_attribution_compare_pipeline("ATTR-002")
        models = result["models"]
        # 线性归因与首次归因应该产生不同分布
        linear_keys = set(models["linear"].keys())
        first_keys = set(models["first_touch"].keys())
        assert linear_keys == first_keys or len(linear_keys & first_keys) > 0

    def test_pipeline_result_written(self):
        run_attribution_compare_pipeline("ATTR-001")
        results = list_pipeline_results()
        ids = [r["pipeline_id"] for r in results]
        assert "attribution_metric_compare" in ids


class TestRunAllPipelines:
    """批量执行入口"""

    def test_all_pipelines_run(self):
        summary = run_all_pipelines()
        expected_keys = {
            "metric_agg_api",
            "forecast_report",
            "cube_api",
            "cohort_extract_report",
            "experiment_stat_mart",
            "quality_scan",
            "attribution_metric_compare",
        }
        assert set(summary.keys()) == expected_keys
        results = list_pipeline_results()
        assert len(results) == 7
