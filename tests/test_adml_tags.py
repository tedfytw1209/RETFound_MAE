# tests/test_adml_tags.py
"""wandb tags of the OCT AD runs (util/adml_tags.py, adml_wandb_retag.py), on run names from OCT_ADML_experiment.xlsx."""
import pytest

from adml_wandb_retag import new_tags
from util import adml_tags

ROOT = '/blue/ruogu.fang/tienyuchang'
SPLITS = f'{ROOT}/OCTAD_ML_pipeline/psrs_oct'


def main_tags(split_dir, data_path, task, passed):
    """What main_finetune_Chua_Jacqueline.py logs (before it appends class_weight)."""
    text = f'{split_dir} {data_path} {task}'
    return adml_tags.tags_for(text, passed, method='DL', task=adml_tags.task_of(task))


def test_new_dl_study3_run():
    assert main_tags(f'{SPLITS}/IRB2024_combined_study3_retinaf_diseaseclean_1yr/split_mcicon',
                     f'{ROOT}/IRB2024v5_ADCON_DL_study3_retinaf_1yr/mci_control_detect_data.csv',
                     'mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Jacqueline-convnext_tiny-Thickness-'
                     'bs64ep100lr1e-3wd5e-4-roc_auceval-subset0',
                     ['study3', 'mci_control', '1yr']) == ['study3', 'DL', 'year1', 'mci_control']


def test_new_dl_run_without_launcher_tags_gets_them_from_the_paths():
    assert main_tags(f'{SPLITS}/IRB2024_combined_study3_retinaf_diseaseclean_10yr/split_admcicon',
                     f'{ROOT}/IRB2024v5_ADCON_DL_study3_retinaf_10yr/ad_mci_control_detect_data.csv',
                     'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_10yr-Wisely-resnet18_paper', []) \
        == ['study3', 'DL', 'year10', 'ad_mci_control']


def test_new_dl_study2_ad_rest_drops_a_leaked_period_and_keeps_extra_tags():
    # ad_rest reads the ad_mci_control CSV: the task comes from the task name, not the data path
    assert main_tags(f'{SPLITS}/IRB2024_prev_combined_study2_retinaf_diseaseclean/split_adrest',
                     f'{ROOT}/IRB2024v5_ADCON_DL_data_retinaf_diseaseclean/ad_mci_control_detect_data.csv',
                     'ad_rest_detect_data-IRB2024v5_ADCON_DL_data_retinaf_diseaseclean-Wisely-resnet18_paper',
                     ['study2', 'ad_rest', '3yr', 'rerun']) == ['study2', 'DL', 'ad_rest', 'rerun']


@pytest.mark.parametrize('name, tags, expected', [
    ('ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_10yr-Hebei-ducan-dual-bs8ep400lr3e-4wd1e-2-'
     'accuracyeval-subset0_CW', ['study3', 'ad_mci_control', '10yr', 'class_weight'],
     ['study3', 'DL', 'year10', 'ad_mci_control', 'class_weight']),
    ('ad_control_detect_data-IRB2024v5_ADCON_DL_data_retinaf-Wisely2-dual_input_cnn-images_only-Thickness-'
     'roc_auceval-subset0', [], ['study2', 'DL', 'ad_control']),
    ('WangXin_ML_New_IRB2024_combined_study3_retinaf_diseaseclean_1yr_ad_mci_control_CVTrue_GSTrue_N10_SUBFalse_SUBR2.0_CW',
     [], ['study3', 'ML', 'year1', 'ad_mci_control']),
    ('Mei_Xi_relative_IRB2024_prev_combined_study2_retinaf_diseaseclean_1777176105', ['DL'], ['study2', 'ML']),
    ('NunesAna_paper_svm_groupedcv_IRB2024_prev_combined_study2_retinaf_1777167963', [], ['study2', 'ML']),
    ('TY_ML_IRB2024_prev_combined_study2_retinaf_diseaseclean_mci_control_CVTrue_N10_SUBFalse_SUBR2.0_modelall_FS20_'
     'SPLITIRB2024_prev_combined_study2_retinaf_diseaseclean', ['ML'], ['study2', 'our_proposed', 'mci_control']),
])
def test_retag_existing_runs(name, tags, expected):
    assert new_tags(name, {}, tags)[2] == expected


def test_retag_ad_rest_dl_run_uses_the_logged_task():
    name = 'ad_rest_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_5yr-Mahendran-ad_oct_model-OCT_CW'
    config = {'task': name, 'data_path': f'{ROOT}/IRB2024v5_ADCON_DL_study3_retinaf_5yr/ad_mci_control_detect_data.csv',
              'split_dir': f'{SPLITS}/IRB2024_combined_study3_retinaf_diseaseclean_5yr/split_adrest'}
    assert new_tags(name, config, ['study3', 'ad_rest', '5yr'])[2] == ['study3', 'DL', 'year5', 'ad_rest']


def test_unrecognized_and_unknown_study_runs():
    assert new_tags('RETFound_mae-something', {}, ['x']) == (None, None, ['x'])
    method, study, tags = new_tags('JinZ_bscan_run', {}, ['3yr'])
    assert (method, study, tags) == ('ML', None, ['ML', 'year3'])


def test_period_needs_a_yr_or_year_token():
    assert adml_tags.period_of('Mei_Xi_relative_1777176105') is None
    assert adml_tags.period_of('x_year3_y') == 'year3'
