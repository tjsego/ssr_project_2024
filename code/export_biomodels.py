"""
Script to export data for BioModels entries.
Requires libSSR Python on path
"""

import json
import libssr
import os
import numpy as np
from stochastic_tests import Test
from xml.dom import minidom
from xml.etree import ElementTree

rr_sig_figs = 9
"""RoadRunner output"""
dpi = 300

this_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(this_dir, 'results', 'test_231205_0')
output_dir = os.path.join(results_dir, 'export')
render_dir = os.path.join(output_dir, 'render')

available_results = [f for f in os.listdir(results_dir) if f.startswith('biomodels_') and f.endswith('.json')]
available_names = [f.replace('biomodels_', '').replace('.json', '') for f in available_results]
results_names = ['MODEL' + f for f in available_names]
available_paths = [os.path.join(results_dir, f) for f in available_results]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(render_dir):
    os.mkdir(render_dir)


def generate_report(test: Test) -> libssr.EFECTReport:
    target_size = test.trials[-1]
    sample_times = test.sims_s[target_size].results_time
    num_times = len(sample_times)
    num_names = len(test.model.results_names)
    # Tests were performed with uniform number of ECF evaluations
    ecf_nval = test.ecf_eval_info[target_size][0][test.model.results_names[0]][0]

    sample_size = target_size // 2

    ecf_eval = np.ndarray((num_times, num_names, ecf_nval, 2), dtype=float)
    ecf_tval = np.ndarray((num_times, num_names), dtype=float)
    for j, name in enumerate(test.model.results_names):
        results = test.sims_s[target_size].results[name][:sample_size, :]
        for i in range(num_times):
            ecf_tval[i, j] = test.ecf_eval_info[target_size][i][name][1]
            eval_t = libssr.get_eval_info_times(ecf_nval, ecf_tval[i, j])
            ecf = libssr.ecf(results[:, i], eval_t)
            ecf_eval[i, j, :, :] = ecf

    return libssr.EFECTReport.create(
        test.model.results_names,
        sample_times,
        sample_size,
        ecf_eval,
        ecf_tval,
        ecf_nval,
        test.ecf_sampling[target_size][0],
        test.ecf_sampling[target_size][1],
        rr_sig_figs
    )


def save_report_xml(report: libssr.EFECTReport, path: str):
    with open(path, 'w') as f:
        xml_str = ElementTree.tostring(report.to_xml())
        xml_str_pretty = minidom.parseString(xml_str).toprettyxml(indent='    ')
        f.write(xml_str_pretty)


def save_report_json(report: libssr.EFECTReport, path: str):
    with open(path, 'w') as f:
        json.dump(report.to_json(), f, indent=4)


def render_test(test: Test, name: str, preview=False):

    test_copy = test.clone()

    # plot EFECT error vs. sample size

    fig, ax = test.plot_ecf_sampling()
    if preview:
        fig.show()
    else:
        fig.savefig(os.path.join(render_dir, name + '_error_vs_size.png'), dpi=dpi)

    # plot EFECT error sampling

    fig, ax = test.plot_ks_sampling()
    if preview:
        fig.show()
    else:
        fig.savefig(os.path.join(render_dir, name + '_error_sampling.png'), dpi=dpi)

    # keep only largest sample

    test_copy.trials = [test_copy.trials[-1]]

    # plot trajectories

    fig, ax = test_copy.plot_results_stochastic(False)
    if preview:
        fig.show()
    else:
        fig.savefig(os.path.join(render_dir, name + '_stochastic.png'), dpi=dpi)

    # plot distribution

    fig, ax = test_copy.plot_distributions()
    if preview:
        fig.show()
    else:
        fig.savefig(os.path.join(render_dir, name + '_distributions.png'), dpi=dpi)


def main():

    for i in range(len(results_names)):
        print(f'{results_names[i]}...')
        print('...loading data')
        test = Test.load(available_paths[i])
        print('...generating report')
        report = generate_report(test)
        print('...saving reports:', output_dir)
        save_report_json(report, os.path.join(output_dir, results_names[i] + '.json'))
        save_report_xml(report, os.path.join(output_dir, results_names[i] + '.xml'))
        print('...rendering:', render_dir)
        render_test(test, results_names[i])


if __name__ == '__main__':
    main()
