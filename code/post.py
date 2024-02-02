import argparse
import json
import os


def export_ecf_diff(fp: str, ecf_diff: dict):
    if not os.path.isdir(os.path.dirname(fp)):
        raise NotADirectoryError(os.path.dirname(fp))

    trials = [int(k) for k in ecf_diff.keys()]
    trials.sort()
    var_names = list(ecf_diff[str(trials[0])].keys())

    data = {name: list() for name in var_names}
    for t in trials:
        for name in var_names:
            data[name].append(ecf_diff[str(t)][name])

    with open(fp, 'w') as f:
        for name, name_data in data.items():
            f.write(f'Name,{name}\n')
            f.write('Replicates,Diff\n')
            for i, t in enumerate(trials):
                f.write(f'{t},{name_data[i]}\n')


def extract_ecf_diff(fp_json: str, fp_output: str):
    with open(fp_json, 'r') as f:
        file_data = json.load(f)
    export_ecf_diff(fp_output, file_data['ecf_diff'])


class ArgParser(argparse.ArgumentParser):

    def __init__(self) -> None:
        super().__init__(description='Post-processing for stochastic model reproducibility pipeline')

        self.add_argument('-o', '--output-dir',
                          type=str,
                          required=False,
                          dest='output_dir',
                          help='Absolute path of output directory. Takes precedence over relative path of output directory.')

        self.add_argument('-ro', '--rel-output-dir',
                          type=str,
                          required=False,
                          dest='rel_output_dir',
                          help='Relative path of output directory')

        self.add_argument('-d', '--dir', 
                          type=str, 
                          required=False, 
                          dest='directory', 
                          help='Absolute path of directory containing pipeline JSON data')

        self.add_argument('-rd', '--rel-dir', 
                          type=str, 
                          required=False, 
                          dest='rel_directory', 
                          help='Absolute path of directory containing pipeline JSON data')

        self.add_argument('-f', '--file',
                          type=str,
                          required=False,
                          dest='file',
                          help='Absolute path of pipeline JSON data')

        self.add_argument('-rf', '--rel-file',
                          type=str,
                          required=False,
                          dest='rel_file',
                          help='Relative path of pipeline JSON data')

        self.parsed_args = self.parse_args()

    @property
    def output_dir(self):
        return self.parsed_args.output_dir

    @property
    def rel_output_dir(self):
        return self.parsed_args.rel_output_dir

    @property
    def directory(self):
        return self.parsed_args.directory

    @property
    def rel_directory(self):
        return self.parsed_args.rel_directory

    @property
    def file(self):
        return self.parsed_args.file

    @property
    def rel_file(self):
        return self.parsed_args.rel_file


if __name__ == '__main__':
    parser = ArgParser()
    parsed_args = parser.parse_args()

    output_dir = None
    if parser.rel_output_dir is not None:
        output_dir = os.path.join(os.getcwd(), parser.rel_output_dir)
    if parser.output_dir is not None:
        output_dir = parser.output_dir

    if output_dir is None:
        raise RuntimeError('No output directory define')

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(output_dir)

    _to_process = []
    if parser.file is not None:
        _to_process.append(parser.file)

    if parser.rel_file is not None:
        _to_process.append(os.path.join(os.getcwd(), parser.rel_file))

    if parser.directory is not None:
        for f in os.listdir(parser.directory):
            if f.endswith('.json'):
                _to_process.append(os.path.join(parser.directory, f))

    if parser.rel_directory is not None:
        for f in os.listdir(os.path.join(os.getcwd(), parser.rel_directory)):
            if f.endswith('.json'):
                _to_process.append(os.path.join(os.getcwd(), parser.rel_directory, f))

    for f in _to_process:
        dirname = os.path.dirname(f)
        basename = os.path.basename(f).replace('.json', '')
        export_path_ecf_diff = os.path.join(dirname, basename + '.csv')
        extract_ecf_diff(f, export_path_ecf_diff)
