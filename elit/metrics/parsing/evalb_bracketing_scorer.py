import os
import shutil
import subprocess
import tempfile
from typing import List

from phrasetree.tree import Tree

from elit.metrics.f1 import F1
from elit.metrics.metric import Metric
from elit.utils.io_util import get_resource, run_cmd
from elit.utils.log_util import cprint


class EvalbBracketingScorer(Metric):
    """
    This class uses the external EVALB software for computing a broad range of metrics
    on parse trees. Here, we use it to compute the Precision, Recall and F1 metrics.
    You can download the source for EVALB from here: <https://nlp.cs.nyu.edu/evalb/>.

    Note that this software is 20 years old. In order to compile it on modern hardware,
    you may need to remove an `include <malloc.h>` statement in `evalb.c` before it
    will compile.

    AllenNLP contains the EVALB software, but you will need to compile it yourself
    before using it because the binary it generates is system dependent. To build it,
    run `make` inside the `allennlp/tools/EVALB` directory.

    Note that this metric reads and writes from disk quite a bit. You probably don't
    want to include it in your training loop; instead, you should calculate this on
    a validation set only.

    # Parameters

    evalb_directory_path : `str`, required.
        The directory containing the EVALB executable.
    evalb_param_filename : `str`, optional (default = `"COLLINS.prm"`)
        The relative name of the EVALB configuration file used when scoring the trees.
        By default, this uses the nk.prm configuration file which comes with LAL-Parser.
        This configuration ignores POS tags, S1 labels and some punctuation labels.
    evalb_num_errors_to_kill : `int`, optional (default = `"10"`)
        The number of errors to tolerate from EVALB before terminating evaluation.
    """

    def __init__(
            self,
            evalb_directory_path: str = None,
            evalb_param_filename: str = "nk.prm",
            evalb_num_errors_to_kill: int = 10,
    ) -> None:
        if not evalb_directory_path:
            evalb_directory_path = get_resource('https://github.com/KhalilMrini/LAL-Parser/archive/master.zip#EVALB/')
        self._evalb_directory_path = evalb_directory_path
        self._evalb_program_path = os.path.join(evalb_directory_path, "evalb")
        self._evalb_param_path = os.path.join(evalb_directory_path, evalb_param_filename)
        self._evalb_num_errors_to_kill = evalb_num_errors_to_kill

        self._header_line = [
            "ID",
            "Len.",
            "Stat.",
            "Recal",
            "Prec.",
            "Bracket",
            "gold",
            "test",
            "Bracket",
            "Words",
            "Tags",
            "Accracy",
        ]

        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    def __call__(self, predicted_trees: List[Tree], gold_trees: List[Tree]) -> None:  # type: ignore
        """
        # Parameters

        predicted_trees : `List[Tree]`
            A list of predicted NLTK Trees to compute score for.
        gold_trees : `List[Tree]`
            A list of gold NLTK Trees to use as a reference.
        """
        if not os.path.exists(self._evalb_program_path):
            cprint(f"EVALB not found at {self._evalb_program_path}.  Attempting to compile it.")
            EvalbBracketingScorer.compile_evalb(self._evalb_directory_path)

            # If EVALB executable still doesn't exist, raise an error.
            if not os.path.exists(self._evalb_program_path):
                compile_command = (
                    f"python -c 'from allennlp.training.metrics import EvalbBracketingScorer; "
                    f'EvalbBracketingScorer.compile_evalb("{self._evalb_directory_path}")\''
                )
                raise RuntimeError(
                    f"EVALB still not found at {self._evalb_program_path}. "
                    "You must compile the EVALB scorer before using it."
                    " Run 'make' in the '{}' directory or run: {}".format(
                        self._evalb_program_path, compile_command
                    )
                )
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")
        with open(gold_path, "w") as gold_file:
            for tree in gold_trees:
                gold_file.write(f"{tree.pformat(margin=1000000)}\n")

        with open(predicted_path, "w") as predicted_file:
            for tree in predicted_trees:
                predicted_file.write(f"{tree.pformat(margin=1000000)}\n")

        command = [
            self._evalb_program_path,
            "-p",
            self._evalb_param_path,
            "-e",
            str(self._evalb_num_errors_to_kill),
            gold_path,
            predicted_path,
        ]
        completed_process = run_cmd(' '.join(command))

        _correct_predicted_brackets = 0.0
        _gold_brackets = 0.0
        _predicted_brackets = 0.0

        for line in completed_process.split("\n"):
            stripped = line.strip().split()
            if len(stripped) == 12 and stripped != self._header_line:
                # This line contains results for a single tree.
                numeric_line = [float(x) for x in stripped]
                _correct_predicted_brackets += numeric_line[5]
                _gold_brackets += numeric_line[6]
                _predicted_brackets += numeric_line[7]

        shutil.rmtree(tempdir)

        self._correct_predicted_brackets += _correct_predicted_brackets
        self._gold_brackets += _gold_brackets
        self._predicted_brackets += _predicted_brackets

    def get_metric(self):
        """
        # Returns

        The average precision, recall and f1.
        """

        return F1(self._predicted_brackets, self._gold_brackets, self._correct_predicted_brackets)

    def reset(self):
        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    @staticmethod
    def compile_evalb(evalb_directory_path: str = None):
        os.system("cd {} && make && cd ../../../".format(evalb_directory_path))
        run_cmd('chmod +x ' + os.path.join(evalb_directory_path, "evalb"))

    @staticmethod
    def clean_evalb(evalb_directory_path: str = None):
        return run_cmd("rm {}".format(os.path.join(evalb_directory_path, "evalb")))

    @property
    def score(self):
        return self.get_metric().prf[-1]

    def __repr__(self) -> str:
        return str(self.get_metric())


def main():
    tree1 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    evalb_scorer = EvalbBracketingScorer()
    evalb_scorer([tree1], [tree2])
    metrics = evalb_scorer.get_metric()
    assert metrics.prf == (1.0, 1.0, 1.0)

    tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
    tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    evalb_scorer = EvalbBracketingScorer()
    evalb_scorer([tree1], [tree2])
    metrics = evalb_scorer.get_metric()
    assert metrics.prf == (0.75, 0.75, 0.75)

    tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
    tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    evalb_scorer = EvalbBracketingScorer()
    evalb_scorer([tree1, tree2], [tree2, tree2])
    metrics = evalb_scorer.get_metric()
    assert metrics.prf == (0.875, 0.875, 0.875)


if __name__ == '__main__':
    main()
