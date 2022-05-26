"""Contains definition for fuzzy network of fuzzy inference systems.

TODO Refactoring for interface and readability/documentation
"""

from __future__ import annotations  # Doc aliases
from typing import Dict, Set, Tuple, Iterable
from numpy.typing import ArrayLike

import numpy as np

from hotfis import FIS, MembGroupset, FuzzyRuleset


class _FuzzyNode(FIS):
    """Node of a FuzzyNetwork, or a FIS with branches that consist of other nodes.

    Attributes:
        name (str): Name of the ruleset.
        branches (Dict[str, FuzzyNetwork.FISNode]): Dict of named FIS nodes that
            output this node's inputs.
        network (FuzzyNetwork): The node's parent network.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, name: str, ruleset: FuzzyRuleset, network):
        # Save name and FIS
        self.name = name
        super().__init__(network.groupset, ruleset)

        # Initialize branches
        self.branches = dict()

        # Save parent network
        self.network = network

    # -------
    # Methods
    # -------

    def insert(self, new_branches: Dict[str, FuzzyRuleset]):
        """Inserts one or more rulesets as FISNodes as branches of this node.
        """
        # Validate all new node names
        self.network._validate_node_names(new_branches.keys())

        # Save new node names in network
        self.network.node_names.update(new_branches.keys())

        # Convert FISs to FuzzyNodes
        conv_branches = {name: _FuzzyNode(name, rset, self.network)
                         for name, rset in new_branches.items()}

        # Save new branches
        self.branches.update(conv_branches)

        # Update network's list of required input and output names
        self.network._update_io(conv_branches)


class FuzzyNetwork:
    """A hierarchical network of fuzzy inference systems.

    Attributes:
        roots (Dict[str, _FuzzyNode]): Root nodes that give final output.
        node_names (Set[str]): Set of all node names in network.
        input_names (Set[Tuple[str, str]): Set of input node and group names.
        output_names (Set[Tuple[str, str]): Set of output node and group names.
    """
    # -----------
    # Constructor
    # -----------

    def __init__(self, groupset: MembGroupset):
        self.groupset = groupset

        self.roots = dict()

        self.node_names: Set[str] = set()

        self.input_names: Set[Tuple[str, str]] = set()
        self.output_names: Set[Tuple[str, str]] = set()

    # --------------
    # Node Retrieval
    # --------------

    def __getitem__(self, node_name):
        """Supports subscripting for node retrieval (ex. network["node_name"]).
        """
        if node_name not in self.node_names:
            # If node not found despite being in saved node names, raise error
            err_str = f"Could not find the '{node_name}' node."
            raise KeyError(err_str)

        # Recursively check subtree of each root for the desired node
        for root in self.roots.values():
            result = self._get_node(node_name, root)

            # Return if found
            if result:
                return result

        # If node not found despite being in saved node names, raise error
        err_str = f"Despite '{node_name}' being in the network's " \
                  f"node_names, could not find the '{node_name}' node."
        raise KeyError(err_str)

    # Helpers

    def _get_node(self, node_name: str, current_node: _FuzzyNode):
        # Return if desired node. If not, return none if leaf node
        if current_node.name == node_name:
            return current_node
        elif not current_node.branches:
            return None

        # Check branches for desired node
        for branch in current_node.branches.values():
            result = self._get_node(node_name, branch)

            # Return if found
            if result:
                return result

        # If not in subtree, return none
        return None

    # -------------------------
    # Creation and Modification
    # -------------------------

    def insert(self, new_roots: Dict[str, FuzzyRuleset]):
        """Inserts one or more rulesets as FISNodes as roots in network.
        """
        # Validate all new node names (ensure no duplicates)
        self._validate_node_names(new_roots.keys())

        # Save new node names in network
        self.node_names.update(new_roots.keys())

        # Convert Rulesets to FISNodes
        conv_roots = {name: _FuzzyNode(name, rset, self)
                      for name, rset in new_roots.items()}

        # Save new roots
        self.roots.update(conv_roots)

        # Update list of required input and output names
        self._update_io(conv_roots)

    # Helpers

    def _validate_node_names(self, new_names: Iterable[str]):
        """Validates that inserted node is not already in network upon insertion.
        """
        for node_name in new_names:
            if node_name in self.node_names:
                err_str = f"Failed to insert '{node_name}' since " \
                          f"a node of the same name already exists."
                raise KeyError(err_str)

    def _update_io(self, new_nodes: Dict[str, _FuzzyNode]):
        """Updates network's list of required inputs and outputs upon insertion.
        """
        for node_name, node in new_nodes.items():
            for input_name in node.ruleset.input_names:
                self.input_names.add((node_name, input_name))
            for output_name in node.ruleset.output_names:
                self.output_names.add((node_name, output_name))

    # -------------------
    # Information Methods
    # -------------------

    def display(self) -> None:
        """Prints a network's nodes with inputs and outputs.
        """
        for root in self.roots.values():
            self._display_subtree(root)

        print()

    def req_inputs(self) -> Set[Tuple[str, str]]:
        """
        Returns a set of tuples indicating the node name and input name
        of ONLY the inputs required from the user for evaluation.

        Example:
            network.req_inputs() -> {(offence, x), (thrust_dir, x), (defense, y))}
                                 This means the offence and thrust_dir nodes need
                                 a value for x and defense needs a value for y.

            outputs = network.eval_tsk({x: x_val, y: y_val}, groupset)
        """
        addressed = set()  # Contains inputs implicitly covered by network

        for root in self.roots.values():
            self._update_addressed(addressed, root)

        # Return unaddressed inputs
        return self.input_names - addressed

    # Helpers

    def _display_subtree(self, node, offset: int = 0):
        print(f"{'    ' * offset} ------------------\n"
              f"{'    ' * offset}| {node.name}\n"
              f"{'    ' * offset}| Inputs:  {node.ruleset.input_names}\n"
              f"{'    ' * offset}| Outputs: {node.ruleset.output_names}")

        for branch in node.branches.values():
            self._display_subtree(branch, offset + 1)

    def _update_addressed(self, addressed, node):
        # For each branch of node
        for branch in node.branches.values():
            # For each output of branch
            for output in branch.ruleset.output_names:
                # If branch output corresponds to node input, save info
                if output in node.ruleset.input_names:
                    addressed.add((node.name, output))

            # Recursively call for branch
            self._update_addressed(addressed, branch)

    # ------------------
    # Evaluation Methods
    # ------------------

    def eval_membership(self, inputs: Dict[str, ArrayLike],
                        return_all: bool = False, use_tsk: bool = True,
                        num_points: int = 100) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluates the entire network and returns memberships to all functions.
        Will return every node's output if return_all is given as True.

        Uses either Takagi-Sugeno or Mamdani inference for evaluation needed
        for subsequent nodes based on whether the use_tsk parameter is True.
        """
        all_inputs = inputs.copy()
        all_outputs = dict()
        fn_type = "raw-tsk" if use_tsk else "raw-mam"

        for root in self.roots.values():
            # Recursively evaluate subtree and update inputs/outputs
            self._eval_subtree(all_inputs, num_points, fn_type,
                               root, all_outputs)

            # Save output for root
            all_outputs[root.name] = root.eval_membership(all_inputs)

        return self._get_final_outputs(all_outputs, return_all)

    def eval_mamdani(self, inputs: Dict[str, ArrayLike], return_all: bool = False,
                     num_points: int = 100) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Evaluates the entire network and returns MamdaniFunctions.
        Will return every node's output if return_all is given as True.
        """
        all_inputs = inputs.copy()
        all_outputs = dict()

        for root in self.roots.values():
            # Recursively evaluate subtree and update inputs/outputs
            self._eval_subtree(all_inputs, num_points, "mam",
                               root, all_outputs)

            # Save output for root
            all_outputs[root.name] = root.eval_mamdani(all_inputs, num_points)

        return self._get_final_outputs(all_outputs, return_all)

    def eval_tsk(self, inputs: Dict[str, ArrayLike],
                 return_all: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the entire network and returns defuzzified Takagi-Sugeno output.
        Will return every node's output if return_all is given as True.
        """
        all_inputs = inputs.copy()
        all_outputs = dict()

        for root in self.roots.values():
            # Recursively evaluate subtree and update inputs/outputs
            self._eval_subtree(all_inputs, 0, "tsk", root, all_outputs)

            # Save output for root
            all_outputs[root.name] = root.eval_tsk(all_inputs)

        return self._get_final_outputs(all_outputs, return_all)

    # Helpers

    def _eval_subtree(self, all_inputs: Dict[str, float], num_points: int,
                      fn_type: str, node: _FuzzyNode, all_outputs) -> None:
        """
        Evaluates according to given output type and returns roots' outputs.
        """
        for branch in node.branches.values():
            # Recursively evaluate subtree and update inputs/outputs
            self._eval_subtree(all_inputs, num_points, fn_type,
                               branch, all_outputs)

            # Evaluate FIS Node and save output
            if fn_type == "mam":
                new_outputs = branch.eval_mamdani(all_inputs, num_points)
            elif fn_type == "tsk":
                new_outputs = branch.eval_tsk(all_inputs)
            elif fn_type in ["raw-tsk", "raw-mam"]:
                new_outputs = branch.eval_membership(all_inputs)
            else:
                raise NameError(f"Invalid evaluation type '{fn_type}'.")

            # Add FIS outputs to all outputs. OVERWRITES IF NAME IS SAME
            all_outputs[branch.name] = new_outputs

            # Convert raw/Mamdani output to defuzzified output if necessary
            if fn_type == "mam":
                new_outputs = {n: branch.defuzz_mamdani(*mam_fn)
                               for n, mam_fn in new_outputs.items()}
            elif fn_type == "raw-tsk":
                new_outputs = branch.convert_to_tsk(new_outputs)
            elif fn_type == "raw-mam":
                new_outputs = branch.convert_to_mamdani(new_outputs, num_points)
                new_outputs = {n: branch.defuzz_mamdani(*mam_fn)
                               for n, mam_fn in new_outputs.items()}

            # Update all inputs
            all_inputs.update(new_outputs)

    def _get_final_outputs(self, all_outputs, return_all: bool):
        # Just return all outputs if that's all that's needed
        if return_all:
            return all_outputs

        # Get only outputs of roots if not returning all
        final_outputs = dict()
        for name, output in all_outputs.items():
            if name in self.roots.keys():
                final_outputs[name] = output

        return final_outputs

    # ---------------------
    # Other Mamdani Methods
    # ---------------------

    @staticmethod
    def defuzz_mamdani(domain: np.ndarray, codomain: np.ndarray) -> ArrayLike:
        """Defuzzifies Mamdani output and returns scalar value(s).

        Args:
            domain: Domain of Mamdani output.
            codomain: Output memberships corresponding to domain.

        Returns:
            Defuzzified output(s).
        """
        return FIS.defuzz_mamdani(domain, codomain)

    @staticmethod
    def plot_mamdani(domain: np.ndarray, codomain: np.ndarray):
        """Plots output of Mamdani inference.

        Can be used in conjunction with an output membership function group's
        plot method to visualize both the group and inference output.

        Args:
            domain: Domain values of output as returned by Mamdani eval.
            codomain: Output membership values as returned by Mamdani eval.
        """
        FIS.plot_mamdani(domain, codomain)
