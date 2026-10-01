import unittest
from dataclasses import replace
from datetime import datetime, timezone

from rampflow.dg_release import (
    Declaration,
    DeclarationLine,
    DangerousGoodsReleaseControl,
    HoldCompartment,
    Placement,
    ReleaseState,
)
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

AT = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)


def declaration(version, lines, flight="F1", declaration_id="DGD-1"):
    return Declaration(declaration_id, flight, version, tuple(lines))


class DangerousGoodsReleaseTests(unittest.TestCase):
    def setUp(self):
        self.compartments = (
            HoldCompartment("FWD", adjacent=frozenset({"AFT"})),
            HoldCompartment("AFT", adjacent=frozenset({"FWD"})),
            HoldCompartment("BULK"),
        )
        self.paint = DeclarationLine("L1", "UN1263", "3", 25.0, "II", "Paint")
        self.misc = DeclarationLine("L2", "UN3077", "9", 40.0, "III", "Environmentally hazardous substance")
        self.oxidizer = DeclarationLine("L3", "UN1479", "5.1", 40.0, "II", "Oxidizing solid, n.o.s.")

    def gate(self, lines=None):
        g = DangerousGoodsReleaseControl("F1", self.compartments)
        g.register_declaration(declaration(1, lines if lines is not None else [self.paint, self.misc]))
        return g

    def place(self, g, line, compartment, as_loaded=None):
        g.record_placement(Placement(line.line_id, compartment, line.hazard_class if as_loaded is None else as_loaded))

    def test_one_pass_release_covers_category_hold_and_document_version(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        d = g.release()
        self.assertTrue(d.accepted, d.reason)
        self.assertEqual(g.state, ReleaseState.RELEASED)
        self.assertEqual(d.details["declaration_version"], 1)
        self.assertEqual(g.affected_lines, frozenset())

    def test_release_blocked_until_loaded_category_confirmed(self):
        g = self.gate()
        self.place(g, self.paint, "FWD", as_loaded="")
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        self.assertEqual(g.release().code, "category-unchecked")
        self.place(g, self.paint, "FWD", as_loaded="8")
        d = g.release()
        self.assertEqual(d.code, "category-mismatch")
        self.assertEqual(d.details["declared"], "3")
        self.place(g, self.paint, "FWD", as_loaded="3")
        self.assertTrue(g.release().accepted)

    def test_segregation_uses_actual_compartments(self):
        g = self.gate(lines=[self.paint, self.oxidizer])
        self.place(g, self.paint, "FWD")
        self.place(g, self.oxidizer, "AFT")
        g.issue_notoc()
        d = g.release()
        self.assertEqual(d.code, "segregation")
        self.assertEqual(g.state, ReleaseState.PENDING)
        self.place(g, self.oxidizer, "BULK")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)

    def test_forbidden_class_in_compartment(self):
        g = DangerousGoodsReleaseControl("F1", (
            HoldCompartment("FWD", forbidden_classes=frozenset({"1"})),
            HoldCompartment("BULK"),
        ))
        g.register_declaration(declaration(1, [DeclarationLine("L9", "UN0336", "1", 5.0)]))
        self.place(g, DeclarationLine("L9", "UN0336", "1", 5.0), "FWD")
        g.issue_notoc()
        self.assertEqual(g.release().code, "compartment-restricted")
        self.place(g, DeclarationLine("L9", "UN0336", "1", 5.0), "BULK")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)

    def test_amendment_revokes_release_and_scopes_recheck(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)
        affected = g.amend_declaration(declaration(2, [replace(self.paint, quantity=30.0), self.misc]))
        self.assertEqual(g.state, ReleaseState.REVOKED)
        self.assertEqual(affected, frozenset({"L1", "L2"}))
        d = g.release()
        self.assertEqual(d.code, "document-version")
        self.assertEqual(d.details["notoc_version"], 1)
        g.issue_notoc()
        d = g.release()
        self.assertTrue(d.accepted)
        self.assertEqual(g.state, ReleaseState.RELEASED)
        self.assertEqual(d.details["declaration_version"], 2)

    def test_amendment_must_continue_version_chain(self):
        g = self.gate()
        with self.assertRaises(ValueError):
            g.amend_declaration(declaration(3, [self.paint, self.misc]))
        with self.assertRaises(ValueError):
            g.amend_declaration(declaration(2, [self.paint], declaration_id="DGD-2"))
        with self.assertRaises(ValueError):
            g.amend_declaration(declaration(2, [self.paint], flight="F2"))

    def test_removed_line_must_be_unloaded_before_release(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        g.amend_declaration(declaration(2, [self.paint]))
        self.assertEqual(g.release().code, "undeclared-load")
        g.remove_placement("L2")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)

    def test_incomplete_loading_blocks_whole_consignment(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        with self.assertRaises(RuntimeError):
            g.issue_notoc()
        d = g.release()
        self.assertEqual(d.code, "load-incomplete")
        self.assertEqual(g.state, ReleaseState.PENDING)
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)

    def test_undeclared_load_blocked(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        g.record_placement(Placement("L-X", "BULK", "8"))
        g.issue_notoc()
        self.assertEqual(g.release().code, "undeclared-load")
        self.assertEqual(g.state, ReleaseState.PENDING)

    def test_stale_notoc_content_blocks_release(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        g.issue_notoc()
        self.place(g, self.misc, "BULK")
        self.assertEqual(g.release().code, "document-content")
        g.issue_notoc()
        self.assertTrue(g.release().accepted)

    def test_declaration_validation(self):
        with self.assertRaises(ValueError):
            declaration(0, [self.paint])
        with self.assertRaises(ValueError):
            declaration(1, [self.paint, replace(self.paint, un_number="UN1993")])
        with self.assertRaises(ValueError):
            DeclarationLine("L5", "", "3", 1.0)
        with self.assertRaises(ValueError):
            DeclarationLine("L5", "UN1263", "3", 0.0)
        g = self.gate()
        with self.assertRaises(ValueError):
            g.register_declaration(declaration(1, [self.paint]))

    def test_workflow_publishes_only_after_domain_release(self):
        g = self.gate()
        self.place(g, self.paint, "FWD")
        self.place(g, self.misc, "AFT")
        wf = RampWorkflowControl()
        with self.assertRaises(WorkflowRejected):
            wf.commit_dg_release("op-1", "F1", AT, g.release, requirements=(g.verify(),))
        self.assertEqual(wf.stream.head, 0)
        self.assertEqual(g.state, ReleaseState.PENDING)
        g.issue_notoc()
        receipt = wf.commit_dg_release("op-1", "F1", AT, g.release, requirements=(g.verify(),))
        self.assertEqual(receipt.event.topic, "dg.release.committed")
        self.assertEqual(g.state, ReleaseState.RELEASED)
        again = wf.commit_dg_release("op-1", "F1", AT, g.release, requirements=(g.verify(),))
        self.assertIs(again, receipt)


if __name__ == "__main__":
    unittest.main()
