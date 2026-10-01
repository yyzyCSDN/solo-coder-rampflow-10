import unittest
from datetime import datetime, timezone

from rampflow.dangerous_goods import (
    DgItem,
    DgPosition,
    DgReleaseControl,
)
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
AT = datetime(2026, 9, 30, 8, tzinfo=UTC)


def item(item_id, hazard_class="9", weight_kg=100.0, un_id="UN0000", packing_group="III"):
    return DgItem(item_id, un_id, hazard_class, packing_group, weight_kg)


def positions():
    return [
        DgPosition("F1", "lower", "fwd", frozenset({"3", "5.1", "9"}), frozenset({"F2"})),
        DgPosition("F2", "lower", "fwd", frozenset({"3", "5.1", "9"}), frozenset({"F1"})),
        DgPosition("A1", "lower", "aft", frozenset({"8", "6.1", "9"}), frozenset({"A2"})),
        DgPosition("A2", "lower", "aft", frozenset({"8", "6.1", "9"}), frozenset({"A1"})),
    ]


class DangerousGoodsGateTests(unittest.TestCase):
    def prepare(self, control, flight="CA123", items=()):
        items = items or (item("DG1"), item("DG2", "3"))
        control.declare(flight, items, AT)
        for index, it in enumerate(items):
            position = [p.position_id for p in positions()
                        if it.hazard_class in p.allowed_classes][index]
            self.assertTrue(control.stow(flight, it.item_id, position).accepted)
        control.notify_captain(flight, "NOTOC-1", AT)
        control.acknowledge_notice(flight)

    def test_full_gate_passes_and_releases_whole_consignment(self):
        control = DgReleaseControl(positions())
        self.prepare(control)

        evaluation = control.evaluate("CA123")
        self.assertTrue(evaluation.accepted, evaluation.failures)

        decision = control.release("CA123", AT)
        self.assertTrue(decision.accepted, decision)
        self.assertEqual(decision.details["items"], ("DG1", "DG2"))
        self.assertTrue(control.is_released("CA123"))

    def test_release_is_idempotent_after_success(self):
        control = DgReleaseControl(positions())
        self.prepare(control)
        first = control.release("CA123", AT)
        second = control.release("CA123", AT)
        self.assertTrue(first.accepted and second.accepted)
        self.assertEqual(second.code, "ok")

    def test_segregation_conflict_blocks_release(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("OX", "5.1"), item("FL", "3")), AT)
        self.assertTrue(control.stow("CA123", "OX", "F1").accepted)
        self.assertTrue(control.stow("CA123", "FL", "F2").accepted)
        control.notify_captain("CA123", "NOTOC-1", AT)
        control.acknowledge_notice("CA123")

        decision = control.release("CA123", AT)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "dg-release-blocked")
        self.assertIn("dg-segregation", decision.details["failed"])
        self.assertFalse(control.is_released("CA123"))

    def test_unauthorized_compartment_blocks_release(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("DG1", "3"),), AT)
        self.assertTrue(control.stow("CA123", "DG1", "A1").accepted)
        control.notify_captain("CA123", "NOTOC-1", AT)
        control.acknowledge_notice("CA123")

        decision = control.release("CA123", AT)
        self.assertFalse(decision.accepted)
        self.assertIn("dg-position-not-authorized", decision.details["failed"])

    def test_missing_stow_means_no_partial_release(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("DG1"), item("DG2")), AT)
        self.assertTrue(control.stow("CA123", "DG1", "F1").accepted)
        control.notify_captain("CA123", "NOTOC-1", AT)
        control.acknowledge_notice("CA123")

        decision = control.release("CA123", AT)
        self.assertFalse(decision.accepted)
        self.assertIn("dg-stow-incomplete", decision.details["failed"])
        # the correctly stowed consignment must not escape on its own
        self.assertFalse(control.is_released("CA123"))

    def test_undeclared_item_cannot_be_stowed(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("DG1"),), AT)
        decision = control.stow("CA123", "GHOST", "F1")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "dg-reference")

    def test_declaration_revision_requires_full_revalidation(self):
        control = DgReleaseControl(positions())
        self.prepare(control)
        self.assertTrue(control.release("CA123", AT).accepted)

        revision = control.revise(
            "CA123", (item("DG1", "9", 120.0), item("DG2", "3")), AT
        )
        self.assertEqual(revision.declaration.version, 2)
        # DG1's declared weight changed; DG2 untouched
        self.assertEqual(revision.affected, frozenset({"DG1"}))
        self.assertFalse(control.is_released("CA123"))

        blocked = control.release("CA123", AT)
        self.assertFalse(blocked.accepted)
        failed = set(blocked.details["failed"])
        self.assertIn("dg-affected-not-rechecked", failed)
        self.assertIn("dg-notice-missing", failed)

        # re-inspect against the corrected document and re-issue the NOTOC
        self.assertTrue(control.reinspect("CA123", "DG1").accepted)
        control.notify_captain("CA123", "NOTOC-2", AT)
        control.acknowledge_notice("CA123")
        decision = control.release("CA123", AT)
        self.assertTrue(decision.accepted, decision)
        self.assertEqual(decision.details["declaration_version"], 2)

    def test_revision_removing_a_consignment_drops_its_stow(self):
        control = DgReleaseControl(positions())
        self.prepare(control)
        control.revise("CA123", (item("DG1"),), AT)
        evaluation = control.evaluate("CA123")
        codes = {d.code for d in evaluation.decisions}
        self.assertNotIn("dg-stow-incomplete", codes)
        self.assertIn("dg-notice-missing", codes)

    def test_added_consignment_must_be_stowed_and_noticed(self):
        control = DgReleaseControl(positions())
        self.prepare(control, items=(item("DG1"),))
        control.revise("CA123", (item("DG1"), item("DG3", "3")), AT)

        # the new consignment does not exist on the aircraft yet
        blocked = control.release("CA123", AT)
        self.assertFalse(blocked.accepted)
        self.assertIn("dg-stow-incomplete", blocked.details["failed"])
        self.assertIn("dg-notice-missing", blocked.details["failed"])

        # stowing is the physical inspection, so it stamps the current version
        self.assertTrue(control.stow("CA123", "DG3", "F2").accepted)
        control.notify_captain("CA123", "NOTOC-2", AT)
        control.acknowledge_notice("CA123")
        self.assertTrue(control.evaluate("CA123").accepted)

    def test_class_change_for_stowed_item_forces_reinspection(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("DG1", "9"), item("DG2", "9")), AT)
        control.stow("CA123", "DG1", "F1")
        control.stow("CA123", "DG2", "F2")
        control.notify_captain("CA123", "NOTOC-1", AT)
        control.acknowledge_notice("CA123")
        self.assertTrue(control.evaluate("CA123").accepted)

        # DG1 changes class but remains in its old compartment F1
        control.revise("CA123", (item("DG1", "5.1"), item("DG2", "9")), AT)
        blocked = control.release("CA123", AT)
        self.assertFalse(blocked.accepted)
        self.assertIn("dg-affected-not-rechecked", blocked.details["failed"])
        self.assertTrue(control.reinspect("CA123", "DG1").accepted)

    def test_stow_change_after_notice_invalidates_versions(self):
        control = DgReleaseControl(positions())
        self.prepare(control)
        # move DG1 to another authorized compartment after the NOTOC went out
        self.assertTrue(control.stow("CA123", "DG1", "A1").accepted)

        evaluation = control.evaluate("CA123")
        codes = {d.code for d in evaluation.decisions}
        self.assertIn("dg-document-version", codes)
        self.assertFalse(evaluation.accepted)

        control.notify_captain("CA123", "NOTOC-2", AT)
        control.acknowledge_notice("CA123")
        self.assertTrue(control.evaluate("CA123").accepted)

    def test_notice_must_be_acknowledged(self):
        control = DgReleaseControl(positions())
        control.declare("CA123", (item("DG1"),), AT)
        control.stow("CA123", "DG1", "F1")
        control.notify_captain("CA123", "NOTOC-1", AT)
        decision = control.release("CA123", AT)
        self.assertFalse(decision.accepted)
        self.assertIn("dg-notice-unacknowledged", decision.details["failed"])

    def test_no_release_without_declaration(self):
        control = DgReleaseControl(positions())
        decision = control.release("EMPTY", AT)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "dg-declaration-missing")

    def test_stow_locked_after_release_until_revision(self):
        control = DgReleaseControl(positions())
        self.prepare(control)
        self.assertTrue(control.release("CA123", AT).accepted)
        locked = control.stow("CA123", "DG1", "F2")
        self.assertFalse(locked.accepted)
        self.assertEqual(locked.code, "dg-already-released")


class DangerousGoodsWorkflowTests(unittest.TestCase):
    def _ready_control(self):
        dg = DgReleaseControl(positions())
        dg.declare("CA123", (item("DG1", "9"),), AT)
        dg.stow("CA123", "DG1", "F1")
        dg.notify_captain("CA123", "NOTOC-1", AT)
        dg.acknowledge_notice("CA123")
        return dg

    def test_event_published_only_after_gate_passes(self):
        dg = self._ready_control()
        workflow = RampWorkflowControl()

        receipt = workflow.release_dangerous_goods(
            "op-dg-1", "CA123", AT,
            lambda: dg.release("CA123", AT),
            requirements=dg.evaluate("CA123").decisions,
        )
        self.assertEqual(receipt.topic, "dangerous.goods.release.committed")
        self.assertEqual(workflow.stream.head, 1)
        self.assertTrue(dg.is_released("CA123"))

    def test_blocked_gate_publishes_nothing_and_is_replay_safe(self):
        dg = DgReleaseControl(positions())
        dg.declare("CA123", (item("OX", "5.1"), item("FL", "3")), AT)
        dg.stow("CA123", "OX", "F1")
        dg.stow("CA123", "FL", "F2")
        dg.notify_captain("CA123", "NOTOC-1", AT)
        dg.acknowledge_notice("CA123")
        workflow = RampWorkflowControl()

        with self.assertRaises(WorkflowRejected):
            workflow.release_dangerous_goods(
                "op-dg-2", "CA123", AT,
                lambda: dg.release("CA123", AT),
                requirements=dg.evaluate("CA123").decisions,
            )
        self.assertEqual(workflow.stream.head, 0)
        self.assertEqual(workflow.receipts, {})
        self.assertFalse(dg.is_released("CA123"))


if __name__ == "__main__":
    unittest.main()
