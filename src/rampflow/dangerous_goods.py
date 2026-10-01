"""Dangerous goods release gate.

Before dangerous goods are released for a flight three independent paper trails
have to agree:

* the dangerous goods declaration (DGD) describing hazard class per consignment,
* the actual compartment stow, including segregation between incompatible classes,
* the notification to the captain (NOTOC), which must quote the document versions
  it was prepared from.

This module makes the release decision a single atomic gate over all three.
A declaration revision bumps the document version, voids any prior release and
forces the affected consignments to be re-inspected.  Release is all or nothing:
there is no item-level release, so an unaffected consignment cannot leave while
a correction is still being checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

from .models import Decision
from .timeutil import utc

# Pairs of hazard classes that may not occupy adjacent positions.  Mirrors the
# small incompatibility tables used by cargo.py and load_control.py.
DEFAULT_INCOMPATIBLE: tuple[frozenset[str], ...] = (
    frozenset({"3", "5.1"}),    # flammable liquids vs oxidizing substances
    frozenset({"4.1", "5.1"}),  # flammable solids vs oxidizing substances
    frozenset({"8", "6.1"}),    # corrosive vs toxic substances
)


@dataclass(frozen=True)
class DgItem:
    item_id: str
    un_id: str
    hazard_class: str
    packing_group: str
    weight_kg: float


@dataclass(frozen=True)
class DgDeclaration:
    flight_id: str
    version: int
    items: Mapping[str, DgItem]
    revised_at: datetime


@dataclass(frozen=True)
class DgRevision:
    declaration: DgDeclaration
    affected: frozenset[str]


@dataclass(frozen=True)
class DgPosition:
    """A stow position.  ``allowed_classes`` is explicit: an empty set means the
    position may not carry dangerous goods at all."""
    position_id: str
    deck: str
    zone: str
    allowed_classes: frozenset[str] = frozenset()
    adjacent: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StowEntry:
    item_id: str
    position_id: str
    declared_version: int


@dataclass(frozen=True)
class CaptainNotice:
    notice_id: str
    declaration_version: int
    stow_revision: int
    issued_at: datetime
    acknowledged: bool = False

    def acknowledge(self) -> "CaptainNotice":
        return CaptainNotice(
            self.notice_id, self.declaration_version, self.stow_revision,
            self.issued_at, True,
        )


@dataclass(frozen=True)
class DgEvaluation:
    decisions: tuple[Decision, ...]

    @property
    def accepted(self) -> bool:
        return all(d.accepted for d in self.decisions)

    @property
    def failures(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if not d.accepted)


class _FlightState:
    def __init__(self) -> None:
        self.declaration: DgDeclaration | None = None
        self.stow: dict[str, StowEntry] = {}
        self.stow_revision = 0
        self.notice: CaptainNotice | None = None
        self.affected: frozenset[str] = frozenset()
        self.released = False
        self.released_version: int | None = None


class DgReleaseControl:
    """Owns declaration versions, actual stow, captain notification and the
    atomic release decision for dangerous goods on each flight."""

    def __init__(
        self,
        positions: Iterable[DgPosition],
        *,
        incompatible_pairs: Iterable[frozenset[str]] = DEFAULT_INCOMPATIBLE,
        accepted_classes: frozenset[str] | None = None,
    ) -> None:
        self.positions = {p.position_id: p for p in positions}
        self.incompatible = tuple(incompatible_pairs)
        # None means the carrier accepts every declared class; otherwise only
        # classes in the set may be carried.
        self.accepted_classes = accepted_classes
        self._flights: dict[str, _FlightState] = {}

    def _state(self, flight_id: str) -> _FlightState:
        return self._flights.setdefault(flight_id, _FlightState())

    # -- declaration --------------------------------------------------------

    def declare(self, flight_id: str, items: Iterable[DgItem], at: datetime) -> DgDeclaration:
        state = self._state(flight_id)
        if state.declaration is not None:
            raise RuntimeError(f"flight {flight_id} already declared; use revise()")
        indexed = self._index_items(items)
        declaration = DgDeclaration(flight_id, 1, indexed, utc(at))
        state.declaration = declaration
        return declaration

    def revise(self, flight_id: str, items: Iterable[DgItem], at: datetime) -> DgRevision:
        """Apply a corrected declaration.

        Consignments added, removed or changed in any attribute are reported as
        affected.  The new version voids the release and the captain notice
        (which quoted the old document version); affected stow entries keep
        their old stamp until they are re-inspected.
        """
        state = self._state(flight_id)
        if state.declaration is None:
            raise RuntimeError(f"flight {flight_id} has no declaration; use declare()")
        previous = state.declaration
        indexed = self._index_items(items)
        affected = set(indexed.keys()) | set(previous.items.keys())
        affected.difference_update(
            iid for iid in indexed.keys() & previous.items.keys()
            if indexed[iid] == previous.items[iid]
        )
        declaration = DgDeclaration(flight_id, previous.version + 1, indexed, utc(at))
        for removed in previous.items.keys() - indexed.keys():
            state.stow.pop(removed, None)
        state.declaration = declaration
        state.affected = frozenset(affected)
        state.notice = None
        state.released = False
        state.released_version = None
        return DgRevision(declaration, frozenset(affected))

    @staticmethod
    def _index_items(items: Iterable[DgItem]) -> dict[str, DgItem]:
        out: dict[str, DgItem] = {}
        for item in items:
            if item.item_id in out:
                raise ValueError(f"duplicate dangerous item {item.item_id}")
            if item.weight_kg <= 0:
                raise ValueError(f"invalid weight for {item.item_id}")
            out[item.item_id] = item
        return out

    # -- actual stow --------------------------------------------------------

    def stow(self, flight_id: str, item_id: str, position_id: str) -> Decision:
        """Record where a consignment was physically loaded."""
        state = self._require_declaration(flight_id)
        if state.released:
            return Decision(False, "dg-already-released",
                            "flight released; a declaration revision is required to change stow",
                            {"flight": flight_id})
        if item_id not in state.declaration.items:
            return Decision(False, "dg-reference", "stow references undeclared dangerous item",
                            {"item": item_id})
        if position_id not in self.positions:
            return Decision(False, "dg-reference", "stow references unknown position",
                            {"position": position_id})
        occupant = next((other for other in state.stow.values()
                         if other.position_id == position_id and other.item_id != item_id), None)
        if occupant is not None:
            return Decision(False, "dg-position-occupied",
                            "position already occupied by another dangerous consignment",
                            {"position": position_id, "item": occupant.item_id})
        state.stow[item_id] = StowEntry(item_id, position_id, state.declaration.version)
        state.stow_revision += 1
        state.released = False
        state.released_version = None
        return Decision(True, "ok", "stow recorded",
                        {"item": item_id, "position": position_id,
                         "stow_revision": state.stow_revision})

    def reinspect(self, flight_id: str, item_id: str) -> Decision:
        """Re-check an affected consignment after a declaration correction
        without moving it.  Restamps the entry against the current version."""
        state = self._require_declaration(flight_id)
        entry = state.stow.get(item_id)
        if entry is None:
            return Decision(False, "dg-reference", "no stow to reinspect for item",
                            {"item": item_id})
        state.stow[item_id] = StowEntry(item_id, entry.position_id, state.declaration.version)
        return Decision(True, "ok", "affected stow re-inspected",
                        {"item": item_id, "declaration_version": state.declaration.version})

    # -- captain notification ----------------------------------------------

    def notify_captain(self, flight_id: str, notice_id: str, at: datetime) -> Decision:
        state = self._require_declaration(flight_id)
        missing = tuple(sorted(set(state.declaration.items) - set(state.stow)))
        if missing:
            return Decision(False, "dg-stow-incomplete",
                            "cannot notify captain before every consignment is stowed",
                            {"missing": missing})
        state.notice = CaptainNotice(
            notice_id, state.declaration.version, state.stow_revision, utc(at),
        )
        state.released = False
        state.released_version = None
        return Decision(True, "ok", "captain notification issued",
                        {"notice": notice_id,
                         "declaration_version": state.declaration.version,
                         "stow_revision": state.stow_revision})

    def acknowledge_notice(self, flight_id: str) -> Decision:
        state = self._require_declaration(flight_id)
        if state.notice is None:
            return Decision(False, "dg-notice-missing", "no captain notification to acknowledge")
        state.notice = state.notice.acknowledge()
        return Decision(True, "ok", "captain notification acknowledged")

    # -- the gate -----------------------------------------------------------

    def evaluate(self, flight_id: str) -> DgEvaluation:
        state = self._flights.get(flight_id)
        if state is None or state.declaration is None:
            return DgEvaluation((Decision(False, "dg-declaration-missing",
                                          "no dangerous goods declaration on file",
                                          {"flight": flight_id}),))
        declaration = state.declaration
        checks: list[Decision] = []

        rejected_classes = tuple(sorted(
            item.item_id for item in declaration.items.values()
            if self.accepted_classes is not None and item.hazard_class not in self.accepted_classes
        ))
        checks.append(
            Decision(False, "dg-class-not-accepted",
                     "declared hazard class not accepted for carriage",
                     {"items": rejected_classes}) if rejected_classes
            else Decision(True, "ok", "all declared hazard classes accepted")
        )

        missing = tuple(sorted(set(declaration.items) - set(state.stow)))
        checks.append(
            Decision(False, "dg-stow-incomplete",
                     "declared consignments missing from actual stow", {"missing": missing})
            if missing else Decision(True, "ok", "every consignment is stowed")
        )

        misplaced: list[dict[str, str]] = []
        for item_id, entry in state.stow.items():
            item = declaration.items.get(item_id)
            position = self.positions.get(entry.position_id)
            if item is None or position is None:
                continue
            if item.hazard_class not in position.allowed_classes:
                misplaced.append({"item": item_id, "position": entry.position_id,
                                  "class": item.hazard_class})
        checks.append(
            Decision(False, "dg-position-not-authorized",
                     "dangerous goods in a compartment not authorized for their class",
                     {"items": tuple(misplaced)}) if misplaced
            else Decision(True, "ok", "all compartments authorized for the stowed classes")
        )

        conflicts: list[dict[str, str]] = []
        entries = list(state.stow.items())
        for i, (a_id, a_entry) in enumerate(entries):
            a = declaration.items.get(a_id)
            a_pos = self.positions.get(a_entry.position_id)
            if a is None or a_pos is None:
                continue
            for b_id, b_entry in entries[i + 1:]:
                b = declaration.items.get(b_id)
                b_pos = self.positions.get(b_entry.position_id)
                if b is None or b_pos is None:
                    continue
                adjacent = (b_entry.position_id in a_pos.adjacent
                            or a_entry.position_id in b_pos.adjacent)
                if adjacent and frozenset({a.hazard_class, b.hazard_class}) in self.incompatible:
                    conflicts.append({"item": a_id, "other": b_id,
                                      "positions": (a_entry.position_id, b_entry.position_id)})
        checks.append(
            Decision(False, "dg-segregation",
                     "incompatible dangerous goods in adjacent compartments",
                     {"conflicts": tuple(conflicts)}) if conflicts
            else Decision(True, "ok", "segregation between incompatible classes holds")
        )

        stale = tuple(sorted(
            item_id for item_id in state.affected
            if (entry := state.stow.get(item_id)) is not None
            and entry.declared_version != declaration.version
        ))
        checks.append(
            Decision(False, "dg-affected-not-rechecked",
                     "loads affected by the declaration correction are not re-inspected",
                     {"items": stale}) if stale
            else Decision(True, "ok", "all affected loads re-inspected")
        )

        if state.notice is None:
            checks.append(Decision(False, "dg-notice-missing",
                                   "captain has not been notified for this declaration"))
        else:
            current = (state.notice.declaration_version == declaration.version
                       and state.notice.stow_revision == state.stow_revision)
            checks.append(
                Decision(False, "dg-document-version",
                         "captain notification quotes superseded document versions",
                         {"notice_declaration": state.notice.declaration_version,
                          "current_declaration": declaration.version,
                          "notice_stow": state.notice.stow_revision,
                          "current_stow": state.stow_revision})
                if not current else Decision(True, "ok", "notification matches current document versions")
            )
            checks.append(
                Decision(False, "dg-notice-unacknowledged",
                         "captain notification not acknowledged")
                if not state.notice.acknowledged
                else Decision(True, "ok", "captain notification acknowledged")
            )

        return DgEvaluation(tuple(checks))

    def release(self, flight_id: str, at: datetime) -> Decision:
        """Atomically release the whole dangerous goods consignment.

        Every check must pass; failures are returned together and no partial
        release is possible.  A correction afterwards voids the release and the
        full gate has to be cleared again.
        """
        state = self._flights.get(flight_id)
        if state is None or state.declaration is None:
            return Decision(False, "dg-declaration-missing",
                            "no dangerous goods declaration on file", {"flight": flight_id})
        evaluation = self.evaluate(flight_id)
        if not evaluation.accepted:
            return Decision(
                False, "dg-release-blocked",
                "dangerous goods checks incomplete; consignment cannot be released",
                {"failed": tuple(d.code for d in evaluation.failures),
                 "details": tuple(d.details for d in evaluation.failures)},
            )
        if state.released:
            return Decision(True, "ok", "dangerous goods consignment already released",
                            {"declaration_version": state.released_version})
        state.released = True
        state.released_version = state.declaration.version
        return Decision(True, "ok", "dangerous goods consignment released",
                        {"declaration_version": state.declaration.version,
                         "stow_revision": state.stow_revision,
                         "items": tuple(sorted(state.stow))})

    def is_released(self, flight_id: str) -> bool:
        state = self._flights.get(flight_id)
        return bool(state and state.released)

    def _require_declaration(self, flight_id: str) -> _FlightState:
        state = self._state(flight_id)
        if state.declaration is None:
            raise RuntimeError(f"flight {flight_id} has no dangerous goods declaration")
        return state
