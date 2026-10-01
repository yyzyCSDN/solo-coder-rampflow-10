from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable,Mapping
from .models import Decision

# Simplified package segregation in the spirit of IATA DGR table 9.3.A: paired
# hazard classes must not be loaded in the same or in adjacent compartments.
# Operators may supply their own table to match their approved programme.
DEFAULT_SEGREGATION:tuple[frozenset[str],...]=(
    frozenset({"1","2.1"}),frozenset({"1","3"}),frozenset({"1","4.1"}),frozenset({"1","5.1"}),
    frozenset({"2.1","5.1"}),frozenset({"3","5.1"}),
    frozenset({"4.1","5.1"}),frozenset({"4.2","5.1"}),frozenset({"4.3","5.1"}),
    frozenset({"4.3","8"}),
)

class ReleaseState(str,Enum):
    PENDING="pending"
    RELEASED="released"
    REVOKED="revoked"

@dataclass(frozen=True)
class DeclarationLine:
    line_id:str
    un_number:str
    hazard_class:str
    quantity:float
    packing_group:str=""
    proper_name:str=""
    def __post_init__(self):
        if not self.line_id or not self.un_number or not self.hazard_class:
            raise ValueError("declaration line requires id, UN number and hazard class")
        if self.quantity<=0:raise ValueError("positive quantity required")

@dataclass(frozen=True)
class Declaration:
    declaration_id:str
    flight_id:str
    version:int
    lines:tuple[DeclarationLine,...]
    def __post_init__(self):
        if self.version<1:raise ValueError("declaration version starts at 1")
        ids=[l.line_id for l in self.lines]
        if len(set(ids))!=len(ids):raise ValueError("duplicate declaration line")

@dataclass(frozen=True)
class HoldCompartment:
    compartment_id:str
    adjacent:frozenset[str]=frozenset()
    forbidden_classes:frozenset[str]=frozenset()

@dataclass(frozen=True)
class Placement:
    line_id:str
    compartment:str
    as_loaded_class:str=""

@dataclass(frozen=True)
class NotocEntry:
    line_id:str
    un_number:str
    hazard_class:str
    compartment:str
    quantity:float

@dataclass(frozen=True)
class Notoc:
    flight_id:str
    declaration_version:int
    entries:tuple[NotocEntry,...]

class DangerousGoodsReleaseControl:
    """Binds declaration, hold segregation and captain notification into one gate.

    The three tracks used to be confirmed separately and reconciled by hand.
    This control keeps them together per flight: release() verifies cargo
    category, actual compartment and document version in a single pass and
    either releases the whole consignment or nothing.  Any amendment or load
    change invalidates a previous release and reports the affected loading
    that must be re-checked.
    """

    def __init__(self,flight_id:str,compartments:Iterable[HoldCompartment],segregation:Iterable[frozenset[str]]|None=None):
        self.flight_id=flight_id
        self.compartments={c.compartment_id:c for c in compartments}
        if not self.compartments:raise ValueError("compartments required")
        self.segregation=DEFAULT_SEGREGATION if segregation is None else tuple(frozenset(p) for p in segregation)
        self._declaration:Declaration|None=None
        self._placements:dict[str,Placement]={}
        self._notoc:Notoc|None=None
        self._state=ReleaseState.PENDING
        self._affected:frozenset[str]=frozenset()

    @property
    def state(self)->ReleaseState:return self._state
    @property
    def affected_lines(self)->frozenset[str]:return self._affected
    @property
    def declaration(self)->Declaration|None:return self._declaration
    @property
    def notoc(self)->Notoc|None:return self._notoc
    @property
    def placements(self)->Mapping[str,Placement]:return dict(self._placements)

    def _invalidate(self,affected:Iterable[str])->None:
        if self._state==ReleaseState.RELEASED:self._state=ReleaseState.REVOKED
        self._affected=frozenset(affected)

    def _co_loaded(self,compartment:str)->set[str]:
        comp=self.compartments.get(compartment)
        zone={compartment}|(set(comp.adjacent) if comp else set())
        return {lid for lid,p in self._placements.items() if p.compartment in zone}

    def _require_declaration(self)->Declaration:
        if self._declaration is None:raise RuntimeError("no declaration registered")
        return self._declaration

    def _build_notoc(self)->Notoc:
        d=self._require_declaration()
        entries=tuple(sorted(
            (NotocEntry(l.line_id,l.un_number,l.hazard_class,self._placements[l.line_id].compartment,l.quantity) for l in d.lines),
            key=lambda e:e.line_id))
        return Notoc(self.flight_id,d.version,entries)

    def register_declaration(self,declaration:Declaration)->None:
        if declaration.flight_id!=self.flight_id:raise ValueError("declaration belongs to another flight")
        if self._declaration is not None:raise ValueError("declaration already registered; amend it instead")
        self._declaration=declaration
        self._invalidate(l.line_id for l in declaration.lines)

    def amend_declaration(self,declaration:Declaration)->frozenset[str]:
        current=self._require_declaration()
        if declaration.declaration_id!=current.declaration_id or declaration.flight_id!=self.flight_id:
            raise ValueError("amendment must continue the registered declaration")
        if declaration.version!=current.version+1:raise ValueError("amendment must bump the declaration version by one")
        old={l.line_id:l for l in current.lines}
        new={l.line_id:l for l in declaration.lines}
        changed={lid for lid in old.keys()&new.keys() if old[lid]!=new[lid]}|(old.keys()^new.keys())
        affected=set(changed)
        for lid in changed:
            p=self._placements.get(lid)
            if p is not None:affected|=self._co_loaded(p.compartment)
        self._declaration=declaration
        self._invalidate(affected)
        return self._affected

    def record_placement(self,placement:Placement)->None:
        if placement.compartment not in self.compartments:raise ValueError("unknown compartment")
        self._placements[placement.line_id]=placement
        self._invalidate(self._co_loaded(placement.compartment)|{placement.line_id})

    def remove_placement(self,line_id:str)->None:
        p=self._placements.pop(line_id,None)
        if p is not None:self._invalidate(self._co_loaded(p.compartment)|{line_id})

    def issue_notoc(self)->Notoc:
        d=self._require_declaration()
        missing=sorted(l.line_id for l in d.lines if l.line_id not in self._placements)
        if missing:raise RuntimeError(f"declared lines not loaded: {missing}")
        notoc=self._build_notoc()
        if notoc!=self._notoc:
            self._notoc=notoc
            self._invalidate(l.line_id for l in d.lines)
        return notoc

    def verify(self)->Decision:
        d=self._declaration
        if d is None:return Decision(False,"declaration-missing","no dangerous goods declaration registered")
        declared={l.line_id:l for l in d.lines}
        for lid in self._placements:
            if lid not in declared:return Decision(False,"undeclared-load","loaded item has no declaration line",{"line":lid})
        for lid in declared:
            if lid not in self._placements:return Decision(False,"load-incomplete","declared line has no recorded placement",{"line":lid})
        for lid,l in declared.items():
            p=self._placements[lid]
            if not p.as_loaded_class:return Decision(False,"category-unchecked","loaded hazard class not yet confirmed against the declaration",{"line":lid})
            if p.as_loaded_class!=l.hazard_class:return Decision(False,"category-mismatch","loaded hazard class differs from the declaration",{"line":lid,"declared":l.hazard_class,"loaded":p.as_loaded_class})
        for lid,l in declared.items():
            p=self._placements[lid]
            if l.hazard_class in self.compartments[p.compartment].forbidden_classes:
                return Decision(False,"compartment-restricted","hazard class not permitted in the actual compartment",{"line":lid,"compartment":p.compartment})
        placed=[(lid,self._placements[lid].compartment) for lid in declared]
        for i,(a,ca) in enumerate(placed):
            for b,cb in placed[i+1:]:
                if ca!=cb and cb not in self.compartments[ca].adjacent and ca not in self.compartments[cb].adjacent:continue
                if frozenset({declared[a].hazard_class,declared[b].hazard_class}) in self.segregation:
                    return Decision(False,"segregation","incompatible hazard classes in the same or adjacent compartments",{"lines":(a,b),"compartments":(ca,cb)})
        if self._notoc is None:return Decision(False,"notoc-missing","captain notification has not been issued")
        if self._notoc.declaration_version!=d.version:
            return Decision(False,"document-version","captain notification reflects a superseded declaration version",{"notoc_version":self._notoc.declaration_version,"declaration_version":d.version})
        if self._notoc.entries!=self._build_notoc().entries:
            return Decision(False,"document-content","captain notification does not match the current declaration and placements")
        return Decision(True,"ok","category, compartments and document version verified in one pass",{"lines":len(declared),"declaration_version":d.version})

    def release(self)->Decision:
        d=self.verify()
        if not d.accepted:return d
        self._state=ReleaseState.RELEASED
        self._affected=frozenset()
        return Decision(True,"ok","dangerous goods consignment released",{"flight":self.flight_id,"lines":len(self._declaration.lines),"declaration_version":self._declaration.version})
