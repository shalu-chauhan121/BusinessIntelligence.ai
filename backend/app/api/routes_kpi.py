"""
The KPI Contract API — discovering, reviewing, overriding and approving the
authoritative KPI definitions for a dataset.

Reads are open to any signed-in user; every write is analyst-gated, because a
KPI definition is an organisational decision and the contract is what the rest
of the platform treats as true.

These endpoints declare real response models, which the analysis endpoints do
not. That is deliberate: the contract is the artefact downstream systems depend
on, so its shape belongs in the OpenAPI document rather than being discovered by
reading the code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..db.repositories import DatasetRepository, KpiContractRepository
from ..deps import current_user, require_analyst
from ..engines.observe import Timeframe, available_timeframes, slice_period
from ..kpi import service as kpi_service
from ..kpi.contract import KpiContract, KpiDefinition
from ..kpi.library import ALL_LIBRARY_KPIS
from ..kpi.resolver import KpiResolutionError, compile_kpi
from ..kpi.service import KpiContractError
from ..llm.client import get_llm
from ..models.schemas import (
    ConflictResolveRequest,
    KpiCreateRequest,
    KpiDiscoverRequest,
    KpiRejectRequest,
    KpiUpdateRequest,
)
from ..services import dataset_service
from .redact import is_analyst, redact_contract

router = APIRouter(prefix="/api/kpi", tags=["kpi contract"])


# ---------------------------------------------------------------------------
# response models
# ---------------------------------------------------------------------------
class ContractResponse(BaseModel):
    contract: KpiContract
    summary: Dict[str, Any]


class KpiResponse(BaseModel):
    kpi: KpiDefinition
    summary: Dict[str, Any]


class PreviewPoint(BaseModel):
    period: str
    value: Optional[float] = None


class PreviewResponse(BaseModel):
    kpi_id: str
    name: str
    unit: str
    granularity: str
    points: List[PreviewPoint]
    note: str = ""


class LibraryEntryResponse(BaseModel):
    id: str
    name: str
    domain: str
    definition: str
    unit: str
    concepts: List[str]
    bound: bool
    why_unavailable: str = ""


class VersionResponse(BaseModel):
    contract_id: str
    version: int
    status: str
    is_current: bool
    kpi_count: int
    created_at: str = ""
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dataset_for(user: Dict[str, Any], dataset_id: Optional[str] = None) -> Dict[str, Any]:
    repo = DatasetRepository()
    ds = repo.get(user["uid"], dataset_id) if dataset_id else repo.active(user["uid"])
    if not ds:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No dataset available. Upload a business metrics CSV on the Data page first.",
        )
    return ds


def _load(user: Dict[str, Any], dataset_id: Optional[str] = None):
    """(dataset, df, schema) with the contract deliberately NOT attached."""
    ds = _dataset_for(user, dataset_id)
    try:
        # uid omitted on purpose: the contract is what we are about to build or
        # edit, so resolving it here would be circular.
        df, schema = dataset_service.load(ds)
    except dataset_service.DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return ds, df, schema


def _editable(user: Dict[str, Any], dataset_id: Optional[str] = None) -> KpiContract:
    """
    The contract the Studio operates on: the draft under review if one exists,
    otherwise the live contract (bootstrapping one if the dataset has none).
    """
    ds, df, schema = _load(user, dataset_id)
    existing = kpi_service.load_editable(user["uid"], ds["_id"])
    if existing:
        return existing
    return kpi_service.get_or_bootstrap(user["uid"], ds, df, schema)


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KpiContractError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except KpiResolutionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _shaped(contract: KpiContract, user: Dict[str, Any]) -> ContractResponse:
    """Role-shaped contract. The definitions stay; the derivation detail may not."""
    payload = redact_contract(contract.model_dump(mode="json"), is_analyst(user))
    return ContractResponse(contract=KpiContract.model_validate(payload),
                            summary=contract.summary())


def _respond(contract: KpiContract, user: Dict[str, Any]) -> ContractResponse:
    kpi_service.save(contract)
    dataset_service.clear_cache()
    return _shaped(contract, user)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
@router.get("/contract", response_model=ContractResponse)
def get_contract(dataset_id: Optional[str] = None,
                 user: Dict[str, Any] = Depends(current_user)) -> ContractResponse:
    """
    The current contract for a dataset.

    A dataset that has never been through discovery gets a provisional contract
    generated from the general library, so this never 404s on a real dataset.
    """
    contract = _editable(user, dataset_id)
    return _shaped(contract, user)


@router.post("/contract/discover", response_model=ContractResponse)
def discover(body: KpiDiscoverRequest,
             user: Dict[str, Any] = Depends(require_analyst)) -> ContractResponse:
    """
    Profile the dataset and propose KPIs, as a new draft version.

    Previously approved and user-defined KPIs are carried forward — re-running
    discovery must never silently revert a human decision.
    """
    ds, df, schema = _load(user, body.dataset_id)
    llm = get_llm() if body.use_llm else None
    contract = _guard(kpi_service.replace_with_discovery, user["uid"], ds, df, schema, llm)
    return ContractResponse(contract=contract, summary=contract.summary())


@router.get("/contract/proposals")
def proposals(dataset_id: Optional[str] = None,
              user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """Everything awaiting a human decision, grouped the way the reviewer works."""
    contract = _editable(user, dataset_id)
    groups: Dict[str, List[KpiDefinition]] = {
        "general_library": [], "discovered_atomic": [], "discovered_derived": [],
        "llm_suggested": [], "user_defined": [],
    }
    bucket = {
        "general_library": "general_library", "dynamic_atomic": "discovered_atomic",
        "dynamic_derived": "discovered_derived", "llm_suggested": "llm_suggested",
        "user_defined": "user_defined",
    }
    for kpi in contract.kpis:
        groups[bucket.get(kpi.provenance.origin, "discovered_atomic")].append(kpi)
    analyst = is_analyst(user)
    shaped = redact_contract(contract.model_dump(mode="json"), analyst)
    by_id = {k["kpi_id"]: k for k in shaped.get("kpis", [])}
    return {
        "summary": contract.summary(),
        "groups": {k: [by_id.get(d.kpi_id, d.model_dump(mode="json")) for d in v]
                   for k, v in groups.items()},
        "conflicts": [c.model_dump(mode="json") for c in contract.conflicts],
        "unavailable": [u.model_dump(mode="json") for u in contract.unavailable],
        "rejected_candidates": shaped.get("rejected_candidates", []),
        "field_profiles": shaped.get("field_profiles", []),
        "discovery_note": contract.discovery_note,
        "screened_by": shaped.get("screened_by", ""),
        "analyst_detail_included": analyst,
    }


@router.post("/contract/approve", response_model=ContractResponse)
def approve_contract(dataset_id: Optional[str] = None,
                     user: Dict[str, Any] = Depends(require_analyst)) -> ContractResponse:
    """Make the contract authoritative. Refused while a blocking conflict stands."""
    contract = _editable(user, dataset_id)
    _guard(kpi_service.approve_contract, contract, user["uid"])
    # Save before flipping the flags: `make_current` reads the stored documents,
    # so it has to see this contract already marked approved.
    kpi_service.save(contract)
    KpiContractRepository().make_current(user["uid"], contract.dataset_id, contract.contract_id)
    dataset_service.clear_cache()
    return ContractResponse(contract=contract, summary=contract.summary())


@router.get("/contract/versions", response_model=List[VersionResponse])
def versions(dataset_id: Optional[str] = None,
             user: Dict[str, Any] = Depends(current_user)) -> List[VersionResponse]:
    ds = _dataset_for(user, dataset_id)
    docs = KpiContractRepository().versions(user["uid"], ds["_id"])
    return [
        VersionResponse(
            contract_id=d.get("_id", ""), version=int(d.get("version", 1)),
            status=d.get("status", ""), is_current=bool(d.get("is_current")),
            kpi_count=len(d.get("kpis", []) or []),
            created_at=d.get("created_at", ""), approved_at=d.get("approved_at"),
            approved_by=d.get("approved_by"),
        )
        for d in docs
    ]


# ---------------------------------------------------------------------------
# individual KPIs
# ---------------------------------------------------------------------------
@router.post("/contract/kpis", response_model=KpiResponse, status_code=status.HTTP_201_CREATED)
def create_kpi(body: KpiCreateRequest, dataset_id: Optional[str] = None,
               user: Dict[str, Any] = Depends(require_analyst)) -> KpiResponse:
    """Define a KPI the dataset cannot imply — an organisation-specific metric."""
    contract = _editable(user, dataset_id)
    kpi = _guard(kpi_service.add_user_kpi, contract,
                 body.model_dump(exclude_none=True), user["uid"])
    kpi_service.save(contract)
    return KpiResponse(kpi=kpi, summary=contract.summary())


@router.patch("/contract/kpis/{kpi_id}", response_model=KpiResponse)
def update_kpi(kpi_id: str, body: KpiUpdateRequest, dataset_id: Optional[str] = None,
               user: Dict[str, Any] = Depends(require_analyst)) -> KpiResponse:
    """Override any part of a definition. The KPI returns to `proposed`."""
    contract = _editable(user, dataset_id)
    kpi = _guard(kpi_service.update_kpi, contract, kpi_id, body.patch, user["uid"])
    kpi_service.save(contract)
    dataset_service.clear_cache()
    return KpiResponse(kpi=kpi, summary=contract.summary())


@router.delete("/contract/kpis/{kpi_id}")
def delete_kpi(kpi_id: str, dataset_id: Optional[str] = None,
               user: Dict[str, Any] = Depends(require_analyst)) -> Dict[str, Any]:
    contract = _editable(user, dataset_id)
    _guard(kpi_service.delete_kpi, contract, kpi_id)
    kpi_service.save(contract)
    dataset_service.clear_cache()
    return {"deleted": kpi_id, "summary": contract.summary()}


@router.post("/contract/kpis/{kpi_id}/approve", response_model=KpiResponse)
def approve_kpi(kpi_id: str, dataset_id: Optional[str] = None,
                user: Dict[str, Any] = Depends(require_analyst)) -> KpiResponse:
    """
    Make one KPI authoritative.

    Refused while it carries a blocking conflict, or while its granularity was
    inferred but never confirmed.
    """
    contract = _editable(user, dataset_id)
    kpi = _guard(kpi_service.approve_kpi, contract, kpi_id, user["uid"])
    kpi_service.save(contract)
    dataset_service.clear_cache()
    return KpiResponse(kpi=kpi, summary=contract.summary())


@router.post("/contract/kpis/{kpi_id}/reject", response_model=KpiResponse)
def reject_kpi(kpi_id: str, body: KpiRejectRequest, dataset_id: Optional[str] = None,
               user: Dict[str, Any] = Depends(require_analyst)) -> KpiResponse:
    contract = _editable(user, dataset_id)
    kpi = _guard(kpi_service.reject_kpi, contract, kpi_id, user["uid"], body.reason)
    kpi_service.save(contract)
    dataset_service.clear_cache()
    return KpiResponse(kpi=kpi, summary=contract.summary())


@router.post("/contract/kpis/{kpi_id}/preview", response_model=PreviewResponse)
def preview_kpi(kpi_id: str, dataset_id: Optional[str] = None,
                user: Dict[str, Any] = Depends(current_user)) -> PreviewResponse:
    """
    Compute a KPI over recent periods without approving it.

    This is what lets an analyst check that a definition produces sensible
    numbers before making it authoritative.
    """
    ds, df, schema = _load(user, dataset_id)
    contract = kpi_service.get_or_bootstrap(user["uid"], ds, df, schema)
    kpi = contract.kpi(kpi_id)
    if kpi is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"This contract has no KPI '{kpi_id}'.")
    try:
        compiled = compile_kpi(kpi)
    except KpiResolutionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    points: List[PreviewPoint] = []
    for frame in available_timeframes(df)[-8:]:
        rows = slice_period(df, Timeframe(frame["year"], frame["quarter"]))
        try:
            value = compiled.compute(rows)
        except KpiResolutionError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        points.append(PreviewPoint(period=frame["label"],
                                   value=None if value != value else round(float(value), 4)))
    return PreviewResponse(
        kpi_id=kpi.kpi_id, name=kpi.name, unit=kpi.unit,
        granularity=kpi.granularity.label, points=points,
        note=("Computed from the current definition without approving it. "
              f"Aggregated by {kpi.aggregation.rollup_policy.replace('_', ' ')}."),
    )


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------
@router.post("/contract/conflicts/{conflict_id}/resolve", response_model=ContractResponse)
def resolve_conflict(conflict_id: str, body: ConflictResolveRequest,
                     dataset_id: Optional[str] = None,
                     user: Dict[str, Any] = Depends(require_analyst)) -> ContractResponse:
    """Record a human decision on an ambiguity the system refused to settle."""
    contract = _editable(user, dataset_id)
    _guard(kpi_service.resolve_conflict, contract, conflict_id,
           body.option_id, body.rationale, user["uid"])
    # Persist the resolution before rebuilding: `rebuild_reconciled_view` reads
    # the contract back from storage to find which conflicts are resolved, so
    # saving first is what lets it see this one rather than the stale draft.
    kpi_service.save(contract)
    # A source disagreement is settled by naming an authoritative source, so the
    # reconciled view is rebuilt with that ruling applied — no re-upload.
    dataset_service.rebuild_reconciled_view(user["uid"], contract.dataset_id)
    return _respond(contract, user)


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------

@router.get("/library", response_model=List[LibraryEntryResponse])
def library(dataset_id: Optional[str] = None,
            user: Dict[str, Any] = Depends(current_user)) -> List[LibraryEntryResponse]:
    """
    Every KPI the library knows, and for this dataset, which ones bound.

    The entries that did NOT bind matter as much as the ones that did: they are
    the evidence that a KPI was left out because the data cannot support it,
    rather than because the system did not consider it.
    """
    contract = _editable(user, dataset_id)
    bound = {k.provenance.derived_from[0] for k in contract.kpis
             if k.provenance.origin == "general_library" and k.provenance.derived_from}
    unavailable = {u.library_id: u.why_unavailable for u in contract.unavailable}
    return [
        LibraryEntryResponse(
            id=entry.id, name=entry.name, domain=entry.domain,
            definition=entry.definition, unit=entry.unit,
            concepts=[c.concept for c in entry.concepts],
            bound=entry.id in bound,
            why_unavailable=unavailable.get(entry.id, ""),
        )
        for entry in ALL_LIBRARY_KPIS
    ]
