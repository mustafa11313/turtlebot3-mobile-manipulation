"""Behaviour tree of the fetch & deliver mission.

FetchAndDeliver (Sequence, memory)
├── WaitForSystem
├── Phase:SEARCH ... AcquireObject  (Retry x max_pick_attempts)
│   └── Sequence
│       ├── FindTarget (Selector): TargetKnown? | SearchForTarget
│       ├── Phase:APPROACH  ApproachTarget        (Nav2, alternative standoff on failure)
│       ├── Phase:ALIGN     AlignWithTarget       (visual servo + shape verification)
│       └── Phase:PICK      PickTarget            (MoveIt + honest attach)
│                           on failure: BackUp (FailureIsFailure wrapper), retry
├── BackAwayFromTable  (the table is below the LiDAR: reverse before turning)
├── MarkPickedTable    (put that table on the costmaps)
├── Phase:DELIVER  LocateDropoff (Selector): DropoffKnown? | LookForMarker | UseConfiguredDropoff
│                  NavigateToDropoff
├── Phase:PLACE    FineApproachDropoff (closed-loop base nudge), PlaceAtDropoff
└── Phase:RETURN   BackAwayFromObject, MarkDropoff, FaceHome, ReturnHome
"""

import math

import py_trees
from py_trees.common import Status

from . import behaviours as bh


class FailAfter(py_trees.decorators.Decorator):
    """Run the child (a recovery action) but always report FAILURE afterwards."""

    def update(self):
        if self.decorated.status == Status.RUNNING:
            return Status.RUNNING
        return Status.FAILURE


def build_tree(io, cfg) -> py_trees.behaviour.Behaviour:
    standoff_nav = cfg['approach_standoff']
    standoff_align = cfg['grasp_standoff']

    find_target = py_trees.composites.Selector('FindTarget', memory=False, children=[
        bh.TargetKnown(io),
        py_trees.composites.Sequence('Search', memory=True, children=[
            bh.SetPhase(io, 'SEARCH'),
            bh.SearchForTarget(io, cfg['search_waypoints'], cfg['search_strategy']),
        ]),
    ])
    attempt = py_trees.composites.Sequence('PickAttempt', memory=True, children=[
        find_target,
        bh.SetPhase(io, 'APPROACH'),
        bh.ApproachTarget(io, standoff_nav, standoff_align),
        bh.SetPhase(io, 'ALIGN'),
        bh.AlignWithTarget(io, standoff_align),
        bh.SetPhase(io, 'PICK'),
        bh.PickTarget(io),
    ])
    attempt_with_recovery = py_trees.composites.Selector('PickOrRecover', memory=True, children=[
        attempt,
        FailAfter('RecoverAndRetry', bh.BackUpBehaviour('BackOff', io, 0.30)),
    ])
    acquire = py_trees.decorators.Retry('AcquireObject', attempt_with_recovery,
                                        cfg['max_pick_attempts'])

    marker_view = cfg['marker_viewpoint']
    locate_dropoff = py_trees.composites.Selector('LocateDropoff', memory=True, children=[
        bh.DropoffKnown(io, cfg['marker_offset']),
        py_trees.composites.Sequence('LookForMarker', memory=True, children=[
            bh.NavigateTo('GoToMarkerViewpoint', io, lambda: marker_view, retries=2,
                          abort_fn=lambda: io.ctx.marker_map is not None),
            bh.SpinInPlace('ScanForMarker', io, angle=math.pi / 2,
                           abort_fn=lambda: io.ctx.marker_map is not None),
            bh.DropoffKnown(io, cfg['marker_offset']),
        ]),
        bh.UseConfiguredDropoff(io, cfg['dropoff_place']),
    ])
    deliver = py_trees.composites.Sequence('Deliver', memory=True, children=[
        bh.SetPhase(io, 'DELIVER'),
        locate_dropoff,
        py_trees.decorators.Retry('NavigateToDropoff', bh.NavigateTo(
            'NavigateToDropoff', io, lambda: bh.dropoff_standoff(io, cfg['place_reach']), retries=2), 2),
        bh.SetPhase(io, 'PLACE'),
        bh.FineApproachPoint('FineApproachDropoff', io,
                             lambda: bh.current_dropoff(io, cfg['marker_offset']), cfg['place_reach']),
        py_trees.decorators.Retry('Place', bh.PlaceAtDropoff(io), 2),
    ])
    root = py_trees.composites.Sequence('FetchAndDeliver', memory=True, children=[
        bh.WaitForSystem(io),
        acquire,
        # The object's table is below the LiDAR plane, so Nav2 does not know it is
        # there: reverse clear of it before turning towards the drop-off.
        bh.BackUpBehaviour('BackAwayFromTable', io, 0.30),
        bh.MarkPickedTable(io),
        deliver,
        bh.SetPhase(io, 'RETURN'),
        # The delivered object is on the floor in front: reverse straight away from
        # it, put it on the costmaps, and face home before Nav2 takes over.
        bh.BackUpBehaviour('BackAwayFromObject', io, 0.35),
        bh.MarkDropoff(io),
        bh.TurnTowards('FaceHome', io, lambda: io.ctx.home),
        bh.NavigateTo('ReturnHome', io, lambda: io.ctx.home, retries=2),
    ])
    return root
