"""Static MJCF asset validation for the 37-DoF RobotLab G1 policy.

The deployment policy is tied to names and semantics, not to the order in
which joints happen to appear in an XML file.  This module therefore validates
an MJCF before a simulator is started and returns the explicit XML-to-policy
index maps that a runner can use at its boundary.

Only the Python standard library is used.  In particular, validation does not
import :mod:`mujoco`, create a model, or start a simulator.  Raw ``<include>``
directives are deliberately rejected: :mod:`xml.etree.ElementTree` cannot
reproduce MuJoCo's compiler/include semantics safely.  Validate the leaf robot
MJCF directly, or pass a model that has already been flattened by a trusted
MuJoCo-aware tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from .contract import DEFAULT_CONTRACT, DeploymentContract


class MJCFAssetError(ValueError):
    """Base class for a static G1 MJCF contract violation."""


class MJCFParseError(MJCFAssetError):
    """Raised when an input is not a readable, well-formed MJCF document."""


class MJCFIncludeError(MJCFAssetError):
    """Raised when static validation encounters unresolved MJCF includes."""


def _local_tag(element: ET.Element) -> str:
    """Return an XML tag without an optional namespace prefix."""

    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_tag(child) == tag)


def _descendants(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    return (child for child in element.iter() if _local_tag(child) == tag)


@dataclass(frozen=True)
class MJCFAssetReport:
    """Validated bindings between an MJCF and the RobotLab policy contract.

    ``policy_to_xml_joint`` and ``policy_to_motor`` are indexed in policy
    order.  Their values point into ``xml_joint_names`` and
    ``motor_joint_names`` respectively.  A runner must use these maps instead
    of assuming the two files share an order.
    """

    source: str
    source_sha256: str
    xml_joint_names: tuple[str, ...]
    motor_joint_names: tuple[str, ...]
    motor_names: tuple[str | None, ...]
    policy_to_xml_joint: tuple[int, ...]
    policy_to_motor: tuple[int, ...]
    free_joint_name: str | None
    free_root_body_name: str | None
    torso_body_name: str

    @property
    def policy_joint_count(self) -> int:
        return len(self.policy_to_xml_joint)

    @property
    def policy_motor_count(self) -> int:
        return len(self.policy_to_motor)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serialisable provenance for deployment manifests."""

        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "xml_joint_names": list(self.xml_joint_names),
            "motor_joint_names": list(self.motor_joint_names),
            "motor_names": list(self.motor_names),
            "policy_to_xml_joint": list(self.policy_to_xml_joint),
            "policy_to_motor": list(self.policy_to_motor),
            "free_joint_name": self.free_joint_name,
            "free_root_body_name": self.free_root_body_name,
            "torso_body_name": self.torso_body_name,
            "policy_joint_count": self.policy_joint_count,
            "policy_motor_count": self.policy_motor_count,
        }


@dataclass(frozen=True)
class MJCFSceneBindingReport:
    """Proof that the scene actually includes the statically validated robot."""

    scene_path: str
    scene_sha256: str
    robot_path: str
    robot_sha256: str
    include_file: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scene_path": self.scene_path,
            "scene_sha256": self.scene_sha256,
            "robot_path": self.robot_path,
            "robot_sha256": self.robot_sha256,
            "include_file": self.include_file,
        }


@dataclass(frozen=True)
class G1MJCFAssetContract:
    """Static requirements for a MuJoCo asset used by this deployment stack.

    Extra passive joints and extra actuators are permitted because scene
    wrappers may contain props.  Every *policy* joint must nevertheless be a
    unique scalar hinge/slide joint with exactly one ``<motor>`` actuator.
    """

    deployment: DeploymentContract = DEFAULT_CONTRACT
    torso_body_names: tuple[str, ...] = ("torso_link",)

    def __post_init__(self) -> None:
        if not self.torso_body_names:
            raise ValueError("torso_body_names cannot be empty")
        if len(set(self.torso_body_names)) != len(self.torso_body_names):
            raise ValueError("torso_body_names must be unique")
        if any(not name for name in self.torso_body_names):
            raise ValueError("torso body aliases cannot be empty")

    def validate_file(self, path: str | Path) -> MJCFAssetReport:
        """Parse and validate a standalone or already-flattened MJCF file."""

        xml_path = Path(path).expanduser()
        try:
            payload = xml_path.read_bytes()
        except OSError as exc:
            raise MJCFParseError(f"cannot read MJCF {xml_path}: {exc}") from exc
        return self.validate_bytes(payload, source=str(xml_path.resolve()))

    def validate_xml(self, xml: str, *, source: str = "<memory>") -> MJCFAssetReport:
        """Validate MJCF text without requiring a filesystem or MuJoCo."""

        if not isinstance(xml, str):
            raise TypeError("xml must be a string")
        return self.validate_bytes(xml.encode("utf-8"), source=source)

    def validate_bytes(self, payload: bytes, *, source: str = "<memory>") -> MJCFAssetReport:
        """Validate raw XML bytes and record their exact SHA-256 digest."""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise MJCFParseError(f"malformed MJCF {source}: {exc}") from exc
        return self.validate_element(
            root,
            source=source,
            source_sha256=sha256(payload).hexdigest(),
        )

    def validate_element(
        self,
        root: ET.Element,
        *,
        source: str = "<Element>",
        source_sha256: str = "",
    ) -> MJCFAssetReport:
        """Validate an already-parsed, include-free XML element tree.

        This is the integration point for callers that flatten includes using
        a trusted MuJoCo-aware compiler.  They may pass the resulting root
        here, while preserving a digest in ``source_sha256`` if available.
        """

        if _local_tag(root) != "mujoco":
            raise MJCFAssetError(
                f"{source}: expected <mujoco> document root, got <{_local_tag(root)}>"
            )

        includes = tuple(
            include.get("file") or "<missing file attribute>"
            for include in _descendants(root, "include")
        )
        if includes:
            joined = ", ".join(repr(name) for name in includes)
            raise MJCFIncludeError(
                f"{source}: unresolved <include> directives: {joined}. "
                "Static validation cannot emulate MuJoCo include semantics; "
                "validate the leaf robot MJCF directly or pass a trusted "
                "flattened <mujoco> tree to validate_element()."
            )

        worldbodies = tuple(_children(root, "worldbody"))
        if len(worldbodies) != 1:
            raise MJCFAssetError(
                f"{source}: expected exactly one <worldbody>, found {len(worldbodies)}"
            )
        worldbody = worldbodies[0]

        bodies = tuple(_descendants(worldbody, "body"))
        body_names = tuple(body.get("name") for body in bodies)
        torso_matches = tuple(
            body for body in bodies if body.get("name") in self.torso_body_names
        )
        if len(torso_matches) != 1:
            found = sorted(name for name in body_names if name is not None)
            raise MJCFAssetError(
                f"{source}: expected exactly one torso body named one of "
                f"{list(self.torso_body_names)}, found {len(torso_matches)}; "
                f"body names={found}"
            )
        torso = torso_matches[0]

        # Only joints beneath worldbody instantiate dynamics.  Joint elements
        # in <default> sections are templates and must not enter this list.
        scalar_joints = tuple(_descendants(worldbody, "joint"))
        named_joints: list[tuple[str, ET.Element]] = []
        free_joint_entries: list[tuple[str | None, ET.Element]] = [
            (element.get("name"), element)
            for element in _descendants(worldbody, "freejoint")
        ]
        for joint in scalar_joints:
            if joint.get("type", "hinge") == "free":
                free_joint_entries.append((joint.get("name"), joint))
                continue
            name = joint.get("name")
            if name is not None:
                named_joints.append((name, joint))

        if len(free_joint_entries) != 1:
            raise MJCFAssetError(
                f"{source}: expected exactly one floating-base joint "
                f"(<freejoint> or <joint type='free'>), found {len(free_joint_entries)}"
            )

        xml_joint_names = tuple(name for name, _ in named_joints)
        duplicate_joints = _duplicates(xml_joint_names)
        if duplicate_joints:
            raise MJCFAssetError(
                f"{source}: duplicate MJCF joint names: {list(duplicate_joints)}"
            )

        policy_names = self.deployment.joint_names
        by_joint_name = {name: element for name, element in named_joints}
        missing_joints = tuple(name for name in policy_names if name not in by_joint_name)
        if missing_joints:
            raise MJCFAssetError(
                f"{source}: missing {len(missing_joints)} policy joints: "
                f"{list(missing_joints)}"
            )
        invalid_joint_types = {
            name: by_joint_name[name].get("type", "hinge")
            for name in policy_names
            if by_joint_name[name].get("type", "hinge") != "hinge"
        }
        if invalid_joint_types:
            raise MJCFAssetError(
                f"{source}: policy joints must be scalar hinge joints; "
                f"invalid={invalid_joint_types}"
            )

        actuator_sections = tuple(_children(root, "actuator"))
        if len(actuator_sections) != 1:
            raise MJCFAssetError(
                f"{source}: expected exactly one <actuator> section, "
                f"found {len(actuator_sections)}"
            )
        motors = tuple(_descendants(actuator_sections[0], "motor"))
        motor_joint_names = tuple(motor.get("joint", "") for motor in motors)
        motor_names = tuple(motor.get("name") for motor in motors)
        motor_indices_by_joint: dict[str, list[int]] = {}
        for index, joint_name in enumerate(motor_joint_names):
            if joint_name:
                motor_indices_by_joint.setdefault(joint_name, []).append(index)

        missing_motors = tuple(
            name for name in policy_names if name not in motor_indices_by_joint
        )
        duplicate_motors = {
            name: indices
            for name, indices in motor_indices_by_joint.items()
            if name in set(policy_names) and len(indices) != 1
        }
        if missing_motors or duplicate_motors:
            raise MJCFAssetError(
                f"{source}: every policy joint needs exactly one <motor>; "
                f"missing={list(missing_motors)}, duplicate={duplicate_motors}"
            )

        parent_by_element = {
            child: parent for parent in worldbody.iter() for child in parent
        }
        free_element = free_joint_entries[0][1]
        free_body = _ancestor_body(free_element, parent_by_element)
        if free_body is None:
            raise MJCFAssetError(f"{source}: floating-base joint is not inside a <body>")
        if not _is_descendant_or_same(torso, free_body, parent_by_element):
            raise MJCFAssetError(
                f"{source}: torso body {torso.get('name')!r} is not in the "
                f"floating-base body subtree rooted at {free_body.get('name')!r}"
            )
        outside_free_subtree = tuple(
            name
            for name in policy_names
            if not _is_descendant_or_same(
                _ancestor_body(by_joint_name[name], parent_by_element),
                free_body,
                parent_by_element,
            )
        )
        if outside_free_subtree:
            raise MJCFAssetError(
                f"{source}: policy joints outside floating-base robot subtree: "
                f"{list(outside_free_subtree)}"
            )

        xml_index = {name: index for index, name in enumerate(xml_joint_names)}
        policy_to_xml = tuple(xml_index[name] for name in policy_names)
        policy_to_motor = tuple(motor_indices_by_joint[name][0] for name in policy_names)
        return MJCFAssetReport(
            source=source,
            source_sha256=source_sha256,
            xml_joint_names=xml_joint_names,
            motor_joint_names=motor_joint_names,
            motor_names=motor_names,
            policy_to_xml_joint=policy_to_xml,
            policy_to_motor=policy_to_motor,
            free_joint_name=free_joint_entries[0][0],
            free_root_body_name=free_body.get("name"),
            torso_body_name=torso.get("name", ""),
        )


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _ancestor_body(
    element: ET.Element, parent_by_element: dict[ET.Element, ET.Element]
) -> ET.Element | None:
    current: ET.Element | None = parent_by_element.get(element)
    while current is not None:
        if _local_tag(current) == "body":
            return current
        current = parent_by_element.get(current)
    return None


def _is_descendant_or_same(
    candidate: ET.Element | None,
    ancestor: ET.Element,
    parent_by_element: dict[ET.Element, ET.Element],
) -> bool:
    current: ET.Element | None = candidate
    while current is not None:
        if current is ancestor:
            return True
        current = parent_by_element.get(current)
    return False


DEFAULT_G1_MJCF_ASSET_CONTRACT = G1MJCFAssetContract()


def validate_g1_mjcf(path: str | Path) -> MJCFAssetReport:
    """Convenience wrapper around the default 37-DoF G1 asset contract."""

    return DEFAULT_G1_MJCF_ASSET_CONTRACT.validate_file(path)


def validate_scene_robot_binding(
    scene_path: str | Path,
    robot_path: str | Path,
) -> MJCFSceneBindingReport:
    """Require a scene to include the exact robot leaf passed to the gate.

    This closes the unsafe ``validate robot A, execute scene B`` gap.  The
    generated minimum scene deliberately has one include; richer scenes need a
    trusted flattening/provenance step before they enter this smoke runner.
    """

    scene = Path(scene_path).expanduser().resolve()
    robot = Path(robot_path).expanduser().resolve()
    try:
        scene_payload = scene.read_bytes()
        robot_payload = robot.read_bytes()
    except OSError as exc:
        raise MJCFParseError(f"cannot read scene/robot MJCF: {exc}") from exc
    try:
        root = ET.fromstring(scene_payload)
    except ET.ParseError as exc:
        raise MJCFParseError(f"malformed scene MJCF {scene}: {exc}") from exc
    if _local_tag(root) != "mujoco":
        raise MJCFAssetError(f"{scene}: expected <mujoco> document root")
    includes = tuple(_descendants(root, "include"))
    if len(includes) != 1:
        raise MJCFIncludeError(
            f"{scene}: minimum deployment scene must have exactly one <include>, "
            f"found {len(includes)}"
        )
    raw = includes[0].get("file")
    if not raw:
        raise MJCFIncludeError(f"{scene}: <include> is missing file")
    included = Path(raw).expanduser()
    if not included.is_absolute():
        included = scene.parent / included
    included = included.resolve()
    if included != robot:
        raise MJCFIncludeError(
            f"{scene}: scene includes {included}, but validated robot is {robot}"
        )
    # Apply the full leaf contract to the exact bytes bound above.
    DEFAULT_G1_MJCF_ASSET_CONTRACT.validate_bytes(robot_payload, source=str(robot))
    return MJCFSceneBindingReport(
        scene_path=str(scene),
        scene_sha256=sha256(scene_payload).hexdigest(),
        robot_path=str(robot),
        robot_sha256=sha256(robot_payload).hexdigest(),
        include_file=raw,
    )


__all__ = [
    "DEFAULT_G1_MJCF_ASSET_CONTRACT",
    "G1MJCFAssetContract",
    "MJCFAssetError",
    "MJCFAssetReport",
    "MJCFSceneBindingReport",
    "MJCFIncludeError",
    "MJCFParseError",
    "validate_g1_mjcf",
    "validate_scene_robot_binding",
]
