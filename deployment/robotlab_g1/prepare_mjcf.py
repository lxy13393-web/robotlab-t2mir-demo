"""Prepare Unitree's published 37-DoF G1 MJCF for torque-PD deployment.

The upstream ``kinect_teleoperate`` model is intentionally useful as an asset
source, but it is not directly compatible with this deployment boundary: its
base is fixed and its 37 actuators are MuJoCo ``position`` servos.  RobotLab's
policy instead expects a floating robot whose raw action is converted to a
joint target and then to torque by the explicit PD controller in this package.

This module performs the small, auditable conversion without ever editing the
upstream checkout in place.  It rejects mixed/ambiguous actuator layouts,
creates one free root, replaces all policy ``position`` actuators by unit-gear
torque ``motor`` actuators, and writes hashes for both source and outputs.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from .assets import MJCFAssetError
from .contract import DEFAULT_CONTRACT, DeploymentContract
from .control import robotlab_g1_joint_armatures, robotlab_g1_pd_gains


class MJCFPreparationError(ValueError):
    """Raised when a source asset cannot be converted unambiguously."""


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_tag(child) == tag)


def _descendants(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    return (child for child in element.iter() if _local_tag(child) == tag)


def _sha256(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _read_xml(path: Path) -> tuple[bytes, ET.Element]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MJCFPreparationError(f"cannot read source MJCF {path}: {exc}") from exc
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MJCFPreparationError(f"malformed source MJCF {path}: {exc}") from exc
    if _local_tag(root) != "mujoco":
        raise MJCFPreparationError(f"{path}: expected a <mujoco> root")
    return payload, root


def _serialize(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _rewrite_compiler_directories(root: ET.Element, source_parent: Path) -> None:
    """Keep relative mesh/texture references valid after moving the XML."""

    compilers = tuple(_children(root, "compiler"))
    if len(compilers) > 1:
        raise MJCFPreparationError("source has more than one <compiler> section")
    has_mesh_files = any(item.get("file") for item in _descendants(root, "mesh"))
    has_texture_files = any(item.get("file") for item in _descendants(root, "texture"))
    # A scene containing only builtin textures needs no compiler section.  In
    # particular, inserting an empty <compiler/> before an included robot XML
    # makes MuJoCo resolve the included mesh files relative to the generated
    # scene instead of honoring the robot's absolute meshdir.
    if not compilers and not has_mesh_files and not has_texture_files:
        return
    if compilers:
        compiler = compilers[0]
    else:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    assetdir_raw = compiler.get("assetdir")
    for attribute in ("meshdir", "texturedir"):
        raw = compiler.get(attribute)
        configured = raw or assetdir_raw
        directory = source_parent if configured is None else Path(configured).expanduser()
        if not directory.is_absolute():
            directory = source_parent / directory
        # Do not require the directory when the document has no matching file
        # assets; generated/simple MJCFs often need neither one.
        has_files = has_mesh_files if attribute == "meshdir" else has_texture_files
        directory = directory.resolve()
        if has_files and not directory.is_dir():
            raise MJCFPreparationError(
                f"source {attribute} does not exist: {directory}"
            )
        if configured is not None or has_files:
            compiler.set(attribute, str(directory))
    if assetdir_raw is not None:
        assetdir = Path(assetdir_raw).expanduser()
        if not assetdir.is_absolute():
            assetdir = source_parent / assetdir
        compiler.set("assetdir", str(assetdir.resolve()))


def _source_asset_inventory(root: ET.Element, source_parent: Path) -> list[dict[str, Any]]:
    """Hash every external mesh/texture referenced by the source robot."""

    compilers = tuple(_children(root, "compiler"))
    if len(compilers) > 1:
        raise MJCFPreparationError("source has more than one <compiler> section")
    compiler = compilers[0] if compilers else None
    assetdir = None if compiler is None else compiler.get("assetdir")
    inventory: list[dict[str, Any]] = []
    for tag, directory_attribute in (("mesh", "meshdir"), ("texture", "texturedir")):
        configured = None if compiler is None else compiler.get(directory_attribute)
        configured = configured or assetdir
        base = source_parent if configured is None else Path(configured).expanduser()
        if not base.is_absolute():
            base = source_parent / base
        for element in _descendants(root, tag):
            raw = element.get("file")
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise MJCFPreparationError(
                    f"referenced {tag} asset is unavailable: {path}: {exc}"
                ) from exc
            inventory.append(
                {
                    "kind": tag,
                    "name": element.get("name"),
                    "path": str(path),
                    "sha256": _sha256(payload),
                    "bytes": len(payload),
                }
            )
    return sorted(inventory, key=lambda item: (str(item["kind"]), str(item["path"])))


def _inline_external_asset_paths(root: ET.Element) -> None:
    """Make mesh/texture file paths include-safe for MuJoCo 3.1.x.

    MuJoCo combines compiler directory prefixes when an MJCF containing its
    own compiler is included by a scene.  Rewriting every external file to its
    already-validated absolute path avoids that ambiguous composition while
    the preparation manifest still pins every referenced byte by SHA-256.
    """

    compilers = tuple(_children(root, "compiler"))
    compiler = compilers[0] if compilers else None
    assetdir = None if compiler is None else compiler.get("assetdir")
    for tag, directory_attribute in (("mesh", "meshdir"), ("texture", "texturedir")):
        configured = None if compiler is None else compiler.get(directory_attribute)
        configured = configured or assetdir
        base = Path(configured).expanduser() if configured is not None else None
        for element in _descendants(root, tag):
            raw = element.get("file")
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                if base is None:
                    raise MJCFPreparationError(
                        f"cannot make relative {tag} asset include-safe without a directory: {raw}"
                    )
                path = base / path
            path = path.resolve()
            if not path.is_file():
                raise MJCFPreparationError(f"referenced {tag} asset is unavailable: {path}")
            element.set("file", str(path))
    if compiler is not None:
        for attribute in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(attribute, None)


def _body_contains(body: ET.Element, names: set[str]) -> bool:
    return any(candidate.get("name") in names for candidate in _descendants(body, "body"))


def _convert_robot_tree(
    root: ET.Element,
    *,
    source_parent: Path,
    contract: DeploymentContract,
) -> dict[str, Any]:
    if tuple(_descendants(root, "include")):
        raise MJCFPreparationError("source robot MJCF must not contain <include>")
    worldbodies = tuple(_children(root, "worldbody"))
    if len(worldbodies) != 1:
        raise MJCFPreparationError(
            f"source robot needs exactly one <worldbody>, found {len(worldbodies)}"
        )
    worldbody = worldbodies[0]

    active_free = list(_descendants(worldbody, "freejoint")) + [
        joint
        for joint in _descendants(worldbody, "joint")
        if joint.get("type", "hinge") == "free"
    ]
    if active_free:
        raise MJCFPreparationError(
            "source already has an active floating joint; refusing to guess whether it is safe"
        )

    named_scalar: dict[str, ET.Element] = {}
    duplicates: list[str] = []
    for joint in _descendants(worldbody, "joint"):
        name = joint.get("name")
        if not name:
            continue
        if name in named_scalar:
            duplicates.append(name)
        named_scalar[name] = joint
    if duplicates:
        raise MJCFPreparationError(f"duplicate source joints: {sorted(set(duplicates))}")
    policy_names = set(contract.joint_names)
    missing = sorted(policy_names - set(named_scalar))
    extra = sorted(set(named_scalar) - policy_names)
    if missing or extra:
        raise MJCFPreparationError(
            f"source joint set is not the exact 37-DoF policy set; missing={missing}, extra={extra}"
        )
    invalid_types = {
        name: named_scalar[name].get("type", "hinge")
        for name in contract.joint_names
        if named_scalar[name].get("type", "hinge") != "hinge"
    }
    if invalid_types:
        raise MJCFPreparationError(f"all policy joints must be hinge joints: {invalid_types}")

    # The upstream model's default joint template currently contributes
    # damping=0.5, frictionloss=0.1 and armature=0.01.  This deployment applies
    # RobotLab's actuator damping explicitly in PositionPDController, while the
    # recorded PhysX asset has no joint friction and uses 0.001 armature on the
    # hand joints.  Explicit attributes override any upstream defaults and
    # prevent double damping/friction from entering the benchmark.
    armatures = robotlab_g1_joint_armatures(contract)
    for name, armature in zip(contract.joint_names, armatures, strict=True):
        joint = named_scalar[name]
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        joint.set("armature", f"{float(armature):g}")

    root_bodies = tuple(_children(worldbody, "body"))
    torso_aliases = {"torso_link", "torso"}
    candidates = tuple(body for body in root_bodies if _body_contains(body, torso_aliases))
    if len(candidates) != 1:
        raise MJCFPreparationError(
            "cannot identify exactly one top-level robot body containing torso_link/torso"
        )
    free_root = candidates[0]
    if any(_local_tag(child) in {"joint", "freejoint"} for child in free_root):
        raise MJCFPreparationError(
            f"root body {free_root.get('name')!r} already contains an active joint"
        )
    free_root.set(
        "pos", " ".join(f"{value:g}" for value in contract.initial_root_position)
    )
    free_root.set(
        "quat",
        " ".join(
            f"{value:g}" for value in contract.initial_root_quaternion_wxyz
        ),
    )
    free_root.insert(0, ET.Element("freejoint", {"name": "floating_base_joint"}))

    actuator_sections = tuple(_children(root, "actuator"))
    if len(actuator_sections) != 1:
        raise MJCFPreparationError(
            f"source needs exactly one <actuator> section, found {len(actuator_sections)}"
        )
    actuator = actuator_sections[0]
    actuator_children = list(actuator)
    by_joint: dict[str, tuple[int, ET.Element]] = {}
    for index, element in enumerate(actuator_children):
        joint_name = element.get("joint")
        if joint_name in policy_names:
            if joint_name in by_joint:
                raise MJCFPreparationError(f"duplicate actuator for joint {joint_name!r}")
            by_joint[joint_name] = (index, element)
        else:
            raise MJCFPreparationError(
                f"unexpected actuator without a policy joint: tag={_local_tag(element)!r}, "
                f"joint={joint_name!r}"
            )
    missing_actuators = sorted(policy_names - set(by_joint))
    if missing_actuators:
        raise MJCFPreparationError(f"missing source actuators: {missing_actuators}")
    wrong_tags = {
        name: _local_tag(element)
        for name, (_, element) in by_joint.items()
        if _local_tag(element) != "position"
    }
    if wrong_tags:
        raise MJCFPreparationError(
            "source must contain exactly the upstream position-servo layout; "
            f"non-position actuators={wrong_tags}"
        )

    limits = robotlab_g1_pd_gains(contract).effort_limit
    limit_by_name = dict(zip(contract.joint_names, limits, strict=True))
    for joint_name, (index, old) in sorted(by_joint.items(), key=lambda item: item[1][0]):
        limit = float(limit_by_name[joint_name])
        attributes = {
            "name": old.get("name") or f"{joint_name}_motor",
            "joint": joint_name,
            "gear": "1",
            "ctrllimited": "true",
            "ctrlrange": f"{-limit:g} {limit:g}",
        }
        actuator.remove(old)
        actuator.insert(index, ET.Element("motor", attributes))

    _rewrite_compiler_directories(root, source_parent)
    return {
        "free_root_body": free_root.get("name"),
        "free_joint": "floating_base_joint",
        "source_actuator_type": "position",
        "output_actuator_type": "direct-unit-gear-torque-motor",
        "converted_actuators": contract.action_dim,
        "joint_dynamics": {
            "damping": 0.0,
            "frictionloss": 0.0,
            "armature_policy_order": armatures.astype(float).tolist(),
        },
        "initial_root_position": list(contract.initial_root_position),
        "initial_root_quaternion_wxyz": list(
            contract.initial_root_quaternion_wxyz
        ),
    }


def _prepare_scene(
    source_scene: Path | None,
    source_robot: Path,
    robot_output_name: str,
) -> tuple[bytes | None, ET.Element]:
    if source_scene is None:
        root = ET.Element("mujoco", {"model": "robotlab_g1_37dof_scene"})
        ET.SubElement(root, "include", {"file": robot_output_name})
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3"})
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1"})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "0 0 0.05",
                "friction": "1 0.005 0.0001",
            },
        )
        return None, root

    scene_payload, root = _read_xml(source_scene)
    includes = tuple(_descendants(root, "include"))
    matching: list[ET.Element] = []
    for include in includes:
        raw = include.get("file")
        if not raw:
            raise MJCFPreparationError(f"{source_scene}: <include> lacks file")
        included = Path(raw).expanduser()
        if not included.is_absolute():
            included = source_scene.parent / included
        if included.resolve() == source_robot.resolve():
            matching.append(include)
        else:
            raise MJCFPreparationError(
                f"{source_scene}: extra include {raw!r} cannot be relocated safely"
            )
    if len(matching) != 1:
        raise MJCFPreparationError(
            f"{source_scene}: expected exactly one include of {source_robot.name}, found {len(matching)}"
        )
    matching[0].set("file", robot_output_name)
    _rewrite_compiler_directories(root, source_scene.parent)
    return scene_payload, root


def _inherit_robot_asset_directories(
    scene_root: ET.Element, robot_root: ET.Element
) -> None:
    """Expose included robot asset directories on the top-level scene.

    MuJoCo resolves file assets using the compiler options of the top-level
    document.  Compiler directives inside an included robot file are not
    sufficient after relocation, even when they contain absolute paths.
    """

    robot_compilers = tuple(_children(robot_root, "compiler"))
    if not robot_compilers:
        return
    scene_compilers = tuple(_children(scene_root, "compiler"))
    if len(scene_compilers) > 1:
        raise MJCFPreparationError("scene has more than one <compiler> section")
    if scene_compilers:
        scene_compiler = scene_compilers[0]
    else:
        scene_compiler = ET.Element("compiler")
        scene_root.insert(0, scene_compiler)
    robot_compiler = robot_compilers[0]
    for attribute in ("assetdir", "meshdir", "texturedir"):
        value = robot_compiler.get(attribute)
        if value is not None:
            scene_compiler.set(attribute, value)


def _check_write(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(
            f"refusing to overwrite different generated artifact: {path}; choose a new --output-dir"
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_mjcf(
    source_robot: str | Path,
    output_dir: str | Path,
    *,
    source_scene: str | Path | None = None,
    contract: DeploymentContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Convert and write a deployment leaf, scene and provenance manifest."""

    source_robot_path = Path(source_robot).expanduser().resolve()
    source_scene_path = None if source_scene is None else Path(source_scene).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    robot_path = destination / "g1_robotlab_torque.xml"
    scene_path = destination / "scene_robotlab.xml"
    manifest_path = destination / "mjcf_preparation.json"

    robot_source_payload, robot_root = _read_xml(source_robot_path)
    source_assets = _source_asset_inventory(robot_root, source_robot_path.parent)
    transformation = _convert_robot_tree(
        robot_root,
        source_parent=source_robot_path.parent,
        contract=contract,
    )
    _inline_external_asset_paths(robot_root)
    robot_payload = _serialize(robot_root)
    source_scene_payload, scene_root = _prepare_scene(
        source_scene_path,
        source_robot_path,
        robot_path.name,
    )
    _inherit_robot_asset_directories(scene_root, robot_root)
    scene_payload = _serialize(scene_root)

    # Validate the exact bytes that will be consumed, before touching disk.
    report = validate_g1_mjcf_bytes(robot_payload, source=str(robot_path))
    manifest: dict[str, Any] = {
        "format_version": 1,
        "contract": {
            "state_dim": contract.state_dim,
            "action_dim": contract.action_dim,
            "joint_names": list(contract.joint_names),
            "sha256": contract.sha256,
        },
        "source_robot": {
            "path": str(source_robot_path),
            "sha256": _sha256(robot_source_payload),
        },
        "source_scene": None
        if source_scene_path is None
        else {
            "path": str(source_scene_path),
            "sha256": _sha256(source_scene_payload or b""),
        },
        "source_assets": source_assets,
        "transformation": transformation,
        "outputs": {
            "robot": {"path": str(robot_path), "sha256": _sha256(robot_payload)},
            "scene": {"path": str(scene_path), "sha256": _sha256(scene_payload)},
        },
        "static_contract": report.as_dict(),
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    for path, payload in (
        (robot_path, robot_payload),
        (scene_path, scene_payload),
        (manifest_path, manifest_payload),
    ):
        _check_write(path, payload)
    for path, payload in (
        (robot_path, robot_payload),
        (scene_path, scene_payload),
        (manifest_path, manifest_payload),
    ):
        _atomic_write(path, payload)
    return manifest


def validate_g1_mjcf_bytes(payload: bytes, *, source: str):
    """Use the deployment contract without requiring a temporary file."""

    from .assets import DEFAULT_G1_MJCF_ASSET_CONTRACT

    try:
        return DEFAULT_G1_MJCF_ASSET_CONTRACT.validate_bytes(payload, source=source)
    except MJCFAssetError as exc:
        raise MJCFPreparationError(f"generated MJCF failed deployment contract: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_robot", type=Path, help="upstream fixed-base 37-DoF g1.xml")
    parser.add_argument("--source-scene", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_mjcf(
        args.source_robot,
        args.output_dir,
        source_scene=args.source_scene,
    )
    print(
        "[MJCF-PREPARE] PASS "
        f"joints={manifest['contract']['action_dim']} "
        f"robot={manifest['outputs']['robot']['path']} "
        f"scene={manifest['outputs']['scene']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MJCFPreparationError", "prepare_mjcf"]
