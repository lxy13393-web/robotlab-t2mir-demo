from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from deployment.robotlab_g1.assets import validate_g1_mjcf
from deployment.robotlab_g1.contract import G1_POLICY_JOINT_NAMES
from deployment.robotlab_g1.control import robotlab_g1_joint_armatures
from deployment.robotlab_g1.prepare_mjcf import (
    MJCFPreparationError,
    prepare_mjcf,
)


def source_xml(*, actuator_tag: str = "position", free: bool = False) -> str:
    joints = "\n".join(f'<joint name="{name}"/>' for name in G1_POLICY_JOINT_NAMES)
    actuators = "\n".join(
        f'<{actuator_tag} name="{name}" joint="{name}" kp="30"/>'
        for name in G1_POLICY_JOINT_NAMES
    )
    root_joint = '<freejoint name="already_free"/>' if free else ""
    return f"""<mujoco model="upstream_g1">
  <compiler angle="radian" meshdir="meshes"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      {root_joint}
      <body name="torso_link">{joints}</body>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>"""


class PrepareMjcfTest(unittest.TestCase):
    def test_fixed_position_asset_becomes_floating_unit_motor_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").mkdir()
            source = root / "g1.xml"
            source_bytes = source_xml().encode()
            source.write_bytes(source_bytes)
            output = root / "generated"

            manifest = prepare_mjcf(source, output)
            robot = output / "g1_robotlab_torque.xml"
            scene = output / "scene_robotlab.xml"
            report = validate_g1_mjcf(robot)

            self.assertEqual(report.policy_joint_count, 37)
            self.assertEqual(report.policy_motor_count, 37)
            self.assertEqual(report.free_joint_name, "floating_base_joint")
            tree = ET.parse(robot)
            motors = tuple(tree.getroot().iter("motor"))
            self.assertEqual(len(motors), 37)
            self.assertTrue(all(motor.get("gear") == "1" for motor in motors))
            self.assertTrue(all(motor.get("ctrllimited") == "true" for motor in motors))
            self.assertEqual(len(tuple(tree.getroot().iter("position"))), 0)
            generated_root = tree.getroot()
            pelvis = next(
                body for body in generated_root.iter("body") if body.get("name") == "pelvis"
            )
            self.assertEqual(pelvis.get("pos"), "0 0 0.74")
            self.assertEqual(pelvis.get("quat"), "1 0 0 0")
            by_name = {
                joint.get("name"): joint for joint in generated_root.iter("joint")
            }
            expected_armatures = robotlab_g1_joint_armatures()
            for name, expected in zip(
                G1_POLICY_JOINT_NAMES, expected_armatures, strict=True
            ):
                self.assertEqual(by_name[name].get("damping"), "0")
                self.assertEqual(by_name[name].get("frictionloss"), "0")
                self.assertAlmostEqual(float(by_name[name].get("armature")), float(expected))
            self.assertTrue(scene.is_file())
            self.assertEqual(manifest["source_robot"]["sha256"], sha256(source_bytes).hexdigest())
            self.assertEqual(
                json.loads((output / "mjcf_preparation.json").read_text())["outputs"]["robot"]["sha256"],
                sha256(robot.read_bytes()).hexdigest(),
            )

            # The conversion is deterministic and safely repeatable.
            self.assertEqual(prepare_mjcf(source, output), manifest)

    def test_source_scene_include_is_rebound_to_generated_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").mkdir()
            source = root / "g1.xml"
            source.write_text(source_xml())
            scene = root / "scene.xml"
            scene.write_text(
                '<mujoco model="scene"><include file="g1.xml"/>'
                '<worldbody><geom name="floor" type="plane" size="0 0 0.1"/></worldbody></mujoco>'
            )
            output = root / "generated"
            prepare_mjcf(source, output, source_scene=scene)
            generated = ET.parse(output / "scene_robotlab.xml").getroot()
            includes = tuple(generated.iter("include"))
            self.assertEqual(len(includes), 1)
            self.assertEqual(includes[0].get("file"), "g1_robotlab_torque.xml")
            # MuJoCo resolves included assets with the top-level compiler, so
            # the generated scene must inherit the robot's relocated meshdir.
            scene_compiler = generated.find("compiler")
            robot_compiler = ET.parse(output / "g1_robotlab_torque.xml").getroot().find(
                "compiler"
            )
            self.assertIsNotNone(scene_compiler)
            self.assertIsNotNone(robot_compiler)
            self.assertEqual(
                scene_compiler.get("meshdir"), robot_compiler.get("meshdir")
            )

    def test_external_meshes_are_hashed_in_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meshes = root / "meshes"
            meshes.mkdir()
            mesh = meshes / "piece.obj"
            mesh.write_bytes(b"synthetic mesh bytes")
            source = root / "g1.xml"
            xml = source_xml().replace(
                "<worldbody>",
                '<asset><mesh name="piece" file="piece.obj"/></asset><worldbody>',
            )
            source.write_text(xml)
            manifest = prepare_mjcf(source, root / "generated")
            self.assertEqual(len(manifest["source_assets"]), 1)
            self.assertEqual(
                manifest["source_assets"][0]["sha256"], sha256(mesh.read_bytes()).hexdigest()
            )
            generated_mesh = next(
                ET.parse(root / "generated/g1_robotlab_torque.xml").getroot().iter("mesh")
            )
            self.assertEqual(generated_mesh.get("file"), str(mesh.resolve()))

    def test_assetdir_is_inlined_after_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "piece.obj").write_bytes(b"synthetic mesh bytes")
            source = root / "g1.xml"
            xml = source_xml().replace('meshdir="meshes"', 'assetdir="assets"').replace(
                "<worldbody>",
                '<asset><mesh name="piece" file="piece.obj"/></asset><worldbody>',
            )
            source.write_text(xml)
            output = root / "generated"
            prepare_mjcf(source, output)
            compiler = ET.parse(output / "g1_robotlab_torque.xml").getroot().find(
                "compiler"
            )
            self.assertIsNotNone(compiler)
            self.assertIsNone(compiler.get("assetdir"))
            self.assertIsNone(compiler.get("meshdir"))
            generated_mesh = next(
                ET.parse(output / "g1_robotlab_torque.xml").getroot().iter("mesh")
            )
            self.assertEqual(generated_mesh.get("file"), str((assets / "piece.obj").resolve()))

    def test_refuses_active_free_joint_or_non_position_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").mkdir()
            source = root / "g1.xml"
            source.write_text(source_xml(free=True))
            with self.assertRaisesRegex(MJCFPreparationError, "already has an active"):
                prepare_mjcf(source, root / "out-free")

            source.write_text(source_xml(actuator_tag="motor"))
            with self.assertRaisesRegex(MJCFPreparationError, "non-position actuators"):
                prepare_mjcf(source, root / "out-motor")

    def test_refuses_to_overwrite_changed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meshes").mkdir()
            source = root / "g1.xml"
            source.write_text(source_xml())
            output = root / "generated"
            prepare_mjcf(source, output)
            (output / "g1_robotlab_torque.xml").write_text("changed")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_mjcf(source, output)


if __name__ == "__main__":
    unittest.main()
