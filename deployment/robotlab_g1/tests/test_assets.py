from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from deployment.robotlab_g1.assets import (
    G1MJCFAssetContract,
    MJCFAssetError,
    MJCFIncludeError,
    MJCFParseError,
    validate_g1_mjcf,
    validate_scene_robot_binding,
)
from deployment.robotlab_g1.contract import G1_POLICY_JOINT_NAMES


def make_mjcf(
    *,
    joints: tuple[str, ...] = G1_POLICY_JOINT_NAMES,
    motors: tuple[str, ...] | None = None,
    torso_name: str = "torso_link",
    freejoint: str = '<freejoint name="floating_base"/>',
    actuator_tag: str = "motor",
) -> str:
    if motors is None:
        motors = joints
    joint_xml = "\n".join(f'<joint name="{name}"/>' for name in joints)
    motor_xml = "\n".join(
        f'<{actuator_tag} name="{name}_motor" joint="{name}"/>' for name in motors
    )
    return f"""<mujoco model="g1_37dof_test">
  <worldbody>
    <body name="pelvis">
      {freejoint}
      <body name="{torso_name}">
        {joint_xml}
      </body>
    </body>
  </worldbody>
  <actuator>
    {motor_xml}
  </actuator>
</mujoco>"""


class G1MJCFAssetContractTests(unittest.TestCase):
    def test_valid_asset_builds_explicit_joint_and_motor_maps(self) -> None:
        xml_order = tuple(reversed(G1_POLICY_JOINT_NAMES))
        motor_order = G1_POLICY_JOINT_NAMES[::2] + G1_POLICY_JOINT_NAMES[1::2]
        xml = make_mjcf(joints=xml_order, motors=motor_order)

        report = G1MJCFAssetContract().validate_xml(xml, source="synthetic.xml")

        self.assertEqual(report.policy_joint_count, 37)
        self.assertEqual(report.policy_motor_count, 37)
        self.assertEqual(report.free_joint_name, "floating_base")
        self.assertEqual(report.free_root_body_name, "pelvis")
        self.assertEqual(report.torso_body_name, "torso_link")
        self.assertEqual(
            tuple(report.xml_joint_names[index] for index in report.policy_to_xml_joint),
            G1_POLICY_JOINT_NAMES,
        )
        self.assertEqual(
            tuple(report.motor_joint_names[index] for index in report.policy_to_motor),
            G1_POLICY_JOINT_NAMES,
        )
        self.assertEqual(report.as_dict()["policy_joint_count"], 37)

    def test_file_validation_records_exact_provenance_hash(self) -> None:
        payload = make_mjcf().encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.xml"
            path.write_bytes(payload)
            report = validate_g1_mjcf(path)

        self.assertEqual(report.source_sha256, sha256(payload).hexdigest())
        self.assertTrue(report.source.endswith("g1.xml"))

    def test_missing_joint_and_non_motor_actuator_are_rejected(self) -> None:
        missing = G1_POLICY_JOINT_NAMES[:-1]
        with self.assertRaisesRegex(MJCFAssetError, "missing 1 policy joints"):
            G1MJCFAssetContract().validate_xml(make_mjcf(joints=missing))

        with self.assertRaisesRegex(MJCFAssetError, "exactly one <motor>"):
            G1MJCFAssetContract().validate_xml(make_mjcf(actuator_tag="position"))

    def test_duplicate_joint_and_duplicate_motor_are_rejected(self) -> None:
        duplicate_joints = G1_POLICY_JOINT_NAMES + (G1_POLICY_JOINT_NAMES[0],)
        with self.assertRaisesRegex(MJCFAssetError, "duplicate MJCF joint names"):
            G1MJCFAssetContract().validate_xml(make_mjcf(joints=duplicate_joints))

        duplicate_motors = G1_POLICY_JOINT_NAMES + (G1_POLICY_JOINT_NAMES[0],)
        with self.assertRaisesRegex(MJCFAssetError, "duplicate=.*left_hip_pitch_joint"):
            G1MJCFAssetContract().validate_xml(make_mjcf(motors=duplicate_motors))

    def test_requires_one_free_joint_and_torso_in_its_subtree(self) -> None:
        with self.assertRaisesRegex(MJCFAssetError, "floating-base joint"):
            G1MJCFAssetContract().validate_xml(make_mjcf(freejoint=""))

        with self.assertRaisesRegex(MJCFAssetError, "torso body"):
            G1MJCFAssetContract().validate_xml(make_mjcf(torso_name="not_the_torso"))

        # The torso name alone is insufficient when it belongs to a static
        # decoration rather than the floating robot subtree.
        xml = make_mjcf().replace(
            '<body name="torso_link">',
            '<body name="robot_torso">',
        ).replace(
            "</worldbody>",
            '<body name="torso_link"/></worldbody>',
        )
        with self.assertRaisesRegex(MJCFAssetError, "not in the floating-base body subtree"):
            G1MJCFAssetContract(torso_body_names=("torso_link",)).validate_xml(xml)

    def test_joint_type_free_is_accepted_as_the_floating_base(self) -> None:
        report = G1MJCFAssetContract().validate_xml(
            make_mjcf(freejoint='<joint name="root" type="free"/>')
        )
        self.assertEqual(report.free_joint_name, "root")
        self.assertNotIn("root", report.xml_joint_names)

    def test_include_has_a_specific_non_false_positive_failure(self) -> None:
        xml = '<mujoco><include file="g1.xml"/><worldbody/><actuator/></mujoco>'
        with self.assertRaisesRegex(MJCFIncludeError, "g1.xml.*flattened"):
            G1MJCFAssetContract().validate_xml(xml, source="scene.xml")

    def test_malformed_or_non_mujoco_documents_are_rejected(self) -> None:
        with self.assertRaises(MJCFParseError):
            G1MJCFAssetContract().validate_xml("<mujoco>")
        with self.assertRaisesRegex(MJCFAssetError, "expected <mujoco>"):
            G1MJCFAssetContract().validate_xml("<robot/>")

    def test_scene_must_include_the_exact_validated_robot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = root / "robot.xml"
            robot.write_text(make_mjcf())
            other = root / "other.xml"
            other.write_text(make_mjcf())
            scene = root / "scene.xml"
            scene.write_text('<mujoco><include file="robot.xml"/></mujoco>')
            report = validate_scene_robot_binding(scene, robot)
            self.assertEqual(report.robot_path, str(robot.resolve()))
            with self.assertRaisesRegex(MJCFIncludeError, "validated robot"):
                validate_scene_robot_binding(scene, other)


if __name__ == "__main__":
    unittest.main()
