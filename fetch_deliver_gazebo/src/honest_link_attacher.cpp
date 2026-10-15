// Copyright 2026 Ali
// SPDX-License-Identifier: Apache-2.0
//
// HonestLinkAttacher: a Gazebo Classic world plugin that simulates a grasp by
// rigidly joining an object to the gripper, but only when the grasp would be
// physically plausible. Every condition is checked against Gazebo ground
// truth, never against what the caller claims:
//
//   1. pose reached   - the actual end-effector pose (from the simulated arm)
//                       is within tolerance of the planned grasp pose;
//   2. gripper closed - the fingers have closed below a threshold opening;
//   3. in tolerance   - the object center lies within a distance tolerance of
//                       the grasp center between the fingers.
//
// If any condition fails the attach is rejected and the grasp counts as failed.

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Pose3.hh>
#include <rclcpp/rclcpp.hpp>

#include "fetch_deliver_interfaces/srv/attach_object.hpp"
#include "fetch_deliver_interfaces/srv/detach_object.hpp"

namespace fetch_deliver_gazebo
{

using AttachObject = fetch_deliver_interfaces::srv::AttachObject;
using DetachObject = fetch_deliver_interfaces::srv::DetachObject;

class HonestLinkAttacher : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    node_ = gazebo_ros::Node::Get(sdf);

    auto get_str = [&](const std::string & key, const std::string & def) {
        return sdf->HasElement(key) ? sdf->Get<std::string>(key) : def;
      };
    auto get_dbl = [&](const std::string & key, double def) {
        return sdf->HasElement(key) ? sdf->Get<double>(key) : def;
      };

    robot_model_ = get_str("robot_model", "turtlebot3_manipulation_system");
    base_link_ = get_str("base_link", "base_footprint");
    ee_link_ = get_str("ee_link", "link5");
    ee_offset_ = sdf->HasElement("ee_offset") ?
      sdf->Get<ignition::math::Vector3d>("ee_offset") : ignition::math::Vector3d(0.126, 0, 0);
    left_finger_joint_ = get_str("left_finger_joint", "gripper_left_joint");
    right_finger_joint_ = get_str("right_finger_joint", "gripper_right_joint");
    open_joint_position_ = get_dbl("open_joint_position", 0.019);
    finger_base_gap_ = get_dbl("finger_base_gap", 0.0);
    max_closed_opening_ = get_dbl("max_closed_opening", 0.032);
    pose_tolerance_ = get_dbl("pose_tolerance", 0.02);
    angle_tolerance_ = get_dbl("angle_tolerance", 0.35);
    object_tolerance_ = get_dbl("object_tolerance", 0.025);

    attach_srv_ = node_->create_service<AttachObject>(
      "attach",
      std::bind(&HonestLinkAttacher::OnAttach, this, std::placeholders::_1, std::placeholders::_2));
    detach_srv_ = node_->create_service<DetachObject>(
      "detach",
      std::bind(&HonestLinkAttacher::OnDetach, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
      node_->get_logger(),
      "HonestLinkAttacher ready (robot=%s, ee=%s, pose_tol=%.3f m, object_tol=%.3f m, "
      "max_closed_opening=%.3f m)",
      robot_model_.c_str(), ee_link_.c_str(), pose_tolerance_, object_tolerance_,
      max_closed_opening_);
  }

private:
  // Opening between the fingers, from the simulated joint positions.
  double FingerOpening(const gazebo::physics::ModelPtr & robot) const
  {
    auto left = robot->GetJoint(left_finger_joint_);
    auto right = robot->GetJoint(right_finger_joint_);
    if (!left) {
      return std::numeric_limits<double>::infinity();
    }
    const double l = left->Position(0);
    const double r = right ? right->Position(0) : l;
    return finger_base_gap_ + l + r;
  }

  ignition::math::Pose3d EndEffectorWorldPose(const gazebo::physics::LinkPtr & ee) const
  {
    return ignition::math::Pose3d(ee_offset_, ignition::math::Quaterniond::Identity) +
           ee->WorldPose();
  }

  void OnAttach(
    const std::shared_ptr<AttachObject::Request> req,
    std::shared_ptr<AttachObject::Response> res)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    boost::recursive_mutex::scoped_lock lock(*world_->Physics()->GetPhysicsUpdateMutex());

    auto robot = world_->ModelByName(robot_model_);
    if (!robot) {
      res->success = false;
      res->message = "robot model not found";
      return;
    }
    auto ee_for_search = robot->GetLink(ee_link_);
    // The robot does not need to know the object's simulator name: with an
    // empty object_model the plugin takes the movable model closest to the
    // grasp center (the tolerance check below still applies).
    gazebo::physics::ModelPtr object;
    if (!req->object_model.empty()) {
      object = world_->ModelByName(req->object_model);
    } else if (ee_for_search) {
      const auto center = EndEffectorWorldPose(ee_for_search).Pos();
      double best = std::numeric_limits<double>::infinity();
      for (const auto & m : world_->Models()) {
        if (m->IsStatic() || m == robot || m->GetLinks().size() != 1) {
          continue;
        }
        const double d = (m->WorldPose().Pos() - center).Length();
        if (d < best) {
          best = d;
          object = m;
        }
      }
    }
    if (!object) {
      res->success = false;
      res->message = "object model not found";
      return;
    }
    if (joint_) {
      res->success = false;
      res->message = "already holding " + attached_model_;
      return;
    }
    auto base = robot->GetLink(base_link_);
    auto ee = robot->GetLink(ee_link_);
    auto obj_link = req->object_link.empty() ? object->GetLink() : object->GetLink(req->object_link);
    if (!base || !ee || !obj_link) {
      res->success = false;
      res->message = "required link missing (base/ee/object)";
      return;
    }

    // 1. Has the arm actually reached the planned grasp pose?
    const auto ee_world = EndEffectorWorldPose(ee);
    const auto ee_in_base = ee_world - base->WorldPose();
    const auto & p = req->planned_grasp_pose;
    const ignition::math::Vector3d planned_pos(p.position.x, p.position.y, p.position.z);
    ignition::math::Quaterniond planned_rot(
      p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    planned_rot.Normalize();
    res->ee_to_plan_error = (ee_in_base.Pos() - planned_pos).Length();
    const auto dq = planned_rot.Inverse() * ee_in_base.Rot();
    res->ee_to_plan_angle_error = 2.0 * std::acos(std::min(1.0, std::abs(dq.W())));
    res->cond_pose_reached = res->ee_to_plan_error <= pose_tolerance_ &&
      res->ee_to_plan_angle_error <= angle_tolerance_;

    // 2. Is the gripper closed?
    res->gripper_opening = FingerOpening(robot);
    res->cond_gripper_closed = res->gripper_opening <= max_closed_opening_;

    // 3. Is the object between the fingers?
    const auto obj_center = obj_link->WorldCoGPose().Pos();
    res->gripper_to_object_error = (obj_center - ee_world.Pos()).Length();
    res->cond_within_tolerance = res->gripper_to_object_error <= object_tolerance_;

    if (!(res->cond_pose_reached && res->cond_gripper_closed && res->cond_within_tolerance)) {
      res->success = false;
      res->message = "attach rejected:";
      if (!res->cond_pose_reached) {res->message += " grasp pose not reached;";}
      if (!res->cond_gripper_closed) {res->message += " gripper not closed;";}
      if (!res->cond_within_tolerance) {res->message += " object out of tolerance;";}
      RCLCPP_WARN(
        node_->get_logger(), "%s (ee err %.3f m / %.2f rad, opening %.3f m, obj err %.3f m)",
        res->message.c_str(), res->ee_to_plan_error, res->ee_to_plan_angle_error,
        res->gripper_opening, res->gripper_to_object_error);
      return;
    }

    // All conditions hold: create a rigid (zero-range revolute) joint. A
    // locked revolute joint is more stable in ODE than a "fixed" joint.
    joint_ = world_->Physics()->CreateJoint("revolute", robot);
    joint_->Attach(ee, obj_link);
    joint_->Load(ee, obj_link, ignition::math::Pose3d());
    joint_->SetModel(robot);
    joint_->SetUpperLimit(0, 0.0);
    joint_->SetLowerLimit(0, 0.0);
    joint_->Init();
    attached_model_ = object->GetName();
    res->success = true;
    res->message = "attached " + attached_model_;
    RCLCPP_INFO(
      node_->get_logger(), "Attached %s (ee err %.4f m, obj err %.4f m, opening %.4f m)",
      attached_model_.c_str(), res->ee_to_plan_error, res->gripper_to_object_error,
      res->gripper_opening);
  }

  void OnDetach(
    const std::shared_ptr<DetachObject::Request> req,
    std::shared_ptr<DetachObject::Response> res)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    boost::recursive_mutex::scoped_lock lock(*world_->Physics()->GetPhysicsUpdateMutex());
    if (!joint_) {
      res->success = false;
      res->message = "nothing attached";
      return;
    }
    if (!req->object_model.empty() && req->object_model != attached_model_) {
      res->success = false;
      res->message = "holding " + attached_model_ + ", not " + req->object_model;
      return;
    }
    joint_->Detach();
    joint_->Fini();
    joint_.reset();
    if (auto obj = world_->ModelByName(attached_model_)) {
      const auto pose = obj->WorldPose();
      res->object_pose.position.x = pose.Pos().X();
      res->object_pose.position.y = pose.Pos().Y();
      res->object_pose.position.z = pose.Pos().Z();
      res->object_pose.orientation.w = pose.Rot().W();
      res->object_pose.orientation.x = pose.Rot().X();
      res->object_pose.orientation.y = pose.Rot().Y();
      res->object_pose.orientation.z = pose.Rot().Z();
    }
    res->success = true;
    res->message = "detached " + attached_model_;
    RCLCPP_INFO(node_->get_logger(), "%s", res->message.c_str());
    attached_model_.clear();
  }

  gazebo::physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Service<AttachObject>::SharedPtr attach_srv_;
  rclcpp::Service<DetachObject>::SharedPtr detach_srv_;
  gazebo::physics::JointPtr joint_;
  std::string attached_model_;
  std::mutex mutex_;

  std::string robot_model_, base_link_, ee_link_, left_finger_joint_, right_finger_joint_;
  ignition::math::Vector3d ee_offset_;
  double open_joint_position_{0.019}, finger_base_gap_{0.0}, max_closed_opening_{0.032};
  double pose_tolerance_{0.02}, angle_tolerance_{0.35}, object_tolerance_{0.025};
};

GZ_REGISTER_WORLD_PLUGIN(HonestLinkAttacher)

}  // namespace fetch_deliver_gazebo
