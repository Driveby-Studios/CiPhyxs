//==================================================================================================
/// @file  Ragdoll.hpp
/// @brief  Ragdoll builder — chain constraints for character and articulated-body simulation.
///
/// A ragdoll is a tree of rigid bodies connected by hinge joints with angular limits,
/// approximating a skeleton.  This builder provides a high-level API for constructing
/// common ragdoll configurations from a set of body descriptors.
///
/// ## Usage
///
/// @code
///   RagdollBuilder builder(world);
///   RagdollBuilder::Bone bones[] = {
///       { thighDesc,  hipJointDesc  },
///       { shinDesc,   kneeJointDesc },
///       { footDesc,   ankleJointDesc },
///   };
///   builder.createChain(bones, 3);
/// @endcode
//==================================================================================================
#pragma once

#include "PhysicsWorld.hpp"
#include "Joint.hpp"
#include <cstdint>
#include <vector>

namespace ciphyxs {

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RagdollBone
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  Describes one bone (rigid body) and the joint connecting it to its parent.
///
/// The first bone in a chain has no parent joint — its `jointDesc.bodyB` is ignored
/// (it is connected to the world via a joint that you specify separately, or left
/// unattached as the root).
struct RagdollBone {
    RigidBodyDesc bodyDesc;   ///< Descriptor for this bone's rigid body.
    JointDesc     jointDesc;  ///< Joint connecting this bone to its parent.
                              ///< `jointDesc.bodyA` is overwritten with the parent handle.
                              ///< `jointDesc.bodyB` is set to this bone's handle after creation.
};

// ────────────────────────────────────────────────────────────────────────────────────────────────
// RagdollBuilder
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// @brief  High-level builder for constructing ragdoll chains (limbs, spines, tentacles).
///
/// RagdollBuilder simplifies the creation of articulated structures by chaining bodies
/// with hinge joints that have angular limits and optional motors.  It supports:
///
///   - **Linear chains** (arms, legs, tentacles, ropes).
///   - **Branched structures** (spine → left arm, right arm, head) via manual extension.
///   - **Automatic anchor placement** — anchor positions are computed from body extents
///     so that joints connect end-to-end without manual offset calculation.
///
/// ## Example: Simple pendulum arm
///
/// @code
///   RagdollBuilder builder(world);
///
///   // Root body (static mount).
///   RigidBodyDesc mount;
///   mount.motionType = MotionType::Static;
///   mount.setShape(world.createShape(Box{Vec3f(0.1f, 0.1f, 0.1f)}));
///   mount.position = Vec3f(0, 3, 0);
///   RigidBodyHandle root = world.createBody(mount);
///
///   // Upper arm.
///   RagdollBone upperArm;
///   upperArm.bodyDesc.mass = 5.0f;
///   upperArm.bodyDesc.setShape(world.createShape(Box{Vec3f(0.2f, 1.0f, 0.2f)}));
///   upperArm.bodyDesc.position = Vec3f(0, 2, 0);
///   upperArm.jointDesc.type = JointType::Hinge;
///   upperArm.jointDesc.anchorA = Vec3f(0, 2.8f, 0);  // top of upper arm
///   upperArm.jointDesc.anchorB = Vec3f(0, -0.8f, 0); // bottom of upper arm
///   upperArm.jointDesc.axisA = Vec3f::unitZ();
///   upperArm.jointDesc.axisB = Vec3f::unitZ();
///   upperArm.jointDesc.enableLimits = true;
///   upperArm.jointDesc.limitMin = -1.5f;
///   upperArm.jointDesc.limitMax =  1.5f;
///
///   builder.createChain(&upperArm, 1, root);
/// @endcode
class RagdollBuilder {
public:
    /// @brief  Construct a builder bound to a PhysicsWorld.
    explicit RagdollBuilder(PhysicsWorld& world) noexcept : m_world(world) {}

    /// @brief  Create a linear chain of bones connected by hinge joints.
    ///
    /// Each bone (except the first) is connected to the previous bone via its jointDesc.
    /// The `jointDesc` fields are updated so that `bodyA` points to the parent body
    /// and `bodyB` points to the newly created body.
    ///
    /// @param bones    Array of bone descriptors.
    /// @param count    Number of bones in the chain.
    /// @param root     Optional root body handle.  If set to `kInvalidHandle`, the first
    ///                 bone is created as a dynamic body with no parent joint.
    ///                 If valid, the first bone is jointed to this root body.
    /// @return  Vector of rigid-body handles for each bone (size = count).
    ///
    /// @note  All joints are created as **Hinge** type by default.  Override
    ///        `bone.jointDesc.type` for other joint types.
    std::vector<RigidBodyHandle> createChain(RagdollBone* bones, std::size_t count,
                                              RigidBodyHandle root = kInvalidHandle) {
        std::vector<RigidBodyHandle> handles;
        handles.reserve(count);

        RigidBodyHandle parent = root;

        for (std::size_t i = 0; i < count; ++i) {
            // Create the bone body.
            RigidBodyHandle h = m_world.createBody(bones[i].bodyDesc);
            handles.push_back(h);

            // Connect to parent via joint (skip only if this is the first bone AND no root).
            if (i > 0 || root != kInvalidHandle) {
                bones[i].jointDesc.bodyA = parent;
                bones[i].jointDesc.bodyB = h;
                // Use BallSocket if no explicit type was set (default).
                if (bones[i].jointDesc.type == JointType::BallSocket && i > 0) {
                    // Auto-compute anchors from body positions if not manually set.
                    // Default BallSocket anchors at the body positions (end-to-end).
                    if (bones[i].jointDesc.anchorA == Vec3f::zero() &&
                        bones[i].jointDesc.anchorB == Vec3f::zero()) {
                        bones[i].jointDesc.anchorA = m_world.bodies().positions[parent];
                        bones[i].jointDesc.anchorB = m_world.bodies().positions[h];
                    }
                }
                m_world.createJoint(bones[i].jointDesc);
            }

            parent = h;
        }

        return handles;
    }

    /// @brief  Create a branched ragdoll from a tree structure.
    ///
    /// Each entry specifies a body descriptor, a joint descriptor connecting it to
    /// its parent (identified by parent handle), and the resulting handle.
    ///
    /// @param bones      Array of branch bone descriptors.
    /// @param count      Number of bones.
    /// @param[out] outHandles  Receives the created body handles (size = count).
    ///
    /// This is a lower-level building block — the caller manages the tree topology.
    void createBranch(const RagdollBone* bones, std::size_t count,
                      std::vector<RigidBodyHandle>& outHandles) {
        outHandles.clear();
        outHandles.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            RigidBodyHandle h = m_world.createBody(bones[i].bodyDesc);
            outHandles.push_back(h);

            if (bones[i].jointDesc.bodyA != kInvalidHandle) {
                // jointDesc.bodyA should already be set to the parent handle.
                auto jd = bones[i].jointDesc;
                jd.bodyB = h;
                m_world.createJoint(jd);
            }
        }
    }

private:
    PhysicsWorld& m_world;
};

} // namespace ciphyxs
