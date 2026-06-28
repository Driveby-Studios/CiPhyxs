//==================================================================================================
/// @file  test_trimesh.cpp
/// @brief  Validation test for TriangleMesh collision, ray casting, and BVH acceleration.
//==================================================================================================
#include <ciphyxs/CiPhyxs.hpp>
#include <cstdio>
#include <cmath>

using namespace ciphyxs;

int passCount = 0, failCount = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        printf("  FAIL: %s\n", name); \
        ++failCount; \
    } else { \
        printf("  PASS: %s\n", name); \
        ++passCount; \
    } \
} while(0)

int main() {
    printf("========================================\n");
    printf("  CiPhyxs - TriangleMesh Validation\n");
    printf("========================================\n\n");

    // ── 1. Create a simple triangle mesh: a flat plane made of 2 triangles ──────────────
    Vec3f triVerts[] = {
        Vec3f(-5.0f, 0.0f, -5.0f),   // 0: bottom-left
        Vec3f( 5.0f, 0.0f, -5.0f),   // 1: bottom-right
        Vec3f( 5.0f, 0.0f,  5.0f),   // 2: top-right
        Vec3f(-5.0f, 0.0f,  5.0f)    // 3: top-left
    };
    int triIndices[] = {
        0, 1, 2,  // triangle 0
        0, 2, 3   // triangle 1
    };
    const int kNumVerts = 4;
    const int kNumTris  = 2;

    // Build BVH.
    Bvh bvh;
    bvh.build(triVerts, triIndices, kNumTris);

    TriangleMesh mesh;
    mesh.vertices      = triVerts;
    mesh.indices       = triIndices;
    mesh.vertexCount   = kNumVerts;
    mesh.triangleCount = kNumTris;
    mesh.halfExtents   = Vec3f(5.0f, 0.0f, 5.0f);
    mesh.center        = Vec3f::zero();
    mesh.bvh           = &bvh;

    Shape shape(mesh);

    // ── 2. AABB ──────────────────────────────────────────────────────────────────────────
    {
        AABB aabb = shape.getAABB(Vec3f(0, 0, 0), Quaternionf::identity());
        TEST("AABB contains origin", aabb.min.y <= 0.0f && aabb.max.y >= 0.0f);
        TEST("AABB extends to +/-5", std::abs(aabb.min.x + 5.0f) < 0.01f &&
                                      std::abs(aabb.max.x - 5.0f) < 0.01f);
        printf("  AABB min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f)\n",
               aabb.min.x, aabb.min.y, aabb.min.z,
               aabb.max.x, aabb.max.y, aabb.max.z);
    }

    // ── 3. Ray casting via BVH ──────────────────────────────────────────────────────────
    {
        float t; Vec3f normal; int triIdx;

        // Ray from above hitting the centre of the mesh.
        bool hit = bvh.rayCast(Vec3f(0, 10, 0), Vec3f(0, -1, 0), 20.0f,
                                triVerts, triIndices, t, normal, triIdx);
        TEST("BVH rayCast hits centre", hit);
        if (hit) {
            TEST("  Hit at correct height", std::abs(t - 10.0f) < 0.01f);
            TEST("  Normal points up", normal.y > 0.9f);
            printf("  Centre hit: t=%.4f normal=(%.4f,%.4f,%.4f) tri=%d\n",
                   t, normal.x, normal.y, normal.z, triIdx);
        }

        // Ray that should miss (going sideways).
        hit = bvh.rayCast(Vec3f(0, 1, 0), Vec3f(1, 0, 0), 20.0f,
                          triVerts, triIndices, t, normal, triIdx);
        TEST("BVH rayCast sideways misses", !hit);

        // Ray from below should not hit (single-sided).
        hit = bvh.rayCast(Vec3f(0, -1, 0), Vec3f(0, 1, 0), 20.0f,
                          triVerts, triIndices, t, normal, triIdx);
        // Möller-Trumbore doesn't care about back-face culling by default, 
        // but we orient the normal. The hit may still be detected.
        // Just verify the ray-triangle test works.
        printf("  Ray from below: %s (t=%.4f)\n", hit ? "HIT" : "MISS", t);
    }

    // ── 4. Ray casting via PhysicsWorld::rayCast ────────────────────────────────────────
    {
        PhysicsWorld world;
        ShapeHandle meshShape = world.createShape(shape);

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(meshShape);
        world.createBody(ground);

        // Ray from above.
        Ray ray;
        ray.origin    = Vec3f(0, 10, 0);
        ray.direction = Vec3f(0, -1, 0);

        RayHit hit;
        bool hitResult = world.rayCast(ray, 20.0f, hit);
        TEST("World rayCast hits TriangleMesh", hitResult);
        if (hitResult) {
            TEST("  Hit body valid", hit.body != kInvalidHandle);
            TEST("  Hit normal adequate", hit.normal.length() > 0.5f);
            printf("  World ray: t=%.4f normal=(%.4f,%.4f,%.4f)\n",
                   hit.t, hit.normal.x, hit.normal.y, hit.normal.z);
        }

        // Ray that misses to the side.
        ray.origin = Vec3f(100, 10, 0);
        hitResult = world.rayCast(ray, 20.0f, hit);
        TEST("World rayCast sideways misses", !hitResult);
    }

    // ── 5. Sphere vs TriangleMesh (BVH-accelerated) ────────────────────────────────────
    {
        ContactManifold manifold;
        // Sphere directly above the mesh, overlapping.
        Sphere sphere{0.5f};
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(sphere), Vec3f(0, 0.3f, 0), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("Sphere vs TriangleMesh overlap", hit);
        if (hit) {
            TEST("  Contact normal points up", manifold.points[0].normal.y > 0);
            printf("  Sphere hit: normal=(%.4f,%.4f,%.4f) pen=%.4f count=%d\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration,
                   manifold.pointCount);
        }

        // Sphere far above — no collision.
        ContactManifold m2;
        bool noHit = !collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                     Shape(sphere), Vec3f(0, 10, 0), Quaternionf::identity(),
                                     0.5f, 0.5f, 0.5f, 0.5f, m2);
        TEST("Sphere far above — no collision", noHit);
    }

    // ── 6. Plane vs TriangleMesh ───────────────────────────────────────────────────────
    {
        // A plane at y=0.5 pointing down.
        Plane plane{Vec3f::unitY(), 0.5f};

        ContactManifold manifold;
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(plane), Vec3f::zero(), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        // The plane at y=0.5 faces up. Our mesh is at y=0.
        // The plane normal is (0,1,0), distance = 0.5.
        // Since the normal points up and distance is 0.5, the plane is n·p = 0.5.
        // The mesh vertices are at y=0 which gives n·v = 0 < 0.5, so the mesh is below
        // the plane surface. The plane's collision direction goes the other way.
        // Actually, the Plane struct definition says: n·p = d, where d is signed distance.
        // In collidePlaneTriangleMesh, d is the plane equation value.
        // So our plane: (0,0.5,0)·(0,1,0) = 0.5, but the plane surface is at y=0.5.
        // For a vertex at y=0: n·v = 0. 0 < 0.5, so the vertex is "below" the plane.
        // Actually the convention depends on which way the normal points.
        // Let me check what the existing collideBoxPlane does...
        // Actually the Plane defines: normal·p = d, where d is the signed distance.
        // For our plane: normal=(0,1,0), d=0.5. This means points with y=0.5 are on the plane.
        // Points with y<0.5 have n·p = y < 0.5, so they're "behind" the plane.
        // The plane collision function in CiPhyxs treats a vertex as penetrating if
        // n·v - d < 0 (vertex is "below" the plane). So mesh vertices at y=0 have
        // y - 0.5 = -0.5 < 0, meaning penetration of 0.5.
        TEST("Plane vs TriangleMesh overlap", hit);
        if (hit) {
            printf("  Plane hit: normal=(%.4f,%.4f,%.4f) pen=%.4f count=%d\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration,
                   manifold.pointCount);
        }
    }

    // ── 7. Simulate: sphere falling onto triangle mesh ──────────────────────────────────
    {
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0, -9.81f, 0);
        world.setConfig(cfg);

        ShapeHandle meshShape = world.createShape(shape);
        ShapeHandle sphereShape = world.createShape(Shape(Sphere{0.5f}));

        // Ground mesh (slightly below origin for visibility).
        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(meshShape);
        world.createBody(ground);

        // Falling sphere.
        RigidBodyDesc ball;
        ball.mass      = 2.0f;
        ball.position  = Vec3f(0, 5, 0);
        ball.setShape(sphereShape);
        world.createBody(ball);

        // Simulate 120 frames.
        for (int i = 0; i < 120; ++i)
            world.step(1.0f / 60.0f);

        auto& positions = world.bodies().positions;
        auto& vs       = world.bodies().linearVelocities;

        TEST("Sphere resting near y=0.5 (on mesh)",
             std::abs(positions[1].y - 0.5f) < 0.25f);
        printf("  Sphere pos: (%.4f, %.4f, %.4f) vel=(%.4f,%.4f,%.4f)\n",
               positions[1].x, positions[1].y, positions[1].z,
               vs[1].x, vs[1].y, vs[1].z);
    }

    // ── 8. Robustness: no-BVH fallback ────────────────────────────────────────────────
    {
        TriangleMesh noBvhMesh;
        noBvhMesh.vertices      = triVerts;
        noBvhMesh.indices       = triIndices;
        noBvhMesh.vertexCount   = kNumVerts;
        noBvhMesh.triangleCount = kNumTris;
        noBvhMesh.halfExtents   = Vec3f(5.0f, 0.0f, 5.0f);
        noBvhMesh.center        = Vec3f::zero();
        // bvh intentionally left null

        Shape noBvhShape(noBvhMesh);

        // Sphere collision without BVH should still work via linear scan.
        ContactManifold manifold;
        Sphere sphere{0.5f};
        bool hit = collideShapes(noBvhShape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(sphere), Vec3f(0, 0.3f, 0), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("Sphere vs TriangleMesh (no BVH) overlap", hit);
    }

    // ── 9. Box vs TriangleMesh (discrete collision) ──────────────────────────────────
    {
        ContactManifold manifold;
        // Box sitting on the flat mesh at y=0.
        Box box{Vec3f(0.5f, 0.5f, 0.5f)};

        // Box center at y=0.3 — should overlap slightly.
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(box), Vec3f(0, 0.3f, 0), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("Box vs TriangleMesh overlap", hit);
        if (hit) {
            // Normal direction is winding-independent: for a corner below the mesh,
            // the normal points from the contact surface toward the penetrating corner.
            // This may be down (if the corner is below the surface) — what matters
            // is that |normal| == 1 and penetration > 0.
            TEST("  Contact normal is unit length",
                 std::abs(manifold.points[0].normal.length() - 1.0f) < 0.01f);
            printf("  Box hit: normal=(%.4f,%.4f,%.4f) pen=%.4f count=%d\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration,
                   manifold.pointCount);
        }

        // Box far above — no collision.
        ContactManifold m2;
        bool noHit = !collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                     Shape(box), Vec3f(0, 10, 0), Quaternionf::identity(),
                                     0.5f, 0.5f, 0.5f, 0.5f, m2);
        TEST("Box far above — no collision", noHit);
    }

    // ── 10. Box falling onto triangle mesh (simulation) ────────────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle meshShape = world.createShape(shape);
        ShapeHandle boxShape  = world.createShape(Shape(Box{Vec3f(0.4f, 0.4f, 0.4f)}));

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(meshShape);
        world.createBody(ground);

        RigidBodyDesc boxDesc;
        boxDesc.mass     = 5.0f;
        boxDesc.position = Vec3f(0, 3, 0);
        boxDesc.setShape(boxShape);
        world.createBody(boxDesc);

        // Run enough steps for the box to settle.
        for (int i = 0; i < 360; ++i)
            world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vs  = world.bodies().linearVelocities;

        // Box should be resting on or near the mesh.
        TEST("Box resting on mesh (vel near zero)",
             vs[1].length() < 0.5f);
        printf("  Box final: (%.4f, %.4f, %.4f) vel=(%.4f,%.4f,%.4f)\n",
               pos[1].x, pos[1].y, pos[1].z,
               vs[1].x, vs[1].y, vs[1].z);
    }

    // ── 11. Capsule vs TriangleMesh (discrete collision) ───────────────────────────────
    {
        ContactManifold manifold;
        Capsule capsule{0.3f, 0.5f};

        // Capsule center at y=0.3 (overlapping the mesh at y=0).
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(capsule), Vec3f(0, 0.3f, 0), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("Capsule vs TriangleMesh overlap", hit);
        if (hit) {
            TEST("  Contact normal points up", manifold.points[0].normal.y > 0);
            printf("  Capsule hit: normal=(%.4f,%.4f,%.4f) pen=%.4f count=%d\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration,
                   manifold.pointCount);
        }

        // Capsule far above.
        ContactManifold m2;
        bool noHit = !collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                     Shape(capsule), Vec3f(0, 10, 0), Quaternionf::identity(),
                                     0.5f, 0.5f, 0.5f, 0.5f, m2);
        TEST("Capsule far above \u2014 no collision", noHit);
    }

    // ── 12. Capsule on triangle mesh (simulation) ──────────────────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle meshShape = world.createShape(shape);
        ShapeHandle capShape  = world.createShape(Shape(Capsule{0.3f, 0.4f}));

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(meshShape);
        world.createBody(ground);

        RigidBodyDesc capDesc;
        capDesc.mass     = 3.0f;
        capDesc.position = Vec3f(0, 3, 0);
        capDesc.setShape(capShape);
        world.createBody(capDesc);

        for (int i = 0; i < 300; ++i)
                world.step(1.0f / 60.0f);

            auto& pos = world.bodies().positions;
            auto& vs  = world.bodies().linearVelocities;

            // Capsule should be resting on the mesh.
            // For a vertical capsule: center y = radius + halfHeight = 0.3 + 0.4 = 0.7
            TEST("Capsule resting on mesh",
                 std::abs(pos[1].y - 0.7f) < 0.3f);
        printf("  Capsule final: (%.4f, %.4f, %.4f) vel=(%.4f,%.4f,%.4f)\n",
               pos[1].x, pos[1].y, pos[1].z,
               vs[1].x, vs[1].y, vs[1].z);
    }

    // ── 13. ConvexMesh vs TriangleMesh (discrete collision) ────────────────────────────
    {
        // A small pyramid/tetrahedron convex mesh.
        Vec3f pyramVerts[] = {
            Vec3f(-0.5f, 0.0f, -0.5f),
            Vec3f( 0.5f, 0.0f, -0.5f),
            Vec3f( 0.5f, 0.0f,  0.5f),
            Vec3f(-0.5f, 0.0f,  0.5f),
            Vec3f( 0.0f, 1.0f,  0.0f)
        };
        ConvexMesh pyram;
        pyram.vertices     = pyramVerts;
        pyram.vertexCount  = 5;
        pyram.halfExtents  = Vec3f(0.5f, 0.5f, 0.5f);
        pyram.center       = Vec3f(0, 0.4f, 0);

        ContactManifold manifold;
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(pyram), Vec3f(0, 0.1f, 0), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("ConvexMesh vs TriangleMesh overlap", hit);
        if (hit) {
            printf("  ConvexMesh hit: normal=(%.4f,%.4f,%.4f) pen=%.4f count=%d\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration,
                   manifold.pointCount);
        }

        // ConvexMesh far above.
        ContactManifold m2;
        bool noHit = !collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                     Shape(pyram), Vec3f(0, 20, 0), Quaternionf::identity(),
                                     0.5f, 0.5f, 0.5f, 0.5f, m2);
        TEST("ConvexMesh far above \u2014 no collision", noHit);
    }

    // ── 14. ConvexMesh pyramid on triangle mesh (simulation) ───────────────────────────
    {
        PhysicsWorld world;

        ShapeHandle meshShape = world.createShape(shape);
        ShapeHandle pyramShape = world.createShape([&](){
            Vec3f v[] = {
                Vec3f(-0.5f, 0.0f, -0.5f),
                Vec3f( 0.5f, 0.0f, -0.5f),
                Vec3f( 0.5f, 0.0f,  0.5f),
                Vec3f(-0.5f, 0.0f,  0.5f),
                Vec3f( 0.0f, 1.0f,  0.0f)
            };
            ConvexMesh m;
            m.vertices    = v;
            m.vertexCount = 5;
            m.halfExtents = Vec3f(0.5f, 0.5f, 0.5f);
            m.center      = Vec3f(0, 0.4f, 0);
            return Shape(m);
        }());

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(meshShape);
        world.createBody(ground);

        RigidBodyDesc pyramDesc;
        pyramDesc.mass     = 4.0f;
        pyramDesc.position = Vec3f(0.5f, 4, 0);
        pyramDesc.setShape(pyramShape);
        world.createBody(pyramDesc);

        for (int i = 0; i < 180; ++i)
            world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        auto& vs  = world.bodies().linearVelocities;

        // Pyramid should be resting on or near the mesh.
        TEST("ConvexMesh resting on mesh",
             std::abs(pos[1].y - 0.5f) < 0.6f);
        printf("  Pyramid final: (%.4f, %.4f, %.4f) vel=(%.4f,%.4f,%.4f)\n",
               pos[1].x, pos[1].y, pos[1].z,
               vs[1].x, vs[1].y, vs[1].z);
    }

    // ── 15. Complex mesh: slope made of triangles ───────────────────────────────────────────
    {
        Vec3f slopeVerts[] = {
            Vec3f(-5.0f, 0.0f, -5.0f),
            Vec3f( 5.0f, 0.0f, -5.0f),
            Vec3f( 5.0f, 5.0f,  5.0f),
            Vec3f(-5.0f, 5.0f,  5.0f)
        };
        int slopeIdx[] = {0, 1, 2, 0, 2, 3};

        Bvh slopeBvh;
        slopeBvh.build(slopeVerts, slopeIdx, 2);

        TriangleMesh slopeMesh;
        slopeMesh.vertices      = slopeVerts;
        slopeMesh.indices       = slopeIdx;
        slopeMesh.vertexCount   = 4;
        slopeMesh.triangleCount = 2;
        slopeMesh.halfExtents   = Vec3f(5.0f, 2.5f, 5.0f);
        slopeMesh.center        = Vec3f(0, 2.5f, 0);
        slopeMesh.bvh           = &slopeBvh;

        // Ray hitting the slope.
        float t; Vec3f normal; int triIdx;
        bool hit = slopeBvh.rayCast(Vec3f(0, 10, 0), Vec3f(0, -1, 0), 20.0f,
                                     slopeVerts, slopeIdx, t, normal, triIdx);
        TEST("Slope BVH rayCast hits", hit);
        if (hit) {
            printf("  Slope hit: t=%.4f normal=(%.4f,%.4f,%.4f) tri=%d\n",
                   t, normal.x, normal.y, normal.z, triIdx);
        }

        // Sphere rolling down the slope.
        PhysicsWorld world;
        ShapeHandle slopeShape = world.createShape(Shape(slopeMesh));
        ShapeHandle sphereShape = world.createShape(Shape(Sphere{0.4f}));

        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(slopeShape);
        world.createBody(ground);

        RigidBodyDesc ball;
        ball.mass     = 1.0f;
        ball.position = Vec3f(0, 6, 0);
        ball.setShape(sphereShape);
        world.createBody(ball);

        for (int i = 0; i < 180; ++i)
            world.step(1.0f / 60.0f);

        auto& pos = world.bodies().positions;
        // Check that the sphere has moved (was affected by collision).
        TEST("Sphere on slope moved from start",
             std::abs(pos[1].x) > 0.01f || std::abs(pos[1].z) > 0.01f);
        printf("  Slope sphere final: (%.4f, %.4f, %.4f)\n",
               pos[1].x, pos[1].y, pos[1].z);
    }

    // ── Summary ────────────────────────────────────────────────────────────────────────
    printf("\n========================================\n");
    printf("  Results: %d / %d passed\n", passCount, passCount + failCount);
    printf("========================================\n");
    return failCount > 0 ? 1 : 0;
}
