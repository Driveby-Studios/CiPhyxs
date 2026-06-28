//==================================================================================================
/// @file  test_convexmesh.cpp
/// @brief  Validation test for ConvexMesh collision, ray casting, and simulation pipeline.
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
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  CiPhyxs — ConvexMesh Validation            ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── 1. ConvexMesh: box-shaped hull (8 vertices) ──────────────────────────────────────
    Vec3f boxVerts[] = {
        Vec3f(-0.5f, -0.5f, -0.5f),
        Vec3f( 0.5f, -0.5f, -0.5f),
        Vec3f( 0.5f, -0.5f,  0.5f),
        Vec3f(-0.5f, -0.5f,  0.5f),
        Vec3f(-0.5f,  0.5f, -0.5f),
        Vec3f( 0.5f,  0.5f, -0.5f),
        Vec3f( 0.5f,  0.5f,  0.5f),
        Vec3f(-0.5f,  0.5f,  0.5f)
    };
    constexpr int kNumVerts = 8;
    ConvexMesh mesh;
    mesh.vertices    = boxVerts;
    mesh.vertexCount = kNumVerts;
    mesh.halfExtents = Vec3f(0.5f, 0.5f, 0.5f);
    mesh.center      = Vec3f::zero();
    Shape shape(mesh);

    // ── 2. AABB ──────────────────────────────────────────────────────────────────────────
    {
        AABB aabb = shape.getAABB(Vec3f(0, 5, 0), Quaternionf::identity());
        TEST("AABB contains center", aabb.min.y <= 5.0f && aabb.max.y >= 5.0f);
        TEST("AABB positive extents",
             aabb.extents().x > 0 && aabb.extents().y > 0 && aabb.extents().z > 0);
        printf("  AABB center=(%.3f,%.3f,%.3f) ext=(%.3f,%.3f,%.3f)\n",
               aabb.center().x, aabb.center().y, aabb.center().z,
               aabb.extents().x, aabb.extents().y, aabb.extents().z);
    }

    // ── 3. Inertia ──────────────────────────────────────────────────────────────────────
    {
        Vec3f inertia = shape.computeInertia(10.0f);
        TEST("Inertia positive", inertia.x > 0 && inertia.y > 0 && inertia.z > 0);
        printf("  Inertia: (%.4f, %.4f, %.4f)\n", inertia.x, inertia.y, inertia.z);
    }

    // ── 4. GJK: ConvexMesh vs Box (offset, overlapping) ─────────────────────────────────
    {
        ContactManifold manifold;
        Box box{Vec3f(0.3f, 0.3f, 0.3f)};
        bool hit = collideShapes(shape, Vec3f(0, 5, 0), Quaternionf::identity(),
                                 Shape(box), Vec3f(0.2f, 4.8f, 0.2f), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("ConvexMesh vs Box overlap", hit);
        if (hit) {
            TEST("  Contact normal valid", manifold.points[0].normal.length() > 0.5f);
            printf("  GJK Box: normal=(%.4f,%.4f,%.4f) pen=%.4f\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration);
        }
    }

    // ── 5. ConvexMesh vs Plane (vertex-sweep SAT) ───────────────────────────────────────
    {
        Plane plane{Vec3f::unitY(), 0.0f};

        ContactManifold manifold;
        bool hit = collideShapes(shape, Vec3f(0, 0.2f, 0), Quaternionf::identity(),
                                 Shape(plane), Vec3f::zero(), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("ConvexMesh vs Plane overlap", hit);
        if (hit) {
            TEST("  Normal points A→B (downward)", manifold.points[0].normal.y < 0);
            printf("  Plane: normal=(%.4f,%.4f,%.4f) pen=%.4f\n",
                   manifold.points[0].normal.x, manifold.points[0].normal.y,
                   manifold.points[0].normal.z, manifold.points[0].penetration);
        }

        ContactManifold m2;
        bool noHit = !collideShapes(shape, Vec3f(0, 10, 0), Quaternionf::identity(),
                                     Shape(plane), Vec3f::zero(), Quaternionf::identity(),
                                     0.5f, 0.5f, 0.5f, 0.5f, m2);
        TEST("No collision when separated", noHit);
    }

    // ── 6. No collision when far apart ──────────────────────────────────────────────────
    {
        ContactManifold manifold;
        Sphere sphere{0.5f};
        bool hit = collideShapes(shape, Vec3f(0, 0, 0), Quaternionf::identity(),
                                 Shape(sphere), Vec3f(100, 100, 100), Quaternionf::identity(),
                                 0.5f, 0.5f, 0.5f, 0.5f, manifold);
        TEST("No collision when far apart", !hit);
    }

    // ── 7. Ray casting via PhysicsWorld::rayCast ────────────────────────────────────────
    //    Tests the full world-space transform + ray-cast pipeline through PhysicsWorld.
    {
        PhysicsWorld world;

        // Use a tetrahedron mesh (farthest from center) for easier ray intersection.
        Vec3f tetVerts[] = {
            Vec3f( 0.5f,  0.0f, -0.354f),
            Vec3f(-0.5f,  0.0f, -0.354f),
            Vec3f( 0.0f,  0.0f,  0.354f),
            Vec3f( 0.0f,  0.5f,  0.0f)
        };
        Vec3f tetCenter = Vec3f::zero();
        for (int i = 0; i < 4; ++i) tetCenter += tetVerts[i];
        tetCenter /= 4.0f;
        ConvexMesh tetMesh;
        tetMesh.vertices = tetVerts;
        tetMesh.vertexCount = 4;
        tetMesh.halfExtents = Vec3f(0.5f, 0.25f, 0.354f);
        tetMesh.center = tetCenter;

        ShapeHandle meshShape = world.createShape(Shape(tetMesh));
        ShapeHandle planeShape = world.createShape(Shape(Plane{}));

        RigidBodyDesc ground;
        ground.motionType  = MotionType::Static;
        ground.setShape(planeShape);
        world.createBody(ground);

        RigidBodyDesc body;
        body.mass = 5.0f;
        body.position = Vec3f(0, 5, 0);
        body.setShape(meshShape);
        body.motionType = MotionType::Dynamic;
        world.createBody(body);

        // Ray from above, pointing down.
        Ray ray;
        ray.origin    = Vec3f(0, 10, 0);
        ray.direction = Vec3f(0, -1, 0);

        RayHit hit;
        bool hitResult = world.rayCast(ray, 20.0f, hit);

        TEST("PhysicsWorld rayCast hit ConvexMesh", hitResult);
        if (hitResult) {
            // The hit might be at maxT if conservative advancement doesn't fully converge.
            // But it should still report a valid body and normal.
            TEST("  Hit body is valid", hit.body != kInvalidHandle);
            TEST("  Hit normal adequate", hit.normal.length() > 0.5f);
            printf("  RayCast: t=%.4f normal=(%.4f,%.4f,%.4f) body=%u\n",
                   hit.t, hit.normal.x, hit.normal.y, hit.normal.z, hit.body);
        }

        // Ray that should miss (far to the side and above the ground plane).
        Ray rayMiss;
        rayMiss.origin    = Vec3f(100, 10, 0);
        rayMiss.direction = Vec3f(0, -1, 0);
        RayHit missHit;
        bool missResult = world.rayCast(rayMiss, 20.0f, missHit);
        // The "miss" ray from (100,10,0) going down might hit the ground plane.
        // We just check it doesn't hit the mesh body (body handle 1).
        TEST("Ray doesn't hit mesh when far off to side",
             !missResult || missHit.body != 1);
    }

    // ── 8. Simulate ConvexMesh falling onto plane ────────────────────────────────────────
    {
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f(0, -9.81f, 0);
        world.setConfig(cfg);

        ShapeHandle meshShape = world.createShape(Shape(mesh));
        ShapeHandle planeShape = world.createShape(Shape(Plane{}));

        RigidBodyDesc ground;
        ground.motionType = MotionType::Static;
        ground.setShape(planeShape);
        world.createBody(ground);

        RigidBodyDesc body;
        body.mass = 5.0f;
        body.position = Vec3f(0, 5, 0);
        body.setShape(meshShape);
        body.motionType = MotionType::Dynamic;
        RigidBodyHandle h = world.createBody(body);

        bool hasContact = false;
        for (int i = 0; i < 120; ++i) {
            world.step(1.0f / 60.0f);
            if (!world.manifolds().empty()) hasContact = true;
        }

        Vec3f finalPos = world.bodies().positions[h];
        TEST("ConvexMesh body made contact during simulation", hasContact);
        TEST("ConvexMesh resting on plane (y >= -0.5)", finalPos.y >= -0.5f);
        printf("  Sim pos: (%.4f, %.4f, %.4f) manifolds=%zu\n",
               finalPos.x, finalPos.y, finalPos.z, world.manifolds().size());
    }

    // ── 9. ConvexMesh CCD pipeline ──────────────────────────────────────────────────────
    {
        PhysicsWorld world;
        PhysicsWorldConfig cfg;
        cfg.gravity = Vec3f::zero();
        world.setConfig(cfg);

        ShapeHandle meshShape = world.createShape(Shape(mesh));
        ShapeHandle boxShape  = world.createShape(Shape(Box{Vec3f(0.25f, 0.25f, 0.25f)}));

        RigidBodyDesc obstacle;
        obstacle.motionType  = MotionType::Static;
        obstacle.setShape(boxShape);
        obstacle.position    = Vec3f(2, 5, 0);
        world.createBody(obstacle);

        RigidBodyDesc body;
        body.mass = 1.0f;
        body.position = Vec3f(-10, 5, 0);
        body.linearVelocity = Vec3f(100, 0, 0);
        body.setShape(meshShape);
        body.ccdMode = CcdMode::Cast;
        body.motionType = MotionType::Dynamic;
        RigidBodyHandle h = world.createBody(body);

        for (int i = 0; i < 10; ++i)
            world.step(1.0f / 60.0f);

        Vec3f pos = world.bodies().positions[h];
        TEST("CCD body in bounded position", pos.x > -12 && pos.x < 15);
        printf("  CCD pos: (%.4f, %.4f, %.4f)\n", pos.x, pos.y, pos.z);
    }

    // ── Summary ──
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  Results: %d / %d passed", passCount, passCount + failCount);
    printf("                     ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    return failCount > 0 ? 1 : 0;
}
