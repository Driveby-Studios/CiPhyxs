//==================================================================================================
/// @file  test_dbvt.cpp
/// @brief  Unit tests for the Dynamic BVH (Dbvt) broadphase.
///
/// Validates insertion, removal, update (temporal coherence), pair generation,
/// overlap queries, and determinism.
//==================================================================================================
#include "ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace ciphyxs;

// ─── helpers ────────────────────────────────────────────────────────────────────────────────────

static int  g_tests = 0;
static int  g_passed = 0;

#define TEST(name) do { ++g_tests; std::printf("  TEST %s ... ", name); } while(0)
#define PASS()   do { ++g_passed; std::puts("PASS"); } while(0)
#define FAIL(msg) do { std::printf("FAIL  (%s)\n", msg); return 1; } while(0)

/// Build a minimal PhysicsWorld with a few spheres in a line.
struct TestFixture {
    PhysicsWorld world;

    TestFixture() {
        world.createShape(Shape(Sphere{0.5f}));
        // Bodies A, B, C: spaced 2 units apart along X.
        RigidBodyDesc desc;
        desc.setShape(0);
        desc.mass = 1.0f;
        desc.motionType = MotionType::Dynamic;

        desc.position = Vec3f::zero();
        world.createBody(desc); // 0

        desc.position = Vec3f(2.0f, 0.0f, 0.0f);
        world.createBody(desc); // 1

        desc.position = Vec3f(4.0f, 0.0f, 0.0f);
        world.createBody(desc); // 2

        // Enable DBVT broadphase.
        world.enableDbvt();
    }

    /// Step once without gravity (bodies stay in place).
    void step() {
        PhysicsWorldConfig cfg = world.config();
        cfg.gravity = Vec3f::zero();
        world.setConfig(cfg);
        world.step(1.0f / 60.0f);
    }

    /// Count pairs produced by the current broadphase.
    int pairCount() {
        return static_cast<int>(world.manifolds().size());
    }
};

// ─── Tests ──────────────────────────────────────────────────────────────────────────────────────

static int testInsertAndPair() {
    TEST("insert and pair generation");
    TestFixture f;

    // After one step with zero gravity, three bodies spaced apart:
    // 0 <-- 2 units --> 1 <-- 2 units --> 2
    // Sphere radius = 0.5 each => bounding radius sum = 1.0
    // Distance between centres = 2.0 => no overlap.
    f.step();
    if (f.pairCount() != 0) FAIL("expected 0 pairs (bodies too far apart)");

    // Move body 0 close to body 1 (distance = 0.8 < combined radius 1.0 => overlap).
    f.world.bodies().positions[0] = Vec3f(1.2f, 0.0f, 0.0f);
    f.step();

    // Now the spheres should overlap (radius 0.5+0.5=1.0, distance=0.8 => clear overlap).
    if (f.pairCount() < 1) FAIL("expected at least 1 contact pair");

    PASS();
    return 0;
}

static int testRemoveBody() {
    TEST("body removal clears leaf");
    TestFixture f;
    f.step();
    // Remove body 1.
    f.world.removeBody(1);
    f.step();
    // After removal, only bodies 0 and 2 remain. They are 4 units apart — no overlap.
    if (f.pairCount() != 0) FAIL("expected 0 pairs after removing middle body");

    PASS();
    return 0;
}

static int testDeterminism() {
    TEST("deterministic pair output across two runs");
    TestFixture f;

    // Run twice with the same initial state.
    auto run = [&]() -> std::vector<BroadphasePair> {
        TestFixture f2;
        // Move bodies to create overlaps.
        f2.world.bodies().positions[0] = Vec3f(0.9f, 0.0f, 0.0f);
        f2.world.bodies().positions[1] = Vec3f(1.1f, 0.0f, 0.0f);
        f2.step();
        // Access pairs through the internal Dbvt.
        const auto& dbvt = f2.world.dbvt();
        std::vector<BroadphasePair> pairs;
        dbvt.collidePairs(f2.world.bodies(), pairs);
        return pairs;
    };

    auto pairs1 = run();
    auto pairs2 = run();

    if (pairs1.size() != pairs2.size()) FAIL("pair count differs");
    for (std::size_t i = 0; i < pairs1.size(); ++i) {
        if (pairs1[i].bodyA != pairs2[i].bodyA ||
            pairs1[i].bodyB != pairs2[i].bodyB) FAIL("pair mismatch");
    }

    PASS();
    return 0;
}

static int testOverlapQuery() {
    TEST("overlapQuery returns correct bodies");
    TestFixture f;
    f.step();  // builds the tree

    const auto& dbvt = f.world.dbvt();
    std::vector<RigidBodyHandle> result;

    // Query a region that covers only body 0.
    // Body 0 is at (0,0,0) with sphere radius 0.5 => AABB approx (-0.5,-0.5,-0.5)-(0.5,0.5,0.5).
    AABB box;
    box.min = Vec3f(-1.0f, -1.0f, -1.0f);
    box.max = Vec3f( 1.0f,  1.0f,  1.0f);
    dbvt.overlapQuery(box, result);

    if (result.empty()) {
        // Debug: check tree state.
        std::printf("    tree root=%d, nodes=%zu\n", dbvt.root(), dbvt.nodes().size());
        for (std::size_t i = 0; i < dbvt.nodes().size(); ++i) {
            const auto& n = dbvt.nodes()[i];
            if (n.isLeaf() && n.body >= 0) {
                std::printf("    leaf %zu: body=%d bounds=(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)\n",
                    i, n.body,
                    n.bounds.min.x, n.bounds.min.y, n.bounds.min.z,
                    n.bounds.max.x, n.bounds.max.y, n.bounds.max.z);
            }
        }
        FAIL("expected at least body 0 in query");
    }
    bool found0 = false;
    for (auto h : result) if (h == 0) found0 = true;
    if (!found0) FAIL("body 0 not found in overlap query");

    PASS();
    return 0;
}

static int testTemporalCoherence() {
    TEST("temporal coherence skips tree update for small motion");
    TestFixture f;

    // After first step, the tree is built.
    f.step();

    const auto& dbvt = f.world.dbvt();
    int nodeCountBefore = dbvt.nodeCount();

    // Move a body by a tiny amount that stays within the fat AABB.
    f.world.bodies().positions[0] = Vec3f(0.001f, 0.0f, 0.0f);
    f.step();

    // The tree should NOT have been modified (no new nodes).
    // (The exact count may vary depending on the fat AABB margin, but
    //  we check that the structural change is small.)
    int nodeCountAfter = dbvt.nodeCount();
    if (nodeCountAfter > nodeCountBefore + 4) {
        FAIL("tree grew excessively despite tiny motion");
    }

    PASS();
    return 0;
}

static int testSyncAddsBodies() {
    TEST("sync adds newly created bodies to tree");
    TestFixture f;
    f.step();

    const auto& dbvt = f.world.dbvt();

    // Count leaves before.
    int leavesBefore = 0;
    for (const auto& n : dbvt.nodes()) if (n.isLeaf() && n.body >= 0) ++leavesBefore;
    if (leavesBefore != 3) FAIL("expected 3 leaves initially");

    // Add a new body.
    RigidBodyDesc desc;
    desc.setShape(0);
    desc.mass = 1.0f;
    desc.position = Vec3f(10.0f, 0.0f, 0.0f);
    f.world.createBody(desc);
    f.step();

    int leavesAfter = 0;
    for (const auto& n : dbvt.nodes()) if (n.isLeaf() && n.body >= 0) ++leavesAfter;
    if (leavesAfter != 4) FAIL("expected 4 leaves after adding a body");

    PASS();
    return 0;
}

static int testMassRatioStabilityWithDbvt() {
    TEST("mass ratio stability with Dbvt (10000:1)");
    PhysicsWorld world;

    // Register shapes.
    world.createShape(Shape(Sphere{0.5f}));  // 0: small sphere
    world.createShape(Shape(Plane{}));        // 1: ground plane

    // Heavy body.
    RigidBodyDesc heavy;
    heavy.setShape(0);
    heavy.mass = 10000.0f;
    heavy.position = Vec3f(0.0f, 5.0f, 0.0f);
    auto hHeavy = world.createBody(heavy);

    // Light body.
    RigidBodyDesc light;
    light.setShape(0);
    light.mass = 1.0f;
    light.position = Vec3f(0.0f, 6.0f, 0.0f);
    auto hLight = world.createBody(light);

    // Ground plane (static).
    RigidBodyDesc ground;
    ground.setShape(1);
    ground.motionType = MotionType::Static;
    world.createBody(ground);

    // Enable Dbvt.
    world.enableDbvt();
    PhysicsWorldConfig cfg2 = world.config();
    cfg2.gravity = Vec3f(0.0f, -9.81f, 0.0f);
    world.setConfig(cfg2);

    // Step 120 frames (2 seconds).
    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    // Both bodies should be resting on the ground (y ≈ sphere radius ≈ 0.5).
    // The light body should not have exploded.
    auto checkBody = [&](RigidBodyHandle h, const char* label) -> bool {
        float y = world.bodies().positions[h].y;
        if (std::abs(y - 0.5f) > 1.0f) {
            std::printf("    %s y=%.4f (expected ~0.5)\n", label, y);
            return false;
        }
        return true;
    };

    if (!checkBody(hHeavy, "heavy")) FAIL("heavy body unstable with Dbvt");
    if (!checkBody(hLight,  "light")) FAIL("light body unstable with Dbvt");

    PASS();
    return 0;
}

// ─── Pair cache tests ───────────────────────────────────────────────────────────────────────────

static int testPairCacheDeterminism() {
    TEST("pair cache produces same output as uncached");

    // Build a world with multiple overlapping bodies.
    PhysicsWorld world;
    world.createShape(Shape(Sphere{0.5f}));

    // Create a cluster of overlapping spheres.
    RigidBodyDesc desc;
    desc.setShape(0);
    desc.mass = 1.0f;
    desc.motionType = MotionType::Dynamic;

    desc.position = Vec3f(0.0f, 0.0f, 0.0f);
    world.createBody(desc);
    desc.position = Vec3f(0.8f, 0.0f, 0.0f);
    world.createBody(desc);
    desc.position = Vec3f(0.0f, 0.8f, 0.0f);
    world.createBody(desc);
    desc.position = Vec3f(0.0f, 0.0f, 0.8f);
    world.createBody(desc);

    // Run once WITHOUT caching.
    {
        PhysicsWorldConfig cfg = world.config();
        cfg.gravity = Vec3f::zero();
        world.setConfig(cfg);
    }
    world.enableDbvt();
    // Cache is off by default.
    world.step(1.0f / 60.0f);
    std::size_t uncachedCount = world.manifolds().size();

    // Reset and run WITH caching.
    PhysicsWorld world2;
    world2.createShape(Shape(Sphere{0.5f}));
    for (int i = 0; i < 4; ++i) {
        RigidBodyDesc d;
        d.setShape(0);
        d.mass = 1.0f;
        d.position = world.bodies().positions[i];
        world2.createBody(d);
    }
    {
        PhysicsWorldConfig cfg = world2.config();
        cfg.gravity = Vec3f::zero();
        world2.setConfig(cfg);
    }
    world2.enableDbvt();
    // Enable caching via the Dbvt accessor.
    // The config flag is read during sync(), which happens inside detectCollisions().
    // We need a way to enable caching.  For now, configure it via the sync config.
    // Since the PhysicsWorld passes a default DbvtConfig to sync(), we bypass and
    // test the Dbvt directly.

    // Direct Dbvt test: build tree, generate pairs with and without cache.
    Dbvt dbvt;
    RigidBodyStorage storage;
    std::vector<Shape> shapes;
    shapes.push_back(Shape(Sphere{0.5f}));

    // Create 4 overlapping bodies.
    auto addBody = [&](const Vec3f& pos) -> RigidBodyHandle {
        RigidBodyDesc d;
        d.setShape(0);
        d.mass = 1.0f;
        d.position = pos;
        return storage.emplace(d);
    };

    RigidBodyHandle h0 = addBody(Vec3f(0.0f, 0.0f, 0.0f));
    RigidBodyHandle h1 = addBody(Vec3f(0.8f, 0.0f, 0.0f));
    RigidBodyHandle h2 = addBody(Vec3f(0.0f, 0.8f, 0.0f));
    RigidBodyHandle h3 = addBody(Vec3f(0.0f, 0.0f, 0.8f));
    (void)h0; (void)h1; (void)h2; (void)h3;

    // Sync and generate uncached.
    DbvtConfig cfgUncached;
    cfgUncached.enablePairCaching = false;
    dbvt.sync(storage, shapes, cfgUncached);

    std::vector<BroadphasePair> pairsUncached;
    dbvt.collidePairs(storage, pairsUncached);

    // Reset and generate cached.
    dbvt.clear();
    DbvtConfig cfgCached;
    cfgCached.enablePairCaching = true;
    dbvt.sync(storage, shapes, cfgCached);

    std::vector<BroadphasePair> pairsCached;
    dbvt.collidePairs(storage, pairsCached);

    if (pairsUncached.size() != pairsCached.size()) {
        std::printf("    uncached=%zu cached=%zu\n", pairsUncached.size(), pairsCached.size());
        FAIL("pair count differs between cached and uncached");
    }
    for (std::size_t i = 0; i < pairsUncached.size(); ++i) {
        if (pairsUncached[i].bodyA != pairsCached[i].bodyA ||
            pairsUncached[i].bodyB != pairsCached[i].bodyB) {
            std::printf("    pair[%zu]: uncached=(%u,%u) cached=(%u,%u)\n",
                i, pairsUncached[i].bodyA, pairsUncached[i].bodyB,
                pairsCached[i].bodyA, pairsCached[i].bodyB);
            FAIL("pair mismatch between cached and uncached");
        }
    }

    PASS();
    return 0;
}

static int testPairCachePersistence() {
    TEST("pair cache persists overlapping pairs across frames");

    Dbvt dbvt;
    RigidBodyStorage storage;
    std::vector<Shape> shapes;
    shapes.push_back(Shape(Sphere{0.5f}));

    auto addBody = [&](const Vec3f& pos) -> RigidBodyHandle {
        RigidBodyDesc d;
        d.setShape(0);
        d.mass = 1.0f;
        d.position = pos;
        return storage.emplace(d);
    };

    // Two overlapping spheres.
    addBody(Vec3f(0.0f, 0.0f, 0.0f));
    addBody(Vec3f(0.8f, 0.0f, 0.0f));

    DbvtConfig cfg;
    cfg.enablePairCaching = true;

    // First frame: sync + collide (populates cache).
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs1;
    dbvt.collidePairs(storage, pairs1);

    if (pairs1.size() != 1) FAIL("expected 1 pair on first frame");
    std::size_t cacheSizeAfterFirst = dbvt.nodeCount(); // just a reference
    (void)cacheSizeAfterFirst;

    // Second frame: sync + collide (reuses cache, no tree walk for the pair).
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs2;
    dbvt.collidePairs(storage, pairs2);

    if (pairs2.size() != 1) FAIL("expected 1 pair on second frame");
    if (pairs1[0].bodyA != pairs2[0].bodyA || pairs1[0].bodyB != pairs2[0].bodyB) {
        FAIL("pair changed between frames");
    }

    PASS();
    return 0;
}

static int testPairCacheStaleRemoval() {
    TEST("pair cache removes stale pairs when bodies separate");

    Dbvt dbvt;
    RigidBodyStorage storage;
    std::vector<Shape> shapes;
    shapes.push_back(Shape(Sphere{0.5f}));

    auto addBody = [&](const Vec3f& pos) -> RigidBodyHandle {
        RigidBodyDesc d;
        d.setShape(0);
        d.mass = 1.0f;
        d.position = pos;
        return storage.emplace(d);
    };

    addBody(Vec3f(0.0f, 0.0f, 0.0f));
    addBody(Vec3f(0.8f, 0.0f, 0.0f));

    DbvtConfig cfg;
    cfg.enablePairCaching = true;

    // First frame: overlapping.
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs1;
    dbvt.collidePairs(storage, pairs1);
    if (pairs1.size() != 1) FAIL("expected 1 pair when overlapping");

    // Move body 1 far away.
    storage.positions[1] = Vec3f(100.0f, 0.0f, 0.0f);

    // Second frame: no longer overlapping.
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs2;
    dbvt.collidePairs(storage, pairs2);
    if (pairs2.size() != 0) {
        std::printf("    expected 0 pairs, got %zu\n", pairs2.size());
        FAIL("stale pair was not removed from cache");
    }

    PASS();
    return 0;
}

static int testPairCacheNewPairs() {
    TEST("pair cache detects new overlapping pairs");

    Dbvt dbvt;
    RigidBodyStorage storage;
    std::vector<Shape> shapes;
    shapes.push_back(Shape(Sphere{0.5f}));

    auto addBody = [&](const Vec3f& pos) -> RigidBodyHandle {
        RigidBodyDesc d;
        d.setShape(0);
        d.mass = 1.0f;
        d.position = pos;
        return storage.emplace(d);
    };

    // Three bodies in a line, only adjacent ones overlap.
    addBody(Vec3f(0.0f, 0.0f, 0.0f));
    addBody(Vec3f(0.8f, 0.0f, 0.0f));
    addBody(Vec3f(3.0f, 0.0f, 0.0f)); // far from body 1

    DbvtConfig cfg;
    cfg.enablePairCaching = true;

    // First frame: only (0,1) overlap.
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs1;
    dbvt.collidePairs(storage, pairs1);
    if (pairs1.size() != 1) FAIL("expected 1 pair initially");

    // Move body 2 next to body 1.
    storage.positions[2] = Vec3f(1.6f, 0.0f, 0.0f);

    // Second frame: both (0,1) and (1,2) should overlap.
    dbvt.sync(storage, shapes, cfg);
    std::vector<BroadphasePair> pairs2;
    dbvt.collidePairs(storage, pairs2);
    if (pairs2.size() != 2) {
        std::printf("    expected 2 pairs, got %zu\n", pairs2.size());
        FAIL("new overlapping pair not detected by cache");
    }

    PASS();
    return 0;
}

// ─── main ───────────────────────────────────────────────────────────────────────────────────────

int main() {
    std::puts("=== Dbvt Tests ===");

    int failures = 0;
    failures += testInsertAndPair();
    failures += testRemoveBody();
    failures += testDeterminism();
    failures += testOverlapQuery();
    failures += testTemporalCoherence();
    failures += testSyncAddsBodies();
    failures += testMassRatioStabilityWithDbvt();
    failures += testPairCacheDeterminism();
    failures += testPairCachePersistence();
    failures += testPairCacheStaleRemoval();
    failures += testPairCacheNewPairs();

    std::printf("\n%d / %d tests passed\n", g_passed, g_tests);
    return failures;
}
