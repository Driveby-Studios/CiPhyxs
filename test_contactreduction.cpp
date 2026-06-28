//==================================================================================================
/// @file  test_contactreduction.cpp
/// @brief  Unit tests for the coplanar contact normal reduction.
///
/// Validates merging of nearly-identical normals, preservation of distinct normals,
/// and determinism.
//==================================================================================================
#include "ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cmath>

using namespace ciphyxs;

static int g_tests = 0;
static int g_passed = 0;

#define TEST(name) do { ++g_tests; std::printf("  TEST %s ... ", name); } while(0)
#define PASS()   do { ++g_passed; std::puts("PASS"); } while(0)
#define FAIL(msg) do { std::printf("FAIL  (%s)\n", msg); return 1; } while(0)

// Helper: build a manifold with a given normal and N coplanar points.
static ContactManifold makeManifold(
    RigidBodyHandle bodyA, RigidBodyHandle bodyB,
    const Vec3f& normal,
    int pointCount,
    float friction = 0.5f,
    float restitution = 0.35f) {

    ContactManifold m;
    m.bodyA = bodyA;
    m.bodyB = bodyB;
    m.combinedFriction = friction;
    m.combinedRestitution = restitution;

    for (int i = 0; i < pointCount && i < 4; ++i) {
        ContactPoint pt;
        pt.position = Vec3f(static_cast<float>(i) * 0.1f, 0.0f, 0.0f);
        pt.normal = normal;
        pt.penetration = 0.05f + static_cast<float>(i) * 0.01f;
        pt.featureA = i;
        pt.featureB = i;
        m.addPoint(pt);
    }
    return m;
}

// ─── Tests ──────────────────────────────────────────────────────────────────────────────────────

static int testMergeCoplanar() {
    TEST("merge two manifolds with nearly-identical normals");
    std::vector<ContactManifold> manifolds;

    // Two points with normals differing by ~1 degree (dot ≈ 0.99985).
    Vec3f n0 = Vec3f(0.0f, 1.0f, 0.0f).normalized();
    Vec3f n1 = Vec3f(0.01745f, 0.99985f, 0.0f).normalized(); // ~1° tilt

    manifolds.push_back(makeManifold(0, 1, n0, 2, 0.5f, 0.3f));
    manifolds.push_back(makeManifold(0, 1, n1, 2, 0.5f, 0.3f));

    ContactReductionConfig cfg;
    cfg.normalCosineThreshold = 0.999f;
    reduceContactNormals(manifolds, cfg);

    // Should have merged into a single manifold.
    if (manifolds.size() != 1) {
        std::printf("    expected 1 manifold, got %zu\n", manifolds.size());
        FAIL("merge did not occur");
    }

    // Unified normal should be approximately (0, 1, 0).
    const auto& m = manifolds[0];
    float dotUp = m.points[0].normal.dot(Vec3f(0.0f, 1.0f, 0.0f));
    if (dotUp < 0.999f) {
        std::printf("    normal=(%.4f, %.4f, %.4f) dot_up=%.6f\n",
            m.points[0].normal.x, m.points[0].normal.y,
            m.points[0].normal.z, dotUp);
        FAIL("unified normal is not vertical");
    }

    // All points in the merged manifold should share the same normal.
    for (int i = 1; i < m.pointCount; ++i) {
        float d = m.points[0].normal.dot(m.points[i].normal);
        if (d < 0.9999f) {
            FAIL("not all points share the merged normal");
        }
    }

    // Friction/restitution should be preserved.
    if (std::abs(m.combinedFriction - 0.5f) > 1e-6f) FAIL("friction not preserved");
    if (std::abs(m.combinedRestitution - 0.3f) > 1e-6f) FAIL("restitution not preserved");

    PASS();
    return 0;
}

static int testKeepDistinct() {
    TEST("keep distinct normals separate");
    std::vector<ContactManifold> manifolds;

    Vec3f nUp   = Vec3f(0.0f, 1.0f, 0.0f);
    Vec3f nSide = Vec3f(1.0f, 0.0f, 0.0f); // perpendicular — not coplanar

    manifolds.push_back(makeManifold(0, 1, nUp, 2));
    manifolds.push_back(makeManifold(0, 1, nSide, 2));

    ContactReductionConfig cfg;
    cfg.normalCosineThreshold = 0.999f;
    reduceContactNormals(manifolds, cfg);

    // Should keep both manifolds separate (normals are perpendicular).
    if (manifolds.size() != 2) {
        std::printf("    expected 2 manifolds, got %zu\n", manifolds.size());
        FAIL("distinct normals were incorrectly merged");
    }

    PASS();
    return 0;
}

static int testSingleManifold() {
    TEST("single manifold is unchanged");
    std::vector<ContactManifold> manifolds;
    manifolds.push_back(makeManifold(0, 1, Vec3f(0.0f, 1.0f, 0.0f), 3));

    ContactReductionConfig cfg;
    reduceContactNormals(manifolds, cfg);

    if (manifolds.size() != 1) FAIL("should still have 1 manifold");
    if (manifolds[0].pointCount != 3) FAIL("points should be unchanged");

    PASS();
    return 0;
}

static int testDifferentBodyPairs() {
    TEST("different body pairs are not merged");
    std::vector<ContactManifold> manifolds;

    // Pair (0,1) and pair (0,2) — different body B.
    Vec3f n = Vec3f(0.0f, 1.0f, 0.0f);
    manifolds.push_back(makeManifold(0, 1, n, 2));
    manifolds.push_back(makeManifold(0, 2, n, 2));

    ContactReductionConfig cfg;
    cfg.normalCosineThreshold = 0.999f;
    reduceContactNormals(manifolds, cfg);

    if (manifolds.size() != 2) {
        FAIL("different body pairs were incorrectly merged");
    }

    PASS();
    return 0;
}

static int testDeterminism() {
    TEST("deterministic reduction");
    auto run = [](const ContactReductionConfig& cfg) -> std::size_t {
        std::vector<ContactManifold> manifolds;
        // Mixed normals: some coplanar, some distinct.
        Vec3f n0 = Vec3f(0.0f, 1.0f, 0.0f);
        Vec3f n1 = Vec3f(0.01f, 0.99995f, 0.0f).normalized();
        Vec3f n2 = Vec3f(0.0f, 0.0f, 1.0f);

        for (int i = 0; i < 3; ++i) {
            manifolds.push_back(makeManifold(0, 1, n0, 2));
            manifolds.push_back(makeManifold(0, 1, n1, 2));
            manifolds.push_back(makeManifold(0, 1, n2, 2));
            manifolds.push_back(makeManifold(1, 2, n0, 2));
        }

        reduceContactNormals(manifolds, cfg);
        return manifolds.size();
    };

    ContactReductionConfig cfg;
    cfg.normalCosineThreshold = 0.999f;

    std::size_t r1 = run(cfg);
    std::size_t r2 = run(cfg);
    if (r1 != r2) {
        std::printf("    run1=%zu run2=%zu\n", r1, r2);
        FAIL("non-deterministic reduction count");
    }

    PASS();
    return 0;
}

static int testBoxOnTrimeshScenario() {
    TEST("simulated box-on-triangle-mesh contact reduction");
    // Simulates a box resting on a flat triangulated surface.
    // The narrowphase would produce 3-4 manifolds with slightly different normals.
    std::vector<ContactManifold> manifolds;

    // Triangle 1: tiny tilt in +X direction
    Vec3f n1 = Vec3f(0.005f, 0.999987f, 0.0f).normalized();
    // Triangle 2: tiny tilt in -X direction
    Vec3f n2 = Vec3f(-0.003f, 0.999995f, 0.0f).normalized();
    // Triangle 3: tiny tilt in +Z direction
    Vec3f n3 = Vec3f(0.0f, 0.999987f, 0.005f).normalized();

    manifolds.push_back(makeManifold(0, 1, n1, 3, 0.6f, 0.3f));
    manifolds.push_back(makeManifold(0, 1, n2, 2, 0.6f, 0.3f));
    manifolds.push_back(makeManifold(0, 1, n3, 4, 0.6f, 0.3f));

    ContactReductionConfig cfg;
    cfg.normalCosineThreshold = 0.999f;
    reduceContactNormals(manifolds, cfg);

    // All three should merge into one manifold.
    if (manifolds.size() != 1) {
        std::printf("    expected 1 manifold, got %zu\n", manifolds.size());
        FAIL("box-on-trimesh scenario did not reduce to single manifold");
    }

    const auto& m = manifolds[0];
    // The unified normal should be very close to (0,1,0).
    float dotUp = m.points[0].normal.dot(Vec3f(0.0f, 1.0f, 0.0f));
    if (dotUp < 0.9999f) {
        std::printf("    unified normal=(%.6f, %.6f, %.6f) dot_up=%.8f\n",
            m.points[0].normal.x, m.points[0].normal.y,
            m.points[0].normal.z, dotUp);
        FAIL("unified normal deviates too much from vertical");
    }

    // Should have up to 4 points (the deepest ones from all input manifolds).
    if (m.pointCount < 1 || m.pointCount > 4) {
        std::printf("    pointCount=%d (expected 1-4)\n", m.pointCount);
        FAIL("unexpected point count");
    }

    PASS();
    return 0;
}

// ─── main ───────────────────────────────────────────────────────────────────────────────────────

int main() {
    std::puts("=== Contact Reduction Tests ===");

    int failures = 0;
    failures += testMergeCoplanar();
    failures += testKeepDistinct();
    failures += testSingleManifold();
    failures += testDifferentBodyPairs();
    failures += testDeterminism();
    failures += testBoxOnTrimeshScenario();

    std::printf("\n%d / %d tests passed\n", g_passed, g_tests);
    return failures;
}
