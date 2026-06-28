//==================================================================================================
/// @file  test_alignment.cpp
/// @brief  Validate that AlignedAllocator, AlignedVector, and SoA fields are correctly aligned.
///
/// Run with:
///   g++ -std=c++20 -O2 -I. test_alignment.cpp -o test_alignment && ./test_alignment
//==================================================================================================
#include "include/ciphyxs/CiPhyxs.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <new>

// ── Helpers ──────────────────────────────────────────────────────────────────────────────────────

/// @brief  Check if `ptr` is aligned to `alignment` bytes.
bool isAligned(const void* ptr, std::size_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/// @brief  Count failing conditions and return total failures.
int g_failures = 0;
int g_tests    = 0;

#define CHECK(cond, msg) do { \
    ++g_tests; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        ++g_failures; \
    } \
} while(0)

// ── Test 1: AlignedAllocator basic allocation ────────────────────────────────────────────────────

void testAlignedAllocator() {
    printf("── Test 1: AlignedAllocator basic allocation ──\n");

    {
        // Allocate 100 floats at 16-byte alignment.
        ciphyxs::AlignedAllocator<float, 16> alloc;
        float* ptr = alloc.allocate(100);
        CHECK(isAligned(ptr, 16), "float* should be 16-byte aligned");
        // Write to the buffer to verify it's usable.
        std::memset(ptr, 0, 100 * sizeof(float));
        ptr[0] = 42.0f;
        CHECK(ptr[0] == 42.0f, "write/read should work");
        alloc.deallocate(ptr, 100);
        printf("  PASS: float allocation aligned at %p\n", (void*)ptr);
    }
    {
        // Allocate Vec3f (alignas(16)) at 16-byte alignment.
        ciphyxs::AlignedAllocator<ciphyxs::Vec3f, 16> alloc;
        auto* ptr = alloc.allocate(50);
        CHECK(isAligned(ptr, 16), "Vec3f* should be 16-byte aligned");
        CHECK(isAligned(ptr, alignof(ciphyxs::Vec3f)),
              "Vec3f* should satisfy alignof(Vec3f)");
        alloc.deallocate(ptr, 50);
        printf("  PASS: Vec3f allocation aligned at %p\n", (void*)ptr);
    }
    {
        // Allocate with 32-byte alignment (for AVX).
        ciphyxs::AlignedAllocator<double, 32> alloc;
        double* ptr = alloc.allocate(256);
        CHECK(isAligned(ptr, 32), "double* should be 32-byte aligned");
        alloc.deallocate(ptr, 256);
        printf("  PASS: double (32-byte) allocation aligned at %p\n", (void*)ptr);
    }
    {
        // Edge case: zero-size allocation should not crash.
        ciphyxs::AlignedAllocator<float, 16> alloc;
        float* ptr = alloc.allocate(0);
        CHECK(ptr == nullptr, "zero-size allocation should return nullptr");
        alloc.deallocate(ptr, 0);
        printf("  PASS: zero-size returns nullptr\n");
    }
}

// ── Test 2: AlignedVector ────────────────────────────────────────────────────────────────────────

void testAlignedVector() {
    printf("\n── Test 2: AlignedVector ──\n");

    ciphyxs::AlignedVector<ciphyxs::Vec3f, 16> vec;
    CHECK(vec.empty(), "fresh AlignedVector should be empty");

    // Push elements and check alignment after growth.
    for (int i = 0; i < 100; ++i) {
        vec.emplace_back(static_cast<float>(i), 0.0f, 0.0f);
        if (!isAligned(vec.data(), 16)) {
            printf("  FAIL: data() lost alignment at size %zu\n", vec.size());
            ++g_failures;
            break;
        }
    }
    CHECK(isAligned(vec.data(), 16), "data() should remain aligned after 100 push_backs");
    CHECK(vec.size() == 100, "size should be 100");

    // Verify data integrity.
    bool ok = true;
    for (int i = 0; i < 100; ++i) {
        if (vec[i].x != static_cast<float>(i)) { ok = false; break; }
    }
    CHECK(ok, "element values should be intact");

    // Reserve larger capacity.
    vec.reserve(500);
    CHECK(isAligned(vec.data(), 16), "data() should remain aligned after reserve(500)");
    CHECK(vec.capacity() >= 500, "capacity should be >= 500");

    // Shrink to fit.
    vec.shrink_to_fit();
    CHECK(isAligned(vec.data(), 16), "data() should remain aligned after shrink_to_fit");

    printf("  PASS: AlignedVector used %zu elements, capacity %zu, aligned at %p\n",
           vec.size(), vec.capacity(), (void*)vec.data());
}

// ── Test 3: RigidBodyStorage SoA alignment ────────────────────────────────────────────────────

void testSoAAlignment() {
    printf("\n── Test 3: RigidBodyStorage SoA alignment ──\n");

    ciphyxs::PhysicsWorld world;

    // Create bodies to populate the SoA storage (at least one body triggers allocation).
    auto hBox = world.createShape(ciphyxs::Box{ciphyxs::Vec3f(0.5f, 0.5f, 0.5f)});
    ciphyxs::RigidBodyDesc desc;
    desc.mass    = 1.0f;
    desc.setShape(hBox);
    for (int i = 0; i < 10; ++i) {
        desc.position = ciphyxs::Vec3f(0.0f, static_cast<float>(i) * 2.0f + 5.0f, 0.0f);
        world.createBody(desc);
    }

    const auto& bodies = world.bodies();

    CHECK(isAligned(bodies.positions.data(), 16),
          "bodies.positions should be 16-byte aligned");
    CHECK(isAligned(bodies.linearVelocities.data(), 16),
          "bodies.linearVelocities should be 16-byte aligned");
    CHECK(isAligned(bodies.angularVelocities.data(), 16),
          "bodies.angularVelocities should be 16-byte aligned");
    CHECK(isAligned(bodies.forces.data(), 16),
          "bodies.forces should be 16-byte aligned");
    CHECK(isAligned(bodies.torques.data(), 16),
          "bodies.torques should be 16-byte aligned");
    CHECK(isAligned(bodies.inverseMasses.data(), 16),
          "bodies.inverseMasses should be 16-byte aligned");
    CHECK(isAligned(bodies.inverseInertiaDiag.data(), 16),
          "bodies.inverseInertiaDiag should be 16-byte aligned");
    CHECK(isAligned(bodies.restitutions.data(), 16),
          "bodies.restitutions should be 16-byte aligned");
    CHECK(isAligned(bodies.frictions.data(), 16),
          "bodies.frictions should be 16-byte aligned");
    CHECK(isAligned(bodies.linearDamping.data(), 16),
          "bodies.linearDamping should be 16-byte aligned");
    CHECK(isAligned(bodies.angularDamping.data(), 16),
          "bodies.angularDamping should be 16-byte aligned");
    CHECK(isAligned(bodies.sleepTimers.data(), 16),
          "bodies.sleepTimers should be 16-byte aligned");

    printf("  PASS: All SoA vectors are 16-byte aligned\n");
}

// ── Test 4: aligned_ptr helper ─────────────────────────────────────────────────────────────────

void testAlignedPtr() {
    printf("\n── Test 4: aligned_ptr helper ──\n");

    ciphyxs::AlignedVector<float, 16> vec(100, 0.0f);
    float* ptr = ciphyxs::aligned_ptr<16>(vec);
    CHECK(ptr == vec.data(), "aligned_ptr should return data()");
    CHECK(isAligned(ptr, 16), "aligned_ptr result should be 16-byte aligned");
    printf("  PASS: aligned_ptr returns %p\n", (void*)ptr);
}

// ── Test 5: Relocation (swap/pop) preserves alignment ─────────────────────────────────────────

void testRelocation() {
    printf("\n── Test 5: Removal + relocation ──\n");

    ciphyxs::PhysicsWorld world;
    auto hBox = world.createShape(ciphyxs::Box{ciphyxs::Vec3f(0.5f, 0.5f, 0.5f)});

    ciphyxs::RigidBodyDesc desc;
    desc.mass    = 1.0f;
    desc.setShape(hBox);
    auto h0 = world.createBody(desc); (void)h0;

    desc.position = ciphyxs::Vec3f(0, 2, 0);
    auto h1 = world.createBody(desc);
    desc.position = ciphyxs::Vec3f(0, 4, 0);
    auto h2 = world.createBody(desc); (void)h2;

    // Simulate a few steps to populate velocities etc.
    for (int i = 0; i < 10; ++i) world.step(1.0f / 60.0f);

    // Remove body h1 (middle) — this triggers swap-pop.
    world.removeBody(h1);

    const auto& bodies = world.bodies();
    CHECK(isAligned(bodies.positions.data(), 16),
          "positions still aligned after remove");
    CHECK(isAligned(bodies.linearVelocities.data(), 16),
          "linearVelocities still aligned after remove");
    CHECK(isAligned(bodies.inverseMasses.data(), 16),
          "inverseMasses still aligned after remove");

    printf("  PASS: Alignment preserved after body removal (swap-pop)\n");
}

// ── Main ─────────────────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║   CiPhyxs — AlignedAllocator & Memory Alignment   ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");

    testAlignedAllocator();
    testAlignedVector();
    testSoAAlignment();
    testAlignedPtr();
    testRelocation();

    printf("\n── Results ──\n");
    printf("  Tests:  %d\n", g_tests);
    printf("  Passed: %d\n", g_tests - g_failures);
    printf("  Failed: %d\n", g_failures);

    return g_failures > 0 ? 1 : 0;
}
