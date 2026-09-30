#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Config/ConfigParam.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <thread>

using Json = nlohmann::ordered_json;

struct IntegerConfig
{
    int count = 7;
    uint64_t wide = 9;
    int custom = 11;
    DNDS_DECLARE_CONFIG(IntegerConfig)
    {
        DNDS_FIELD(count, "count", DNDS::Config::range(0));
        DNDS_FIELD(wide, "wide");
        config.field_schema(&T::custom, "custom", "custom", []
                            { return Json{{"type", "integer"}}; });
    }
};

TEST_CASE("Audit batch 2: config integer conversion rejects invalid values")
{
    for (const char *key : {"count", "custom"})
        for (const Json &bad : {Json(2147483648LL), Json(-2147483649LL), Json(1.5), Json("2"), Json(1e30)})
        {
            IntegerConfig value;
            Json input = value;
            input[key] = bad;
            CHECK_THROWS(from_json(input, value));
            CHECK(value.count == 7);
            CHECK(value.custom == 11);
        }
    for (const Json &bad : {Json(-1), Json(18446744073709551616.0), Json(0.5)})
    {
        IntegerConfig value;
        Json input = value;
        input["wide"] = bad;
        CHECK_THROWS(from_json(input, value));
        CHECK(value.wide == 9);
    }
    IntegerConfig value;
    Json input = value;
    input["count"] = std::numeric_limits<int>::max();
    input["wide"] = std::numeric_limits<uint64_t>::max();
    input["custom"] = std::numeric_limits<int>::min();
    CHECK_NOTHROW(from_json(input, value));
    CHECK(value.count == std::numeric_limits<int>::max());
    CHECK(value.wide == std::numeric_limits<uint64_t>::max());
    CHECK(value.custom == std::numeric_limits<int>::min());
    input["count"] = 2.0;
    CHECK_NOTHROW(from_json(input, value));
    CHECK(value.count == 2);
    // Existing Euler inputs use true/false for integer mode fields such as
    // meshReorderCells. Preserve that exact, representable 0/1 conversion.
    for (bool legacy : {false, true})
    {
        input["count"] = legacy;
        input["custom"] = legacy;
        CHECK_NOTHROW(from_json(input, value));
        CHECK(value.count == int(legacy));
        CHECK(value.wide == std::numeric_limits<uint64_t>::max());
        CHECK(value.custom == int(legacy));
    }
}

static std::atomic<bool> registrationEntered{false}, releaseRegistration{false};
struct ConcurrentConfig
{
    int count = 0;
    DNDS_DECLARE_CONFIG(ConcurrentConfig)
    {
        registrationEntered = true;
        while (!releaseRegistration.load())
            std::this_thread::yield();
        DNDS_FIELD(count, "count");
    }
};

TEST_CASE("Audit batch 2: registration publishes only complete metadata")
{
    auto writer = std::async(std::launch::async, []
                             { return ConcurrentConfig::schema(); });
    while (!registrationEntered.load())
        std::this_thread::yield();
    std::atomic<bool> readerEntered{false};
    auto reader = std::async(std::launch::async, [&]
                             {
        readerEntered = true;
        return ConcurrentConfig::schema(); });
    while (!readerEntered.load())
        std::this_thread::yield();
    auto beforeRelease = reader.wait_for(std::chrono::milliseconds(30));
    releaseRegistration = true;
    auto first = writer.get();
    auto second = reader.get();
    CHECK(beforeRelease == std::future_status::timeout);
    CHECK(first["properties"].contains("count"));
    CHECK(second == first);
}

struct RetryConfig
{
    int first = 1, second = 2;
    DNDS_DECLARE_CONFIG(RetryConfig)
    {
        DNDS_FIELD(first, "first");
        static int attempts = 0;
        if (++attempts == 1)
            throw std::runtime_error("registration test interruption");
        DNDS_FIELD(second, "second");
    }
};

TEST_CASE("Audit batch 2: failed registration retries without partial fields")
{
    CHECK_THROWS(RetryConfig::schema());
    const auto schema = RetryConfig::schema();
    CHECK(schema["properties"].size() == 2);
    CHECK(DNDS::ConfigRegistry<RetryConfig>::fields().size() == 2);
    CHECK(schema["properties"].contains("second"));
}

struct RecursiveConfig
{
    int count = 0;
    DNDS_DECLARE_CONFIG(RecursiveConfig)
    {
        DNDS_FIELD(count, "count");
        T::schema();
    }
};

TEST_CASE("Config registration rejects same-type recursion")
{
    CHECK_THROWS_AS(RecursiveConfig::schema(), std::runtime_error);
    CHECK(DNDS::ConfigRegistry<RecursiveConfig>::fields().empty());
}
